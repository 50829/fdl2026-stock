#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seeded 100-stock core/tail strategy using model pred. "
            "Top30 gets 90%; tail70 is selected by historical risk first, then model score."
        )
    )
    parser.add_argument(
        "--input",
        default="outputs/models/sdd_final_model_handoff/test/test_pred.parquet",
        help="Prediction parquet with trade_date, ts_code, pred, and label_1d.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/strategy/hs300_risk_balanced_tail_seeded_final",
    )
    parser.add_argument("--score-col", default="pred")
    parser.add_argument("--return-col", default="label_1d")
    parser.add_argument("--stages", default="10,100,300")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--risk-window", type=int, default=60)
    parser.add_argument("--tail-risk-candidates", type=int, default=140)
    parser.add_argument("--core-count", type=int, default=30)
    parser.add_argument("--tail-count", type=int, default=70)
    parser.add_argument("--core-weight", type=float, default=0.9)
    parser.add_argument("--max-stock-updates", type=int, default=25)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_prediction_panel(path: str | Path, score_col: str, return_col: str) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    if score_col == return_col or str(score_col).startswith("label_"):
        raise ValueError(
            f"score_col={score_col!r} would use realized label/return data as the selection signal"
        )
    df = pd.read_parquet(path)
    required = {"trade_date", "ts_code", score_col, return_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df[["trade_date", "ts_code", score_col, return_col]].copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    df[score_col] = df[score_col].astype(float)
    df[return_col] = df[return_col].astype(float)
    df = df.dropna(subset=[score_col, return_col])
    score_panel = (
        df.pivot_table(index="trade_date", columns="ts_code", values=score_col, aggfunc="first")
        .sort_index()
    )
    return_panel = (
        df.pivot_table(index="trade_date", columns="ts_code", values=return_col, aggfunc="first")
        .reindex(score_panel.index)
        .reindex(columns=score_panel.columns)
        .fillna(0.0)
    )
    return list(score_panel.index), score_panel, return_panel


def random_equal_weights(universe: pd.Index, n_holdings: int, seed: int) -> dict[str, float]:
    available = sorted(str(c) for c in universe)
    if len(available) < int(n_holdings):
        raise ValueError(f"initial universe has only {len(available)} stocks, need {n_holdings}")
    codes = random.Random(seed).sample(available, int(n_holdings))
    return {code: 1.0 / int(n_holdings) for code in codes}


def portfolio_return(weights: dict[str, float], returns_row: pd.Series) -> float:
    return float(sum(float(weight) * float(returns_row.get(code, 0.0)) for code, weight in weights.items()))


def turnover(prev: dict[str, float], new: dict[str, float]) -> float:
    codes = set(prev) | set(new)
    return float(sum(abs(float(new.get(c, 0.0)) - float(prev.get(c, 0.0))) for c in codes))


def percentile_rank(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    return values.rank(method="average", pct=True).fillna(0.5)


def percentile_rank_array(values: np.ndarray, index: list[str]) -> pd.Series:
    clean = np.asarray(values, dtype=float)
    if clean.size == 0:
        return pd.Series(dtype=float)
    if np.isnan(clean).all():
        return pd.Series(0.5, index=index)
    median = float(np.nanmedian(clean))
    clean = np.nan_to_num(clean, nan=median, posinf=median, neginf=median)
    order = np.argsort(clean, kind="mergesort")
    ranks = np.empty(clean.size, dtype=float)
    sorted_values = clean[order]
    start = 0
    while start < clean.size:
        end = start + 1
        while end < clean.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank / clean.size
        start = end
    return pd.Series(ranks, index=index)


def risk_index(
    returns_panel: pd.DataFrame,
    pos: int,
    core: list[str],
    candidate_codes: list[str],
    risk_window: int,
) -> pd.Series:
    hist = returns_panel.iloc[max(0, pos - int(risk_window)) : pos]
    if len(hist) < 2 or not core or not candidate_codes:
        return pd.Series(0.5, index=candidate_codes)

    core_cols = [c for c in core if c in hist.columns]
    cand_cols = [c for c in candidate_codes if c in hist.columns]
    if not core_cols or not cand_cols:
        return pd.Series(0.5, index=candidate_codes)

    core_ret = hist[core_cols].mean(axis=1).to_numpy(dtype=float)
    candidate_hist = hist[cand_cols].to_numpy(dtype=float)
    vol = np.nanstd(candidate_hist, axis=0, ddof=1)
    downside = np.nanstd(np.minimum(candidate_hist, 0.0), axis=0, ddof=1)

    x = candidate_hist - np.nanmean(candidate_hist, axis=0, keepdims=True)
    y = core_ret - float(np.nanmean(core_ret))
    numerator = np.nansum(x * y[:, None], axis=0)
    denominator = np.sqrt(np.nansum(x * x, axis=0) * np.nansum(y * y))
    corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    risk = (
        0.5 * percentile_rank_array(corr, cand_cols)
        + 0.3 * percentile_rank_array(vol, cand_cols)
        + 0.2 * percentile_rank_array(downside, cand_cols)
    )
    return risk.reindex(candidate_codes).fillna(0.5)


def allocate_core_tail_weights(codes: set[str], core: list[str], args: argparse.Namespace) -> dict[str, float]:
    current_core = [c for c in core if c in codes]
    current_tail = sorted(codes - set(current_core))
    weights: dict[str, float] = {}
    if current_core and current_tail:
        core_total = float(args.core_weight)
        tail_total = 1.0 - core_total
    elif current_core:
        core_total = 1.0
        tail_total = 0.0
    else:
        core_total = 0.0
        tail_total = 1.0

    if current_core:
        core_w = core_total / len(current_core)
        weights.update({c: core_w for c in current_core})
    if current_tail:
        tail_w = tail_total / len(current_tail)
        weights.update({c: tail_w for c in current_tail})
    return weights


def build_target(
    scores_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    pos: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[str], list[str], pd.Series]:
    scores = scores_panel.iloc[pos].dropna().sort_values(ascending=False)
    core = [str(c) for c in scores.head(args.core_count).index]
    candidate_codes = [str(c) for c in scores.index if str(c) not in set(core)]
    risk = risk_index(returns_panel, pos, core, candidate_codes, args.risk_window)

    if len(returns_panel.iloc[max(0, pos - int(args.risk_window)) : pos]) >= 2:
        risk_pool = list(risk.sort_values(ascending=True).head(min(args.tail_risk_candidates, len(risk))).index)
    else:
        # Before risk history is available, use model score to form the candidate pool.
        risk_pool = candidate_codes[: min(args.tail_risk_candidates, len(candidate_codes))]
    tail = [str(c) for c in scores.reindex(risk_pool).dropna().sort_values(ascending=False).head(args.tail_count).index]
    target_codes = set(core) | set(tail)
    weights = allocate_core_tail_weights(target_codes, core, args)
    return weights, core, tail, risk


def precompute_targets(
    scores_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    max_stage_days: int,
    args: argparse.Namespace,
) -> dict[int, tuple[dict[str, float], list[str], list[str], pd.Series]]:
    targets: dict[int, tuple[dict[str, float], list[str], list[str], pd.Series]] = {}
    for pos in range(1, max_stage_days):
        targets[pos] = build_target(scores_panel, returns_panel, pos, args)
    return targets


def rebalance_with_stock_limit(
    current: dict[str, float],
    target_weights: dict[str, float],
    target_core: list[str],
    scores: pd.Series,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[str], list[str]]:
    current_codes = set(current)
    target_codes = set(target_weights)
    additions = [str(c) for c in scores.reindex(list(target_codes - current_codes)).dropna().sort_values(ascending=False).index]
    removals = [str(c) for c in scores.reindex(list(current_codes - target_codes)).dropna().sort_values(ascending=True).index]

    sell_count = min(int(args.max_stock_updates), len(removals))
    next_codes = current_codes - set(removals[:sell_count])
    buy_count = min(len(additions), max(0, len(target_codes) - len(next_codes)))
    next_codes |= set(additions[:buy_count])

    # If some current holdings disappeared from today's score universe, replace them.
    missing = [c for c in next_codes if c not in scores.index]
    if missing:
        next_codes -= set(missing)
        fallback = [str(c) for c in scores.index if str(c) not in next_codes]
        next_codes |= set(fallback[: len(missing)])

    weights = allocate_core_tail_weights(next_codes, target_core, args)
    return weights, additions[:buy_count], removals[:sell_count]


def max_drawdown(equity_values: list[float]) -> float:
    peak = equity_values[0]
    worst = 0.0
    for value in equity_values:
        peak = max(peak, value)
        worst = min(worst, value / (peak + 1e-12) - 1.0)
    return float(worst)


def run_seed(
    seed: int,
    stage_days: int,
    dates: list[str],
    scores_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    target_cache: dict[int, tuple[dict[str, float], list[str], list[str], pd.Series]],
    args: argparse.Namespace,
) -> tuple[dict[str, float | int | str], list[dict[str, object]]]:
    n_holdings = int(args.core_count + args.tail_count)
    initial_universe = scores_panel.iloc[0].dropna().index
    current = random_equal_weights(initial_universe, n_holdings, seed)
    initial_holdings = ";".join(sorted(current))
    equity = 1.0
    equity_curve = [equity]
    daily_rows: list[dict[str, object]] = []
    prev_weights = dict(current)

    for pos in range(stage_days):
        date = dates[pos]
        if pos > 0:
            target_weights, core, tail, risk = target_cache[pos]
            current, added, removed = rebalance_with_stock_limit(
                current,
                target_weights,
                core,
                scores_panel.iloc[pos],
                args,
            )
        else:
            core = []
            tail = []
            risk = pd.Series(dtype=float)
            added = []
            removed = []

        gross_ret = portfolio_return(current, returns_panel.iloc[pos])
        daily_turnover = turnover(prev_weights, current)
        cost = daily_turnover * float(args.transaction_cost_bps) / 10000.0
        net_ret = gross_ret - cost
        equity *= 1.0 + net_ret
        equity_curve.append(equity)
        prev_weights = dict(current)

        daily_rows.append(
            {
                "stage_days": stage_days,
                "seed": seed,
                "trade_date": date,
                "gross_return": gross_ret,
                "transaction_cost": cost,
                "portfolio_return": net_ret,
                "equity": equity,
                "n_holdings": len(current),
                "turnover": daily_turnover,
                "n_added": len(added),
                "n_removed": len(removed),
                "initial_random_holdings": initial_holdings if pos == 0 else "",
                "core_top30": ";".join(core),
                "tail70": ";".join(tail),
                "avg_tail_risk_index": float(risk.reindex(tail).mean()) if tail else math.nan,
                "holdings": ";".join(f"{code}:{weight:.8f}" for code, weight in sorted(current.items())),
            }
        )

    returns = [float(row["portfolio_return"]) for row in daily_rows]
    daily_vol = float(pd.Series(returns).std(ddof=1)) if len(returns) > 1 else 0.0
    annual_vol = daily_vol * math.sqrt(252)
    annual_return = equity ** (252 / len(returns)) - 1.0 if returns else 0.0
    summary = {
        "stage_days": stage_days,
        "seed": seed,
        "start_date": daily_rows[0]["trade_date"],
        "end_date": daily_rows[-1]["trade_date"],
        "n_return_days": len(daily_rows),
        "final_equity": equity,
        "total_return": equity - 1.0,
        "avg_daily_return": sum(returns) / len(returns),
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe_no_risk_free": annual_return / annual_vol if annual_vol > 0 else 0.0,
        "max_drawdown": max_drawdown(equity_curve),
        "avg_turnover": float(pd.Series([row["turnover"] for row in daily_rows]).mean()),
        "avg_n_holdings": float(pd.Series([row["n_holdings"] for row in daily_rows]).mean()),
    }
    return summary, daily_rows


def aggregate_summary(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(summary_rows)
    rows: list[dict[str, object]] = []
    for stage_days, sub in df.groupby("stage_days", sort=True):
        rows.append(
            {
                "stage_days": int(stage_days),
                "n_seeds": int(len(sub)),
                "final_equity_mean": float(sub["final_equity"].mean()),
                "final_equity_min": float(sub["final_equity"].min()),
                "final_equity_max": float(sub["final_equity"].max()),
                "total_return_mean": float(sub["total_return"].mean()),
                "sharpe_mean": float(sub["sharpe_no_risk_free"].mean()),
                "max_drawdown_mean": float(sub["max_drawdown"].mean()),
                "annualized_volatility_mean": float(sub["annualized_volatility"].mean()),
                "avg_turnover_mean": float(sub["avg_turnover"].mean()),
            }
        )
    return rows


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_equity_svg(daily_rows: list[dict[str, object]], stage_days: int, output_dir: Path) -> None:
    df = pd.DataFrame([row for row in daily_rows if int(row["stage_days"]) == int(stage_days)])
    if df.empty:
        return
    width, height = 1100, 620
    left, right, top, bottom = 80, 150, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    dates = sorted(df["trade_date"].astype(str).unique().tolist())
    date_to_x = {d: i for i, d in enumerate(dates)}
    y_min = min(1.0, float(df["equity"].min()))
    y_max = max(1.0, float(df["equity"].max()))
    pad = max(1e-6, (y_max - y_min) * 0.05)
    y_min -= pad
    y_max += pad
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    def sx(date: str) -> float:
        return left + date_to_x[str(date)] / max(1, len(dates) - 1) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - float(value)) / (y_max - y_min) * plot_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{left}' y='30' font-size='20' font-family='Arial'>Seeded risk-balanced tail, {stage_days} days</text>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#333'/>",
    ]
    for i in range(6):
        y = top + i / 5 * plot_h
        val = y_max - i / 5 * (y_max - y_min)
        parts.append(f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' stroke='#ddd'/>")
        parts.append(f"<text x='{left - 8}' y='{y + 4:.2f}' text-anchor='end' font-size='11' font-family='Arial'>{val:.2f}</text>")
    for i, (seed, sub) in enumerate(df.groupby("seed", sort=True)):
        color = colors[i % len(colors)]
        points = " ".join(f"{sx(str(d)):.2f},{sy(float(e)):.2f}" for d, e in zip(sub["trade_date"], sub["equity"]))
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='1.3' opacity='0.7' points='{points}'/>")
        ly = top + 20 + i * 20
        parts.append(f"<text x='{left + plot_w + 20}' y='{ly}' font-size='11' font-family='Arial' fill='{color}'>seed={seed}</text>")
    tick_idx = np.linspace(0, len(dates) - 1, num=min(6, len(dates)), dtype=int)
    for idx in tick_idx:
        x = left + idx / max(1, len(dates) - 1) * plot_w
        parts.append(f"<text x='{x:.2f}' y='{top + plot_h + 24}' text-anchor='middle' font-size='11' font-family='Arial'>{dates[idx]}</text>")
    parts.append("</svg>")
    (output_dir / f"equity_stage_{stage_days}d.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = parse_ints(args.stages)
    seeds = parse_ints(args.seeds)
    dates, scores_panel, returns_panel = load_prediction_panel(args.input, args.score_col, args.return_col)

    if max(stages) > len(dates):
        raise ValueError(f"requested max stage {max(stages)} exceeds available dates {len(dates)}")

    target_cache = precompute_targets(scores_panel, returns_panel, max(stages), args)
    summaries: list[dict[str, object]] = []
    all_daily: list[dict[str, object]] = []
    for stage_days in stages:
        for seed in seeds:
            summary, daily_rows = run_seed(seed, stage_days, dates, scores_panel, returns_panel, target_cache, args)
            summaries.append(summary)
            all_daily.extend(daily_rows)

    write_csv(output_dir / "summary_by_seed.csv", summaries)
    write_csv(output_dir / "summary_aggregate.csv", aggregate_summary(summaries))
    write_csv(output_dir / "daily_performance_by_seed.csv", all_daily)
    for stage_days in stages:
        write_equity_svg(all_daily, stage_days, output_dir)

    meta = {
        "input": str(args.input),
        "score_col": args.score_col,
        "return_col": args.return_col,
        "note": "Selection uses model pred only. label_1d is used for historical risk and ex-post returns.",
        "args": vars(args),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
