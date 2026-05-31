from __future__ import annotations

import json
import math
import zipfile
from html import escape
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyBacktestConfig:
    strategy: str
    score_col: str = "pred"
    return_col: str = "label_1d"
    transaction_cost_bps: float = 5.0
    trading_days_per_year: int = 252
    target_positions: int = 20
    hold_days: int = 5
    daily_buy: int = 0
    topk: int = 20
    drop: int = 2
    buy_rank: int = 30
    sell_rank: int = 100
    min_hold_days: int = 2
    max_hold_days: int = 10
    core_count: int = 30
    tail_count: int = 70
    core_weight: float = 0.9
    tail_risk_candidates: int = 140
    risk_window: int = 60
    max_stock_updates: int = 25
    risk_candidate_count: int = 150
    risk_keep_count: int = 80


def load_prediction_data(path: str | Path, score_col: str = "pred", return_col: str = "label_1d") -> pd.DataFrame:
    if score_col == return_col or str(score_col).startswith("label_"):
        raise ValueError(
            f"score_col={score_col!r} would use realized label/return data as the selection signal"
        )
    df = pd.read_parquet(path)
    required = {"trade_date", "ts_code", score_col, return_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df[["trade_date", "ts_code", score_col, return_col]].copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["ts_code"] = out["ts_code"].astype(str)
    out[score_col] = out[score_col].astype("float32")
    out[return_col] = out[return_col].astype("float32")
    out = out.dropna(subset=[score_col, return_col])
    return out.sort_values(["trade_date", score_col], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return math.nan
    peak = np.maximum.accumulate(equity)
    return float((equity / (peak + 1e-12) - 1.0).min())


def _sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    if len(returns) < 2:
        return math.nan
    sd = float(np.std(returns, ddof=1))
    if sd <= 0:
        return math.nan
    return float(np.mean(returns) / sd * math.sqrt(periods_per_year))


def _equal_weights(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {c: w for c in codes}


def _score_weights(scores: pd.Series, total_weight: float) -> dict[str, float]:
    if scores.empty or total_weight <= 0:
        return {}
    ranks = scores.rank(method="average", pct=True).astype(float).clip(lower=0.0) + 1e-6
    denom = float(ranks.sum())
    if denom <= 0:
        return {str(c): total_weight / len(ranks) for c in ranks.index}
    return {str(c): total_weight * float(v) / denom for c, v in ranks.items()}


def _turnover(prev: dict[str, float], new: dict[str, float]) -> float:
    codes = set(prev) | set(new)
    return float(sum(abs(float(new.get(c, 0.0)) - float(prev.get(c, 0.0))) for c in codes))


def _prepare_maps(df: pd.DataFrame, cfg: StrategyBacktestConfig) -> tuple[list[str], dict[str, pd.DataFrame], pd.DataFrame]:
    dates = sorted(df["trade_date"].unique().tolist())
    day_map: dict[str, pd.DataFrame] = {}
    rows = []
    for d, g in df.groupby("trade_date", sort=True):
        day = g.sort_values(cfg.score_col, ascending=False, kind="mergesort").copy()
        day["rank"] = np.arange(1, len(day) + 1, dtype=np.int32)
        day = day.set_index("ts_code", drop=False)
        day_map[str(d)] = day
        rows.append(day[["trade_date", "ts_code", cfg.return_col]].reset_index(drop=True))
    ret_panel = pd.concat(rows, ignore_index=True).pivot(index="trade_date", columns="ts_code", values=cfg.return_col).sort_index()
    return dates, day_map, ret_panel


def _top_codes(day: pd.DataFrame, n: int, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    return [str(c) for c in day.index if str(c) not in exclude][: max(0, int(n))]


def _drop_missing(holdings: dict[str, int], day: pd.DataFrame) -> dict[str, int]:
    available = set(str(c) for c in day.index)
    return {c: age for c, age in holdings.items() if c in available}


def _rolling_tranche(holdings: dict[str, int], day: pd.DataFrame, cfg: StrategyBacktestConfig) -> tuple[dict[str, int], list[dict[str, Any]]]:
    next_holdings = {c: age for c, age in _drop_missing(holdings, day).items() if age < cfg.hold_days}
    sold = sorted(set(holdings) - set(next_holdings))
    daily_buy = int(cfg.daily_buy or max(1, round(cfg.target_positions / max(1, cfg.hold_days))))
    regular_slots = max(0, min(daily_buy, cfg.target_positions - len(next_holdings)))
    buys = _top_codes(day, regular_slots, exclude=set(next_holdings))
    for code in buys:
        next_holdings[code] = 0
    refill_slots = max(0, cfg.target_positions - len(next_holdings))
    refills = _top_codes(day, refill_slots, exclude=set(next_holdings))
    for code in refills:
        next_holdings[code] = 0
    trades = [{"action": "sell", "ts_code": c, "reason": "expired_or_missing"} for c in sold]
    trades.extend({"action": "buy", "ts_code": c, "reason": "daily_tranche"} for c in buys)
    trades.extend({"action": "buy", "ts_code": c, "reason": "target_refill"} for c in refills)
    return next_holdings, trades


def _topk_drop(holdings: dict[str, int], day: pd.DataFrame, cfg: StrategyBacktestConfig) -> tuple[dict[str, int], list[dict[str, Any]]]:
    current = _drop_missing(holdings, day)
    if not current:
        buys = _top_codes(day, cfg.topk)
        return {c: 0 for c in buys}, [{"action": "buy", "ts_code": c, "reason": "initial_topk"} for c in buys]
    current_codes = list(current)
    held_rank = day.reindex(current_codes)["rank"].fillna(len(day) + 1).sort_values(ascending=False)
    sells = [str(c) for c in held_rank.head(min(cfg.drop, len(held_rank))).index]
    after_sell = {c: age for c, age in current.items() if c not in set(sells)}
    need = max(0, cfg.topk - len(after_sell))
    buys = _top_codes(day, need, exclude=set(after_sell))
    next_holdings = dict(after_sell)
    for code in buys:
        next_holdings[code] = 0
    trades = [{"action": "sell", "ts_code": c, "reason": "drop_worst_rank"} for c in sells]
    trades.extend({"action": "buy", "ts_code": c, "reason": "topk_refill"} for c in buys)
    return next_holdings, trades


def _rank_buffer(holdings: dict[str, int], day: pd.DataFrame, cfg: StrategyBacktestConfig) -> tuple[dict[str, int], list[dict[str, Any]]]:
    current = _drop_missing(holdings, day)
    next_holdings: dict[str, int] = {}
    sells: list[tuple[str, str]] = []
    for code, age in current.items():
        rank = int(day.at[code, "rank"]) if code in day.index else len(day) + 1
        should_sell = False
        reason = ""
        if age >= cfg.max_hold_days and rank > cfg.buy_rank:
            should_sell = True
            reason = "max_hold_rank_check"
        elif age >= cfg.min_hold_days and rank > cfg.sell_rank:
            should_sell = True
            reason = "rank_buffer_exit"
        if should_sell:
            sells.append((code, reason))
        else:
            next_holdings[code] = age

    buy_pool = day[day["rank"] <= cfg.buy_rank]
    buys = [str(c) for c in buy_pool.index if str(c) not in next_holdings][: max(0, cfg.target_positions - len(next_holdings))]
    if len(next_holdings) + len(buys) < cfg.target_positions:
        extra = _top_codes(day, cfg.target_positions - len(next_holdings) - len(buys), exclude=set(next_holdings) | set(buys))
        buys.extend(extra)
    for code in buys:
        next_holdings[code] = 0
    trades = [{"action": "sell", "ts_code": c, "reason": r} for c, r in sells]
    trades.extend({"action": "buy", "ts_code": c, "reason": "rank_buffer_fill"} for c in buys)
    return next_holdings, trades


def _risk_score_from_history(ret_panel: pd.DataFrame, date: str, core: list[str], candidates: list[str], cfg: StrategyBacktestConfig) -> pd.Series:
    hist = ret_panel.loc[ret_panel.index < date].tail(cfg.risk_window)
    if len(hist) < 2 or not candidates:
        return pd.Series(0.5, index=candidates)
    core_cols = [c for c in core if c in hist.columns]
    cand_cols = [c for c in candidates if c in hist.columns]
    if not core_cols or not cand_cols:
        return pd.Series(0.5, index=candidates)
    core_ret = hist[core_cols].mean(axis=1)
    ch = hist[cand_cols]
    vol = ch.std(ddof=1).replace([np.inf, -np.inf], np.nan)
    corr = ch.corrwith(core_ret).replace([np.inf, -np.inf], np.nan)
    downside = ch.clip(upper=0.0).std(ddof=1).replace([np.inf, -np.inf], np.nan)

    def pct(s: pd.Series, fill: float) -> pd.Series:
        s = s.fillna(s.median() if s.notna().any() else fill)
        return s.rank(method="average", pct=True).fillna(0.5)

    risk = 0.5 * pct(corr, 0.0) + 0.3 * pct(vol, 0.0) + 0.2 * pct(downside, 0.0)
    return risk.reindex(candidates).fillna(0.5)


def _risk_balanced_tail(
    holdings: dict[str, int],
    day: pd.DataFrame,
    ret_panel: pd.DataFrame,
    date: str,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], dict[str, float], list[dict[str, Any]]]:
    score = day[cfg.score_col].astype(float)
    core = [str(c) for c in score.sort_values(ascending=False).head(cfg.core_count).index]
    candidates = [str(c) for c in score.sort_values(ascending=False).index if str(c) not in set(core)]
    candidate_pool = candidates[: max(cfg.tail_risk_candidates, cfg.tail_count)]
    risk = _risk_score_from_history(ret_panel, date, core, candidate_pool, cfg)
    low_risk_pool = list(risk.sort_values(ascending=True).head(min(cfg.tail_risk_candidates, len(risk))).index)
    tail = [str(c) for c in score.reindex(low_risk_pool).sort_values(ascending=False).head(cfg.tail_count).index]
    target_codes = core + tail
    current = _drop_missing(holdings, day)
    current_codes = set(current)
    target_set = set(target_codes)
    additions = [str(c) for c in score.reindex(list(target_set - current_codes)).sort_values(ascending=False).index]
    removals = [str(c) for c in score.reindex(list(current_codes - target_set)).sort_values(ascending=True).index]
    if current:
        sell_count = min(cfg.max_stock_updates, len(removals))
        next_codes = current_codes - set(removals[:sell_count])
        buy_count = min(len(additions), max(0, len(target_codes) - len(next_codes)))
        next_codes = next_codes | set(additions[:buy_count])
    else:
        sell_count = 0
        buy_count = len(additions)
        next_codes = set(target_codes)
    if not current:
        next_codes = set(target_codes)

    core_kept = [c for c in core if c in next_codes]
    tail_kept = [c for c in sorted(next_codes - set(core_kept))]
    weights: dict[str, float] = {}
    if core_kept and tail_kept:
        core_total = cfg.core_weight
        tail_total = 1.0 - cfg.core_weight
    elif core_kept:
        core_total = 1.0
        tail_total = 0.0
    else:
        core_total = 0.0
        tail_total = 1.0
    weights.update(_score_weights(score.reindex(core_kept).dropna(), core_total))
    tail_score = score.reindex(tail_kept).rank(method="average", pct=True).fillna(0.5)
    tail_risk = risk.reindex(tail_kept).rank(method="average", pct=True).fillna(0.5)
    weights.update(_score_weights(0.7 * tail_score + 0.3 * (1.0 - tail_risk), tail_total))
    next_holdings = {c: current.get(c, 0) for c in weights}
    trades = [{"action": "sell", "ts_code": c, "reason": "risk_tail_rebalance"} for c in removals[:sell_count]]
    trades.extend({"action": "buy", "ts_code": c, "reason": "risk_tail_rebalance"} for c in additions[:buy_count])
    return next_holdings, weights, trades


def _risk_filtered_rank_buffer(
    holdings: dict[str, int],
    day: pd.DataFrame,
    ret_panel: pd.DataFrame,
    date: str,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], dict[str, float], list[dict[str, Any]]]:
    current = _drop_missing(holdings, day)
    score = day[cfg.score_col].astype(float)
    ranked_codes = [str(c) for c in score.sort_values(ascending=False).index]
    candidate_pool = ranked_codes[: max(cfg.risk_candidate_count, cfg.target_positions)]
    core = ranked_codes[: min(cfg.core_count, len(ranked_codes))]
    risk = _risk_score_from_history(ret_panel, date, core, candidate_pool, cfg)
    low_risk = list(risk.sort_values(ascending=True).head(min(cfg.risk_keep_count, len(risk))).index)
    low_risk_set = set(str(c) for c in low_risk)

    next_holdings: dict[str, int] = {}
    sell_candidates: list[tuple[str, str, int]] = []
    for code, age in current.items():
        rank = int(day.at[code, "rank"]) if code in day.index else len(day) + 1
        risk_excluded = code not in low_risk_set
        should_sell = False
        reason = ""
        if age >= cfg.max_hold_days and (rank > cfg.buy_rank or risk_excluded):
            should_sell = True
            reason = "max_hold_rank_or_risk"
        elif age >= cfg.min_hold_days and rank > cfg.sell_rank:
            should_sell = True
            reason = "rank_buffer_exit"
        elif age >= cfg.min_hold_days and risk_excluded and rank > cfg.buy_rank:
            should_sell = True
            reason = "risk_filter_exit"
        if should_sell:
            sell_candidates.append((code, reason, rank))
        else:
            next_holdings[code] = age

    sell_candidates = sorted(sell_candidates, key=lambda item: item[2], reverse=True)
    if current:
        sell_limit = min(cfg.max_stock_updates, len(sell_candidates))
    else:
        sell_limit = 0
    sells = sell_candidates[:sell_limit]
    for code, _, _ in sells:
        next_holdings.pop(code, None)
    for code, _, _ in sell_candidates[sell_limit:]:
        next_holdings[code] = current[code]

    slots = max(0, cfg.target_positions - len(next_holdings))
    buy_limit = slots

    strict_buy_pool = [
        code
        for code in ranked_codes
        if code in low_risk_set and code not in next_holdings and int(day.at[code, "rank"]) <= cfg.buy_rank
    ]
    fallback_buy_pool = [code for code in ranked_codes if code in low_risk_set and code not in next_holdings]
    broad_buy_pool = [code for code in ranked_codes if code not in next_holdings]
    buy_pool = strict_buy_pool + [c for c in fallback_buy_pool if c not in set(strict_buy_pool)]
    if not current:
        buy_pool += [c for c in broad_buy_pool if c not in set(buy_pool)]
    buys = buy_pool[:buy_limit]
    for code in buys:
        next_holdings[code] = 0

    trades = [{"action": "sell", "ts_code": c, "reason": r} for c, r, _ in sells]
    trades.extend({"action": "buy", "ts_code": c, "reason": "risk_filtered_rank_buffer_fill"} for c in buys)
    return next_holdings, _equal_weights(sorted(next_holdings)), trades


def _target_holdings(
    holdings: dict[str, int],
    day: pd.DataFrame,
    ret_panel: pd.DataFrame,
    date: str,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], dict[str, float], list[dict[str, Any]]]:
    if cfg.strategy == "rolling_tranche":
        next_holdings, trades = _rolling_tranche(holdings, day, cfg)
        return next_holdings, _equal_weights(sorted(next_holdings)), trades
    if cfg.strategy == "topk_drop":
        next_holdings, trades = _topk_drop(holdings, day, cfg)
        return next_holdings, _equal_weights(sorted(next_holdings)), trades
    if cfg.strategy == "rank_buffer":
        next_holdings, trades = _rank_buffer(holdings, day, cfg)
        return next_holdings, _equal_weights(sorted(next_holdings)), trades
    if cfg.strategy == "risk_balanced_tail":
        return _risk_balanced_tail(holdings, day, ret_panel, date, cfg)
    if cfg.strategy == "risk_filtered_rank_buffer":
        return _risk_filtered_rank_buffer(holdings, day, ret_panel, date, cfg)
    raise ValueError(f"unknown strategy: {cfg.strategy}")


def run_strategy(df: pd.DataFrame, cfg: StrategyBacktestConfig, name: str | None = None) -> dict[str, Any]:
    dates, day_map, ret_panel = _prepare_maps(df, cfg)
    holdings: dict[str, int] = {}
    prev_weights: dict[str, float] = {}
    equity = 1.0
    curve_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []

    for date in dates:
        day = day_map[date]
        holdings, weights, trades = _target_holdings(holdings, day, ret_panel, date, cfg)
        turnover = _turnover(prev_weights, weights)
        cost = turnover * cfg.transaction_cost_bps / 10000.0
        returns = day[cfg.return_col].reindex(weights.keys()).fillna(0.0)
        gross_ret = float(sum(weights[c] * float(returns.get(c, 0.0)) for c in weights))
        net_ret = gross_ret - cost
        equity *= 1.0 + net_ret
        curve_rows.append(
            {
                "trade_date": date,
                "gross_return": gross_ret,
                "transaction_cost": cost,
                "net_return": net_ret,
                "turnover": turnover,
                "equity": equity,
                "n_holdings": len(weights),
            }
        )
        for tr in trades:
            trade_rows.append({"trade_date": date, "strategy": name or cfg.strategy, **tr})
        for code, weight in sorted(weights.items()):
            holding_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "weight": weight,
                    "holding_days": holdings.get(code, 0),
                    "score": float(day.at[code, cfg.score_col]) if code in day.index else math.nan,
                    "rank": int(day.at[code, "rank"]) if code in day.index else -1,
                }
            )
        holdings = {c: age + 1 for c, age in holdings.items()}
        prev_weights = weights

    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    holdings_df = pd.DataFrame(holding_rows)
    returns_np = curve["net_return"].to_numpy(dtype=np.float64) if not curve.empty else np.empty(0)
    equity_np = curve["equity"].to_numpy(dtype=np.float64) if not curve.empty else np.empty(0)
    periods = int(len(curve))
    years = max(1e-12, periods / cfg.trading_days_per_year)
    final_equity = float(equity_np[-1]) if len(equity_np) else 1.0
    metrics = {
        "name": name or cfg.strategy,
        "strategy": cfg.strategy,
        "periods": periods,
        "start_date": str(curve["trade_date"].iloc[0]) if periods else None,
        "end_date": str(curve["trade_date"].iloc[-1]) if periods else None,
        "final_equity": final_equity,
        "total_return": final_equity - 1.0,
        "annual_return": float(final_equity ** (1.0 / years) - 1.0),
        "sharpe": _sharpe(returns_np, cfg.trading_days_per_year),
        "max_drawdown": _max_drawdown(equity_np),
        "avg_turnover": float(curve["turnover"].mean()) if periods else math.nan,
        "avg_n_holdings": float(curve["n_holdings"].mean()) if periods else math.nan,
        "transaction_cost_bps": float(cfg.transaction_cost_bps),
        "config": cfg.__dict__,
    }
    return {"metrics": metrics, "curve": curve, "trades": trades, "holdings": holdings_df}


def metrics_from_curve(
    curve: pd.DataFrame,
    name: str,
    strategy: str,
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    returns_np = curve["net_return"].to_numpy(dtype=np.float64) if "net_return" in curve else np.empty(0)
    equity_np = curve["equity"].to_numpy(dtype=np.float64) if "equity" in curve else np.empty(0)
    periods = int(len(curve))
    years = max(1e-12, periods / trading_days_per_year)
    final_equity = float(equity_np[-1]) if len(equity_np) else 1.0
    return {
        "name": name,
        "strategy": strategy,
        "periods": periods,
        "start_date": str(curve["trade_date"].iloc[0]) if periods else None,
        "end_date": str(curve["trade_date"].iloc[-1]) if periods else None,
        "final_equity": final_equity,
        "total_return": final_equity - 1.0,
        "annual_return": float(final_equity ** (1.0 / years) - 1.0),
        "sharpe": _sharpe(returns_np, trading_days_per_year),
        "max_drawdown": _max_drawdown(equity_np),
        "avg_turnover": float(curve["turnover"].mean()) if "turnover" in curve and periods else 0.0,
        "avg_n_holdings": float(curve["n_holdings"].mean()) if "n_holdings" in curve and periods else math.nan,
        "transaction_cost_bps": 0.0,
        "config": {},
    }


def build_equal_weight_benchmark(
    df: pd.DataFrame,
    return_col: str = "label_1d",
    name: str = "benchmark_equal_weight_universe",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    equity = 1.0
    for date, day in df.groupby("trade_date", sort=True):
        returns = day[return_col].dropna()
        net_return = float(returns.mean()) if len(returns) else 0.0
        equity *= 1.0 + net_return
        rows.append(
            {
                "trade_date": str(date),
                "gross_return": net_return,
                "transaction_cost": 0.0,
                "net_return": net_return,
                "turnover": 0.0,
                "equity": equity,
                "n_holdings": int(len(returns)),
            }
        )
    curve = pd.DataFrame(rows)
    return {
        "metrics": metrics_from_curve(curve, name=name, strategy="benchmark_equal_weight"),
        "curve": curve,
        "trades": pd.DataFrame(),
        "holdings": pd.DataFrame(),
    }


def load_index_weight_data(path: str | Path, index_code: str) -> pd.DataFrame:
    path = Path(path)

    def read_csv(file_obj: Any) -> pd.DataFrame:
        return pd.read_csv(
            file_obj,
            dtype={"index_code": str, "con_code": str, "trade_date": str, "weight": float},
        )

    parts: list[pd.DataFrame] = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.endswith(".csv"):
                    continue
                if index_code not in name:
                    continue
                with zf.open(name) as f:
                    parts.append(read_csv(f))
    elif path.is_dir():
        for csv_path in sorted(path.rglob("*.csv")):
            if index_code not in csv_path.name:
                continue
            parts.append(read_csv(csv_path))
    else:
        parts.append(read_csv(path))

    if not parts:
        raise ValueError(f"{path} contains no index weight CSV for {index_code}")
    weights = pd.concat(parts, ignore_index=True)
    required = {"index_code", "con_code", "trade_date", "weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"{path} missing index weight columns: {sorted(missing)}")
    weights = weights[weights["index_code"].astype(str) == str(index_code)].copy()
    if weights.empty:
        raise ValueError(f"{path} contains no rows for index_code={index_code}")
    weights["trade_date"] = weights["trade_date"].astype(str)
    weights["con_code"] = weights["con_code"].astype(str)
    weights["weight"] = weights["weight"].astype(float)
    weights = weights.dropna(subset=["trade_date", "con_code", "weight"])
    weights = weights[weights["weight"] > 0]
    return weights.sort_values(["trade_date", "weight"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def build_index_weight_benchmark(
    df: pd.DataFrame,
    weight_path: str | Path,
    index_code: str = "000300.SH",
    return_col: str = "label_1d",
    name: str | None = None,
) -> dict[str, Any]:
    name = name or f"benchmark_{index_code.replace('.', '_').lower()}_weight"
    weights = load_index_weight_data(weight_path, index_code)
    weight_dates = sorted(weights["trade_date"].unique().tolist())
    weight_by_date = {
        d: g.groupby("con_code", sort=False)["weight"].sum().astype(float)
        for d, g in weights.groupby("trade_date", sort=True)
    }
    returns_by_date = {
        str(d): g.set_index("ts_code")[return_col].astype(float)
        for d, g in df.groupby("trade_date", sort=True)
    }

    rows: list[dict[str, Any]] = []
    equity = 1.0
    weight_pos = -1
    for date in sorted(returns_by_date):
        while weight_pos + 1 < len(weight_dates) and weight_dates[weight_pos + 1] <= date:
            weight_pos += 1
        if weight_pos < 0:
            continue

        source_date = weight_dates[weight_pos]
        weight = weight_by_date[source_date]
        day_ret = returns_by_date[date]
        common = weight.index.intersection(day_ret.index)
        if len(common) == 0:
            continue
        aligned_weight = weight.reindex(common).astype(float)
        aligned_weight = aligned_weight / float(aligned_weight.sum())
        aligned_ret = day_ret.reindex(common).fillna(0.0).astype(float)
        net_return = float((aligned_weight * aligned_ret).sum())
        equity *= 1.0 + net_return
        rows.append(
            {
                "trade_date": str(date),
                "gross_return": net_return,
                "transaction_cost": 0.0,
                "net_return": net_return,
                "turnover": 0.0,
                "equity": equity,
                "n_holdings": int(len(common)),
                "source_weight_date": str(source_date),
                "index_code": str(index_code),
            }
        )

    if not rows:
        raise ValueError(f"index benchmark {index_code} has no overlapping constituents with prediction data")
    curve = pd.DataFrame(rows)
    return {
        "metrics": metrics_from_curve(curve, name=name, strategy="benchmark_index_weight"),
        "curve": curve,
        "trades": pd.DataFrame(),
        "holdings": pd.DataFrame(),
    }


def load_price_benchmark(
    path: str | Path,
    name: str,
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "trade_date" not in df.columns:
        raise ValueError(f"{path} missing trade_date column")
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date", kind="mergesort")

    if "equity" in df.columns:
        equity = df["equity"].astype(float)
        returns = equity.pct_change().fillna(equity.iloc[0] - 1.0)
    elif "return" in df.columns:
        returns = df["return"].astype(float).fillna(0.0)
        equity = (1.0 + returns).cumprod()
    elif "net_return" in df.columns:
        returns = df["net_return"].astype(float).fillna(0.0)
        equity = (1.0 + returns).cumprod()
    else:
        price_col = next((col for col in ["close", "adj_close", "price", "nav"] if col in df.columns), None)
        if price_col is None:
            raise ValueError(f"{path} needs one of equity, return, net_return, close, adj_close, price, nav")
        price = df[price_col].astype(float)
        equity = price / float(price.iloc[0])
        returns = equity.pct_change().fillna(0.0)

    curve = pd.DataFrame(
        {
            "trade_date": df["trade_date"].astype(str),
            "gross_return": returns.astype(float),
            "transaction_cost": 0.0,
            "net_return": returns.astype(float),
            "turnover": 0.0,
            "equity": equity.astype(float),
            "n_holdings": math.nan,
        }
    )
    return {
        "metrics": metrics_from_curve(curve, name=name, strategy="benchmark_index", trading_days_per_year=trading_days_per_year),
        "curve": curve,
        "trades": pd.DataFrame(),
        "holdings": pd.DataFrame(),
    }


def align_benchmark_to_dates(
    benchmark: dict[str, Any],
    dates: pd.Series | list[str],
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    date_set = {str(d) for d in dates}
    curve = benchmark["curve"].copy()
    curve["trade_date"] = curve["trade_date"].astype(str)
    curve = curve[curve["trade_date"].isin(date_set)].sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    if curve.empty:
        raise ValueError("benchmark has no overlapping trade_date values with the prediction split")
    if "net_return" in curve.columns:
        curve["equity"] = (1.0 + curve["net_return"].astype(float).fillna(0.0)).cumprod()
    metrics = benchmark["metrics"]
    return {
        "metrics": metrics_from_curve(
            curve,
            name=str(metrics.get("name", "benchmark")),
            strategy=str(metrics.get("strategy", "benchmark_index")),
            trading_days_per_year=trading_days_per_year,
        ),
        "curve": curve,
        "trades": benchmark.get("trades", pd.DataFrame()),
        "holdings": benchmark.get("holdings", pd.DataFrame()),
    }


def write_strategy_outputs(result: dict[str, Any], out_dir: str | Path) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    result["curve"].to_csv(path / "equity_curve.csv", index=False)
    result["trades"].to_csv(path / "trades.csv", index=False)
    result["holdings"].to_csv(path / "holdings.csv", index=False)
    with (path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, ensure_ascii=False, indent=2)


def plot_comparison(curves: dict[str, pd.DataFrame], out_path: str | Path, title: str, log_scale: bool = True) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    series = {name: curve for name, curve in curves.items() if not curve.empty}
    if not series:
        out.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return

    width, height = 1200, 680
    left, right, top, bottom = 80, 260, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_dates = sorted({d for curve in series.values() for d in curve["trade_date"].astype(str).tolist()})
    date_to_x = {d: i for i, d in enumerate(all_dates)}
    y_values = [float(y) for curve in series.values() for y in curve["equity"].tolist()]
    y_min = min(y_values + [1.0])
    y_max = max(y_values + [1.0])
    use_log = bool(log_scale and y_min > 0)
    if use_log:
        log_min = math.log(y_min)
        log_max = math.log(y_max)
        pad = max(1e-6, (log_max - log_min) * 0.05)
        y_min_t = log_min - pad
        y_max_t = log_max + pad
    else:
        pad = max(1e-6, (y_max - y_min) * 0.05)
        y_min_t = y_min - pad
        y_max_t = y_max + pad
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#003f5c",
        "#bc5090",
        "#ffa600",
        "#58508d",
    ]

    def sx(date: str) -> float:
        if len(all_dates) <= 1:
            return left
        return left + date_to_x[str(date)] / (len(all_dates) - 1) * plot_w

    def transform_y(value: float) -> float:
        return math.log(max(float(value), 1e-12)) if use_log else float(value)

    def sy(value: float) -> float:
        tv = transform_y(value)
        return top + (y_max_t - tv) / (y_max_t - y_min_t) * plot_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{left}' y='28' font-size='20' font-family='Arial'>{escape(title)}</text>",
        f"<text x='{left + plot_w}' y='28' text-anchor='end' font-size='12' font-family='Arial' fill='#555'>y-scale: {'log equity' if use_log else 'linear equity'}</text>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#333'/>",
    ]
    for i in range(6):
        y = top + i / 5 * plot_h
        raw_val = y_max_t - i / 5 * (y_max_t - y_min_t)
        val = math.exp(raw_val) if use_log else raw_val
        parts.append(f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' stroke='#ddd'/>")
        parts.append(f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' font-size='11' font-family='Arial'>{val:.2f}</text>")
    tick_idx = np.linspace(0, len(all_dates) - 1, num=min(6, len(all_dates)), dtype=int)
    for idx in tick_idx:
        x = left + idx / max(1, len(all_dates) - 1) * plot_w
        parts.append(f"<text x='{x:.2f}' y='{top + plot_h + 24}' text-anchor='middle' font-size='11' font-family='Arial'>{all_dates[idx]}</text>")
    for i, (name, curve) in enumerate(series.items()):
        color = colors[i % len(colors)]
        points = " ".join(f"{sx(str(d)):.2f},{sy(float(e)):.2f}" for d, e in zip(curve["trade_date"], curve["equity"]))
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='1.6' points='{points}'/>")
        ly = top + 20 + i * 22
        parts.append(f"<line x1='{left + plot_w + 25}' y1='{ly - 4}' x2='{left + plot_w + 55}' y2='{ly - 4}' stroke='{color}' stroke-width='2'/>")
        parts.append(f"<text x='{left + plot_w + 62}' y='{ly}' font-size='12' font-family='Arial'>{escape(name)}</text>")
    parts.append(f"<text x='{left + plot_w / 2}' y='{height - 18}' text-anchor='middle' font-size='13' font-family='Arial'>trade_date</text>")
    parts.append(f"<text x='18' y='{top + plot_h / 2}' transform='rotate(-90 18,{top + plot_h / 2})' text-anchor='middle' font-size='13' font-family='Arial'>{'log equity' if use_log else 'equity'}</text>")
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_strategy_grid(cost_bps: float = 5.0) -> list[tuple[str, StrategyBacktestConfig]]:
    rows: list[tuple[str, StrategyBacktestConfig]] = []
    base = StrategyBacktestConfig(strategy="rolling_tranche", transaction_cost_bps=cost_bps)
    for target, hold in [(10, 5), (20, 3), (20, 5), (20, 10), (30, 5)]:
        cfg = replace(base, target_positions=target, hold_days=hold, daily_buy=max(1, round(target / hold)))
        rows.append((f"rolling_p{target}_h{hold}", cfg))
    base = StrategyBacktestConfig(strategy="topk_drop", transaction_cost_bps=cost_bps)
    for topk, drop in [(20, 1), (20, 2), (20, 3), (20, 5), (30, 3)]:
        rows.append((f"topk{topk}_drop{drop}", replace(base, topk=topk, drop=drop)))
    base = StrategyBacktestConfig(strategy="rank_buffer", transaction_cost_bps=cost_bps)
    for target, buy, sell, min_hold, max_hold in [(20, 30, 100, 2, 10), (20, 50, 100, 2, 10), (30, 50, 150, 2, 10)]:
        rows.append(
            (
                f"rankbuf_p{target}_b{buy}_s{sell}_min{min_hold}_max{max_hold}",
                replace(base, target_positions=target, buy_rank=buy, sell_rank=sell, min_hold_days=min_hold, max_hold_days=max_hold),
            )
        )
    rows.append(
        (
            "risk_tail_core30_tail70",
            StrategyBacktestConfig(strategy="risk_balanced_tail", transaction_cost_bps=cost_bps),
        )
    )
    base = StrategyBacktestConfig(strategy="risk_filtered_rank_buffer", transaction_cost_bps=cost_bps)
    for target, candidate, keep, buy, sell, min_hold, max_hold, max_updates in [
        (20, 100, 70, 50, 120, 3, 10, 4),
        (30, 150, 80, 60, 150, 3, 10, 5),
        (30, 100, 70, 50, 120, 5, 15, 4),
        (20, 150, 80, 40, 100, 3, 10, 4),
    ]:
        rows.append(
            (
                f"riskbuf_p{target}_top{candidate}_keep{keep}_b{buy}_s{sell}_min{min_hold}_max{max_hold}",
                replace(
                    base,
                    target_positions=target,
                    risk_candidate_count=candidate,
                    risk_keep_count=keep,
                    buy_rank=buy,
                    sell_rank=sell,
                    min_hold_days=min_hold,
                    max_hold_days=max_hold,
                    max_stock_updates=max_updates,
                ),
            )
        )
    return rows
