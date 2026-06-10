from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from src.strategy import StrategyBacktestConfig, run_strategy
from src.strategy.metrics import metrics_from_curve
from src.utils import write_json, write_run_metadata


VARIANT_LABELS = {
    "all_windows": "全窗口",
    "no_20d": "删除20日窗口",
    "short_5_10": "短窗口5/10日",
}

VARIANT_COLORS = {
    "all_windows": "#1F77B4",
    "no_20d": "#D55E00",
    "short_5_10": "#009E73",
}


def prediction_path(root: Path, variant: str, year: int, model: str) -> Path:
    path = root / variant / model / f"expanding_valid{year}" / "valid_pred.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing walk-forward prediction file: {path}")
    return path


def load_prediction(path: Path, return_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"trade_date", "ts_code", "pred", return_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df[["trade_date", "ts_code", "pred", return_col]].dropna().copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["ts_code"] = out["ts_code"].astype(str)
    return out.sort_values(["trade_date", "pred"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def run_topk_curve(
    pred: pd.DataFrame,
    *,
    topk: int,
    drop: int,
    return_col: str,
    transaction_cost_bps: float,
    trading_days_per_year: int,
) -> pd.DataFrame:
    cfg = StrategyBacktestConfig(
        strategy="topk_drop",
        score_col="pred",
        return_col=return_col,
        transaction_cost_bps=float(transaction_cost_bps),
        trading_days_per_year=int(trading_days_per_year),
        topk=int(topk),
        drop=int(drop),
    )
    curve = run_strategy(pred, cfg, name=f"topk{topk}_drop{drop}")["curve"].copy()
    if curve.empty:
        return curve
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
    return curve


def restitch_oos(curves: list[pd.DataFrame]) -> pd.DataFrame:
    if not curves:
        return pd.DataFrame()
    use = pd.concat(curves, ignore_index=True).sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    use["equity"] = (1.0 + use["net_return"].astype(float).fillna(0.0)).cumprod()
    use["drawdown"] = use["equity"] / use["equity"].cummax() - 1.0
    return use


def _date_text(value: object) -> str:
    text = str(value)
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _metric_suffix(row: dict[str, Any]) -> str:
    return f"净值 {float(row['final_equity']):.2f} / Sharpe {float(row['sharpe']):.2f} / 回撤 {float(row['max_drawdown']):.1%}"


def _ticks(values: list[float], count: int = 5) -> list[float]:
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        lo -= 0.01
        hi += 0.01
    return [lo + idx * (hi - lo) / max(1, count - 1) for idx in range(count)]


def _log_ticks(values: list[float]) -> list[float]:
    lo = math.floor(math.log10(max(1e-12, min(values))))
    hi = math.ceil(math.log10(max(values)))
    ticks: list[float] = []
    for exp in range(lo, hi + 1):
        for mult in (1, 2, 5):
            value = mult * (10.0**exp)
            if min(values) <= value <= max(values):
                ticks.append(value)
    return ticks or [10.0**v for v in _ticks([lo, hi], 5)]


def plot_oos_equity_drawdown(
    curves: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, Any]],
    out_path: Path,
    *,
    title: str,
    log_scale: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    curves = {k: v for k, v in curves.items() if not v.empty}
    if not curves:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return

    width, height = 1480, 860
    left, right, top, bottom = 92, 430, 58, 72
    gap = 40
    equity_h = 450
    dd_h = height - top - bottom - gap - equity_h
    plot_w = width - left - right
    all_dates = sorted({_date_text(d) for curve in curves.values() for d in curve["trade_date"].tolist()})
    date_to_x = {d: i for i, d in enumerate(all_dates)}

    y_values = [float(y) for curve in curves.values() for y in curve["equity"].tolist()] + [1.0]
    use_log = bool(log_scale and min(y_values) > 0)
    if use_log:
        transformed = [math.log10(v) for v in y_values]
        y_min_t = min(transformed) - 0.04
        y_max_t = max(transformed) + 0.04
        y_ticks = _log_ticks(y_values)
    else:
        pad = max(0.01, (max(y_values) - min(y_values)) * 0.06)
        y_min_t = min(y_values) - pad
        y_max_t = max(y_values) + pad
        y_ticks = _ticks([y_min_t, y_max_t])

    dd_values = [float(y) for curve in curves.values() for y in curve["drawdown"].tolist()] + [0.0]
    dd_min = min(dd_values)
    dd_max = 0.0
    dd_ticks = _ticks([dd_min, dd_max], 5)

    def sx(date: object) -> float:
        if len(all_dates) <= 1:
            return left
        return left + date_to_x[_date_text(date)] / (len(all_dates) - 1) * plot_w

    def equity_y(value: float) -> float:
        tv = math.log10(max(float(value), 1e-12)) if use_log else float(value)
        return top + (y_max_t - tv) / max(1e-12, y_max_t - y_min_t) * equity_h

    dd_top = top + equity_h + gap

    def dd_y(value: float) -> float:
        return dd_top + (dd_max - float(value)) / max(1e-12, dd_max - dd_min) * dd_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{left}' y='30' font-size='22' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif' font-weight='700'>{escape(title)}</text>",
        f"<text x='{left + plot_w}' y='30' text-anchor='end' font-size='12' font-family='Arial' fill='#555'>净值轴：{'log10' if use_log else 'linear'}</text>",
        f"<line x1='{left}' y1='{top + equity_h}' x2='{left + plot_w}' y2='{top + equity_h}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + equity_h}' stroke='#333'/>",
        f"<line x1='{left}' y1='{dd_top + dd_h}' x2='{left + plot_w}' y2='{dd_top + dd_h}' stroke='#333'/>",
        f"<line x1='{left}' y1='{dd_top}' x2='{left}' y2='{dd_top + dd_h}' stroke='#333'/>",
    ]
    for value in y_ticks:
        y = equity_y(value)
        parts.append(f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' stroke='#e2e2e2'/>")
        parts.append(f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' font-size='11' font-family='Arial'>{value:.2g}</text>")
    for value in dd_ticks:
        y = dd_y(value)
        parts.append(f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' stroke='#e2e2e2'/>")
        parts.append(f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' font-size='11' font-family='Arial'>{value:.0%}</text>")
    tick_idx = [int(round(x)) for x in pd.Series(range(len(all_dates))).iloc[:: max(1, len(all_dates) // 6)].tolist()]
    tick_idx = sorted(set(tick_idx + [0, len(all_dates) - 1]))
    for idx in tick_idx:
        x = left + idx / max(1, len(all_dates) - 1) * plot_w
        parts.append(f"<text x='{x:.2f}' y='{dd_top + dd_h + 26}' text-anchor='middle' font-size='11' font-family='Arial'>{all_dates[idx]}</text>")
    parts.append(f"<text x='18' y='{top + equity_h / 2}' transform='rotate(-90 18,{top + equity_h / 2})' text-anchor='middle' font-size='13' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif'>净值</text>")
    parts.append(f"<text x='18' y='{dd_top + dd_h / 2}' transform='rotate(-90 18,{dd_top + dd_h / 2})' text-anchor='middle' font-size='13' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif'>回撤</text>")

    label_x = left + plot_w + 28
    for idx, variant in enumerate([v for v in VARIANT_LABELS if v in curves]):
        curve = curves[variant]
        color = VARIANT_COLORS.get(variant, "#666666")
        equity_points = " ".join(f"{sx(r.trade_date):.2f},{equity_y(float(r.equity)):.2f}" for r in curve.itertuples())
        dd_points = " ".join(f"{sx(r.trade_date):.2f},{dd_y(float(r.drawdown)):.2f}" for r in curve.itertuples())
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.7' stroke-linejoin='round' stroke-linecap='round' points='{equity_points}'/>")
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.0' stroke-linejoin='round' stroke-linecap='round' opacity='0.85' points='{dd_points}'/>")
        legend_y = top + 28 + idx * 56
        label = f"{VARIANT_LABELS.get(variant, variant)} · {_metric_suffix(metrics[variant])}"
        parts.append(f"<line x1='{label_x}' y1='{legend_y - 5}' x2='{label_x + 32}' y2='{legend_y - 5}' stroke='{color}' stroke-width='3' stroke-linecap='round'/>")
        parts.append(f"<text x='{label_x + 42}' y='{legend_y}' font-size='13' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif' fill='#222'>{escape(label)}</text>")
    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_yearly_equity(
    curves_by_year: dict[int, dict[str, pd.DataFrame]],
    out_path: Path,
    *,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    years = sorted(curves_by_year)
    if not years:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return
    width, height = 1480, 1040
    cols = 2
    rows = math.ceil(len(years) / cols)
    margin_x, margin_y = 74, 70
    cell_gap_x, cell_gap_y = 54, 72
    cell_w = (width - margin_x * 2 - cell_gap_x * (cols - 1)) / cols
    cell_h = (height - margin_y * 2 - cell_gap_y * (rows - 1)) / rows
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{margin_x}' y='32' font-size='22' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif' font-weight='700'>{escape(title)}</text>",
    ]
    for vidx, variant in enumerate(VARIANT_LABELS):
        x = margin_x + 360 + vidx * 180
        y = 30
        color = VARIANT_COLORS[variant]
        parts.append(f"<line x1='{x}' y1='{y - 5}' x2='{x + 30}' y2='{y - 5}' stroke='{color}' stroke-width='3' stroke-linecap='round'/>")
        parts.append(f"<text x='{x + 38}' y='{y}' font-size='12' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif'>{escape(VARIANT_LABELS[variant])}</text>")

    for pos, year in enumerate(years):
        row = pos // cols
        col = pos % cols
        left = margin_x + col * (cell_w + cell_gap_x)
        top = margin_y + row * (cell_h + cell_gap_y)
        curves = curves_by_year[year]
        all_dates = sorted({_date_text(d) for curve in curves.values() for d in curve["trade_date"].tolist()})
        date_to_x = {d: i for i, d in enumerate(all_dates)}
        y_values = [float(y) for curve in curves.values() for y in curve["equity"].tolist()] + [1.0]
        y_min = min(y_values)
        y_max = max(y_values)
        pad = max(0.01, (y_max - y_min) * 0.08)
        y_min -= pad
        y_max += pad

        def sx(date: object) -> float:
            if len(all_dates) <= 1:
                return left
            return left + date_to_x[_date_text(date)] / (len(all_dates) - 1) * cell_w

        def sy(value: float) -> float:
            return top + (y_max - float(value)) / max(1e-12, y_max - y_min) * cell_h

        parts.append(f"<text x='{left}' y='{top - 18}' font-size='15' font-family='Arial' font-weight='700'>{year}</text>")
        parts.append(f"<rect x='{left}' y='{top}' width='{cell_w}' height='{cell_h}' fill='white' stroke='#d7d7d7'/>")
        for value in _ticks([y_min, y_max], 4):
            y = sy(value)
            parts.append(f"<line x1='{left}' y1='{y:.2f}' x2='{left + cell_w}' y2='{y:.2f}' stroke='#eeeeee'/>")
            parts.append(f"<text x='{left - 8}' y='{y + 4:.2f}' text-anchor='end' font-size='10' font-family='Arial'>{value:.2f}</text>")
        for variant in VARIANT_LABELS:
            curve = curves.get(variant)
            if curve is None or curve.empty:
                continue
            color = VARIANT_COLORS[variant]
            points = " ".join(f"{sx(r.trade_date):.2f},{sy(float(r.equity)):.2f}" for r in curve.itertuples())
            parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.1' stroke-linejoin='round' stroke-linecap='round' points='{points}'/>")
        if all_dates:
            parts.append(f"<text x='{left}' y='{top + cell_h + 20}' font-size='10' font-family='Arial'>{all_dates[0]}</text>")
            parts.append(f"<text x='{left + cell_w}' y='{top + cell_h + 20}' text-anchor='end' font-size='10' font-family='Arial'>{all_dates[-1]}</text>")
    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    root = Path(args.walkforward_dir)
    if not root.exists():
        raise FileNotFoundError(f"walk-forward directory does not exist: {root}")
    curve_root = root / "curves"
    plot_root = root / "plots"
    curve_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    all_curve_rows = []
    all_metric_rows = []
    oos_by_drop: dict[int, dict[str, pd.DataFrame]] = {}
    yearly_by_drop: dict[int, dict[int, dict[str, pd.DataFrame]]] = {}
    oos_metrics_by_drop: dict[int, dict[str, dict[str, Any]]] = {}

    for drop in args.topk_drops:
        oos_by_drop[int(drop)] = {}
        yearly_by_drop[int(drop)] = {}
        oos_metrics_by_drop[int(drop)] = {}
        for variant in args.variants:
            fold_curves = []
            for year in args.valid_years:
                pred = load_prediction(prediction_path(root, variant, int(year), args.model), args.return_col)
                curve = run_topk_curve(
                    pred,
                    topk=args.topk,
                    drop=int(drop),
                    return_col=args.return_col,
                    transaction_cost_bps=args.transaction_cost_bps,
                    trading_days_per_year=args.trading_days_per_year,
                )
                curve = curve.assign(variant=variant, valid_year=int(year), drop=int(drop), topk=int(args.topk))
                out_csv = curve_root / variant / f"expanding_valid{year}_topk{args.topk}_drop{drop}.csv"
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                curve.to_csv(out_csv, index=False)
                yearly_by_drop[int(drop)].setdefault(int(year), {})[variant] = curve
                fold_curves.append(curve)
                all_curve_rows.append(curve)
            oos = restitch_oos(fold_curves).assign(variant=variant, drop=int(drop), topk=int(args.topk))
            oos_path = curve_root / variant / f"walkforward_oos_topk{args.topk}_drop{drop}.csv"
            oos.to_csv(oos_path, index=False)
            oos_by_drop[int(drop)][variant] = oos
            metrics = metrics_from_curve(
                oos,
                name=f"{variant}_topk{args.topk}_drop{drop}",
                strategy="topk_drop",
                trading_days_per_year=args.trading_days_per_year,
                transaction_cost_bps=args.transaction_cost_bps,
                config={"variant": variant, "topk": int(args.topk), "drop": int(drop)},
            )
            oos_metrics_by_drop[int(drop)][variant] = metrics
            all_metric_rows.append({"variant": variant, "drop": int(drop), "topk": int(args.topk), **metrics})

    curve_long = pd.concat(all_curve_rows, ignore_index=True) if all_curve_rows else pd.DataFrame()
    curve_long.to_parquet(root / "walkforward_topk_curve_long.parquet", index=False)
    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df.to_csv(root / "walkforward_oos_curve_metrics.csv", index=False)

    plots: dict[str, str] = {}
    for drop in args.topk_drops:
        drop = int(drop)
        oos_path = plot_root / f"walkforward_oos_topk{args.topk}_drop{drop}.svg"
        yearly_path = plot_root / f"yearly_topk{args.topk}_drop{drop}.svg"
        plot_oos_equity_drawdown(
            oos_by_drop[drop],
            oos_metrics_by_drop[drop],
            oos_path,
            title=f"label1d 窗口 expanding walk-forward OOS：topk{args.topk}_drop{drop}",
            log_scale=not args.linear_scale,
        )
        plot_yearly_equity(
            yearly_by_drop[drop],
            yearly_path,
            title=f"label1d 窗口逐年外推净值：topk{args.topk}_drop{drop}",
        )
        plots[f"oos_topk{args.topk}_drop{drop}"] = str(oos_path)
        plots[f"yearly_topk{args.topk}_drop{drop}"] = str(yearly_path)

    summary = {
        "walkforward_dir": str(root),
        "curve_long": str(root / "walkforward_topk_curve_long.parquet"),
        "metrics": str(root / "walkforward_oos_curve_metrics.csv"),
        "plots": plots,
    }
    write_json(root / "walkforward_curve_plot_summary.json", summary)
    write_run_metadata(
        root / "curve_plot_meta",
        command="plot-label1d-window-walkforward",
        args=args,
        inputs={"walkforward_dir": str(root)},
        outputs=summary,
    )
    return root


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walkforward-dir", required=True)
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--variants", nargs="+", default=["all_windows", "no_20d", "short_5_10"])
    parser.add_argument("--valid-years", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--return-col", default="label_1d")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--topk-drops", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--trading-days-per-year", type=int, default=252)
    parser.add_argument("--linear-scale", action="store_true")
    root = run(parser.parse_args())
    print(json.dumps({"out_dir": str(root)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run_cli()
