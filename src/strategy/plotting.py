from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _format_tick(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _date_text(value: object) -> str:
    text = str(value)
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _log_ticks(y_min_t: float, y_max_t: float) -> list[float]:
    lo = math.floor(y_min_t)
    hi = math.ceil(y_max_t)
    ticks: list[float] = []
    for exp in range(lo, hi + 1):
        for mult in (1, 2, 5):
            value = mult * (10.0 ** exp)
            tv = math.log10(value)
            if y_min_t <= tv <= y_max_t:
                ticks.append(value)
    if len(ticks) >= 3:
        return ticks
    return [10.0 ** x for x in np.linspace(y_min_t, y_max_t, num=6)]


FAMILY_ORDER = {
    "benchmark": 0,
    "rolling_tranche": 1,
    "topk_drop": 2,
    "rank_buffer": 3,
    "defensive_rank_buffer": 4,
    "risk_filtered_rank_buffer": 5,
    "risk_budget_rank_buffer": 6,
    "risk_balanced_tail": 7,
}

FAMILY_COLORS = {
    "rolling_tranche": "#1F77B4",
    "topk_drop": "#E66101",
    "rank_buffer": "#1B9E77",
    "defensive_rank_buffer": "#7F7F7F",
    "risk_filtered_rank_buffer": "#7570B3",
    "risk_budget_rank_buffer": "#D62728",
    "risk_balanced_tail": "#8C564B",
    "other": "#17BECF",
}

FAMILY_LABELS = {
    "benchmark": "benchmarks",
    "rolling_tranche": "rolling tranche",
    "topk_drop": "top-k drop",
    "rank_buffer": "rank buffer",
    "defensive_rank_buffer": "defensive",
    "risk_filtered_rank_buffer": "risk filtered",
    "risk_budget_rank_buffer": "risk budget",
    "risk_balanced_tail": "risk tail",
    "other": "other",
}

FAMILY_DASHES = ["", "7,4", "2,3", "10,3,2,3", "1,4", "12,4,3,4"]
VARIANT_SHADES = [0.0, -0.24, 0.26, -0.40, 0.42, -0.12, 0.14, -0.55, 0.58]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    text = color.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, c)):02X}" for c in rgb)


def _mix(color: str, target: str, amount: float) -> str:
    r, g, b = _hex_to_rgb(color)
    tr, tg, tb = _hex_to_rgb(target)
    mixed = (
        round(r + (tr - r) * amount),
        round(g + (tg - g) * amount),
        round(b + (tb - b) * amount),
    )
    return _rgb_to_hex(mixed)


def _family_variant_color(family: str, variant_idx: int) -> str:
    base = FAMILY_COLORS.get(family, FAMILY_COLORS["other"])
    shade = VARIANT_SHADES[variant_idx % len(VARIANT_SHADES)]
    if variant_idx >= len(VARIANT_SHADES):
        shade = shade * 0.7
    if shade < 0:
        return _mix(base, "#000000", abs(shade))
    if shade > 0:
        return _mix(base, "#FFFFFF", shade)
    return base


def _family_from_name(name: str) -> str:
    if name.startswith("benchmark"):
        return "benchmark"
    prefixes = [
        ("rolling_", "rolling_tranche"),
        ("topk", "topk_drop"),
        ("rankbuf_", "rank_buffer"),
        ("defensive_", "defensive_rank_buffer"),
        ("riskbuf_", "risk_filtered_rank_buffer"),
        ("riskbudget_", "risk_budget_rank_buffer"),
        ("risk_tail_", "risk_balanced_tail"),
    ]
    for prefix, family in prefixes:
        if name.startswith(prefix):
            return family
    return "other"


def _sort_curve_names(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: (FAMILY_ORDER.get(_family_from_name(n), 99), n))


def _group_curve_names(names: list[str]) -> list[tuple[str, list[str]]]:
    ordered = _sort_curve_names(names)
    groups: dict[str, list[str]] = {}
    for name in ordered:
        groups.setdefault(_family_from_name(name), []).append(name)
    return sorted(groups.items(), key=lambda item: (FAMILY_ORDER.get(item[0], 99), item[0]))


def _style_for_name(name: str, variant_idx: int) -> tuple[str, str, float, float]:
    if name == "benchmark_000300_sh_weight" or "000300" in name:
        return "#111111", "", 2.8, 1.0
    if name == "benchmark_equal_weight_universe":
        return "#666666", "8,4", 2.4, 0.95
    family = _family_from_name(name)
    color = _family_variant_color(family, variant_idx)
    dash = FAMILY_DASHES[variant_idx % len(FAMILY_DASHES)]
    stroke_width = 2.4 if variant_idx == 0 else 2.0
    return color, dash, stroke_width, 0.94


def _family_from_strategy(strategy: str) -> str:
    if strategy.startswith("benchmark"):
        return "benchmark"
    return strategy


def plot_comparison(curves: dict[str, pd.DataFrame], out_path: str | Path, title: str, log_scale: bool = True) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    series = {name: curve for name, curve in curves.items() if not curve.empty}
    if not series:
        out.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return

    grouped_names = _group_curve_names(list(series))
    legend_rows = len(series) + len(grouped_names)
    width, height = 1320, max(680, 132 + 22 * legend_rows)
    left, right, top, bottom = 92, 330, 50, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_dates = sorted({_date_text(d) for curve in series.values() for d in curve["trade_date"].tolist()})
    date_to_x = {d: i for i, d in enumerate(all_dates)}
    y_values = [float(y) for curve in series.values() for y in curve["equity"].tolist()]
    y_min = min(y_values + [1.0])
    y_max = max(y_values + [1.0])
    use_log = bool(log_scale and y_min > 0)
    if use_log:
        log_min = math.log10(y_min)
        log_max = math.log10(y_max)
        pad = max(0.02, (log_max - log_min) * 0.06)
        y_min_t = log_min - pad
        y_max_t = log_max + pad
        y_ticks = _log_ticks(y_min_t, y_max_t)
    else:
        pad = max(1e-6, (y_max - y_min) * 0.05)
        y_min_t = y_min - pad
        y_max_t = y_max + pad
        y_ticks = [y_max_t - i / 5 * (y_max_t - y_min_t) for i in range(6)]

    def sx(date: str) -> float:
        if len(all_dates) <= 1:
            return left
        return left + date_to_x[_date_text(date)] / (len(all_dates) - 1) * plot_w

    def transform_y(value: float) -> float:
        return math.log10(max(float(value), 1e-12)) if use_log else float(value)

    def sy(value: float) -> float:
        tv = transform_y(value)
        return top + (y_max_t - tv) / (y_max_t - y_min_t) * plot_h

    scale_label = "log10 equity" if use_log else "linear equity"
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        f"<metadata>{{\"y_axis_scale\":\"{scale_label}\"}}</metadata>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{left}' y='28' font-size='20' font-family='Arial'>{escape(title)}</text>",
        f"<text x='{left + plot_w}' y='28' text-anchor='end' font-size='12' font-family='Arial' fill='#555'>y-axis: {scale_label}</text>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#333'/>",
    ]
    for value in y_ticks:
        y = sy(value)
        parts.append(f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' stroke='#ddd'/>")
        parts.append(f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' font-size='11' font-family='Arial'>{_format_tick(value)}</text>")
    tick_idx = np.linspace(0, len(all_dates) - 1, num=min(6, len(all_dates)), dtype=int)
    for idx in tick_idx:
        x = left + idx / max(1, len(all_dates) - 1) * plot_w
        parts.append(f"<text x='{x:.2f}' y='{top + plot_h + 24}' text-anchor='middle' font-size='11' font-family='Arial'>{all_dates[idx]}</text>")
    legend_y = top + 18
    for family, names in grouped_names:
        header_color = FAMILY_COLORS.get(family, "#444444") if family != "benchmark" else "#222222"
        parts.append(
            f"<text x='{left + plot_w + 25}' y='{legend_y}' font-size='11' font-weight='700' "
            f"font-family='Arial' fill='{header_color}'>{escape(FAMILY_LABELS.get(family, family))}</text>"
        )
        legend_y += 18
        for variant_idx, name in enumerate(names):
            curve = series[name]
            color, dash, stroke_width, opacity = _style_for_name(name, variant_idx)
            points = " ".join(f"{sx(d):.2f},{sy(float(e)):.2f}" for d, e in zip(curve["trade_date"], curve["equity"]))
            dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
            parts.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='{stroke_width:.1f}' "
                f"stroke-linejoin='round' stroke-linecap='round' opacity='{opacity:.2f}' points='{points}'{dash_attr}/>"
            )
            if not curve.empty:
                last = curve.iloc[-1]
                parts.append(
                    f"<circle cx='{sx(last['trade_date']):.2f}' cy='{sy(float(last['equity'])):.2f}' "
                    f"r='2.6' fill='{color}' opacity='{opacity:.2f}'/>"
                )
            parts.append(
                f"<line x1='{left + plot_w + 25}' y1='{legend_y - 4}' x2='{left + plot_w + 58}' y2='{legend_y - 4}' "
                f"stroke='{color}' stroke-width='{max(2.4, stroke_width):.1f}' stroke-linecap='round'{dash_attr}/>"
            )
            parts.append(f"<text x='{left + plot_w + 66}' y='{legend_y}' font-size='12' font-family='Arial'>{escape(name)}</text>")
            legend_y += 22
    parts.append(f"<text x='{left + plot_w / 2}' y='{height - 18}' text-anchor='middle' font-size='13' font-family='Arial'>trade_date</text>")
    parts.append(f"<text x='18' y='{top + plot_h / 2}' transform='rotate(-90 18,{top + plot_h / 2})' text-anchor='middle' font-size='13' font-family='Arial'>{'Equity (log10 scale)' if use_log else 'Equity'}</text>")
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _metric_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _benchmark_names(rows: list[dict[str, Any]], curves: dict[str, pd.DataFrame]) -> list[str]:
    df = _metric_frame(rows)
    if df.empty:
        return [name for name in curves if name.startswith("benchmark")]
    names = df[df["strategy"].astype(str).str.startswith("benchmark")]["name"].astype(str).tolist()
    return [name for name in names if name in curves]


def _top_strategy_names(rows: list[dict[str, Any]], curves: dict[str, pd.DataFrame], top_n: int) -> list[str]:
    df = _metric_frame(rows)
    if df.empty:
        return []
    df = df[~df["strategy"].astype(str).str.startswith("benchmark")]
    df = df[df["name"].astype(str).isin(curves)]
    df = df.sort_values(["sharpe", "total_return"], ascending=[False, False], kind="mergesort")
    return df["name"].astype(str).head(top_n).tolist()


def _best_by_family(rows: list[dict[str, Any]], curves: dict[str, pd.DataFrame]) -> list[str]:
    df = _metric_frame(rows)
    if df.empty:
        return []
    df = df[~df["strategy"].astype(str).str.startswith("benchmark")]
    df = df[df["name"].astype(str).isin(curves)]
    df = df.sort_values(["strategy", "sharpe", "total_return"], ascending=[True, False, False], kind="mergesort")
    return df.groupby("strategy", sort=True).head(1)["name"].astype(str).tolist()


def _subset_curves(curves: dict[str, pd.DataFrame], names: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    seen: set[str] = set()
    for name in names:
        if name in curves and name not in seen:
            out[name] = curves[name]
            seen.add(name)
    return out


def write_split_plots(
    curves: dict[str, pd.DataFrame],
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    title_prefix: str,
    log_scale: bool = True,
    valid_rows: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    out = Path(out_dir)
    plot_paths: dict[str, str] = {}
    benchmarks = _benchmark_names(rows, curves)
    selector_rows = valid_rows or rows
    top_valid = _top_strategy_names(selector_rows, curves, top_n=6)
    overview = benchmarks + top_valid[:3] + _best_by_family(selector_rows, curves)

    overview_path = out / "equity_overview.svg"
    plot_comparison(_subset_curves(curves, overview), overview_path, f"{title_prefix} overview", log_scale=log_scale)
    plot_paths["overview"] = str(overview_path)

    top_path = out / "equity_top_valid_sharpe.svg"
    plot_comparison(_subset_curves(curves, benchmarks + top_valid), top_path, f"{title_prefix} top valid Sharpe", log_scale=log_scale)
    plot_paths["top_valid_sharpe"] = str(top_path)

    family_dir = out / "plots_by_family"
    family_dir.mkdir(parents=True, exist_ok=True)
    df = _metric_frame(rows)
    if not df.empty:
        for family, g in df.groupby(df["strategy"].astype(str).map(_family_from_strategy), sort=True):
            if family == "benchmark":
                continue
            names = g["name"].astype(str).tolist()
            path = family_dir / f"{family}.svg"
            plot_comparison(_subset_curves(curves, benchmarks + names), path, f"{title_prefix} {family}", log_scale=log_scale)
            plot_paths[f"family_{family}"] = str(path)

    debug_path = out / "equity_all_debug.svg"
    plot_comparison(curves, debug_path, f"{title_prefix} all strategies debug", log_scale=log_scale)
    plot_paths["all_debug"] = str(debug_path)
    return plot_paths
