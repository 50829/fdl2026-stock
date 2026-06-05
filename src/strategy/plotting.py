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

MODEL_ORDER = {
    "label5d_final": 0,
    "label1d_lgb": 1,
    "label1d_xgb": 2,
    "label1d_fusion_valid_alpha": 3,
}

MODEL_COLORS = {
    "label5d_final": "#111111",
    "label1d_lgb": "#D55E00",
    "label1d_xgb": "#009E73",
    "label1d_fusion_valid_alpha": "#CC79A7",
}

MODEL_LABELS = {
    "label5d_final": "label5d final",
    "label1d_lgb": "label1d LightGBM",
    "label1d_xgb": "label1d XGBoost",
    "label1d_fusion_valid_alpha": "label1d fusion",
}

STRATEGY_DASHES = {
    "rolling_p10_h5": "",
    "rolling_p20_h3": "7,4",
    "rolling_p20_h5": "2,3",
    "rolling_p20_h10": "10,3,2,3",
    "topk20_drop2": "",
    "topk20_drop3": "7,4",
    "rankbuf_p20_b30_s100_min2_max10": "",
    "rankbuf_p20_b50_s100_min2_max10": "7,4",
}

MODEL_DASHES = {
    "label5d_final": "",
    "label1d_lgb": "7,4",
    "label1d_xgb": "2,3",
    "label1d_fusion_valid_alpha": "10,3,2,3",
    "final": "",
    "lgb_top40": "7,4",
    "xgb_top40": "2,3",
}

STRATEGY_COLORS = {
    "rolling_p10_h5": "#1F77B4",
    "rolling_p20_h3": "#4E79A7",
    "rolling_p20_h5": "#76B7B2",
    "rolling_p20_h10": "#59A14F",
    "rolling_p30_h5": "#8CD17D",
    "topk20_drop1": "#E15759",
    "topk20_drop2": "#F28E2B",
    "topk20_drop3": "#D37295",
    "topk20_drop5": "#B07AA1",
    "topk30_drop3": "#FF9DA7",
    "rankbuf_p20_b30_s100_min2_max10": "#9C755F",
    "rankbuf_p20_b50_s100_min2_max10": "#EDC948",
    "rankbuf_p30_b50_s150_min2_max10": "#B6992D",
}

STRATEGY_COLOR_CYCLE = [
    "#1F77B4",
    "#E15759",
    "#59A14F",
    "#F28E2B",
    "#B07AA1",
    "#76B7B2",
    "#9C755F",
    "#EDC948",
    "#4E79A7",
    "#D37295",
]


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
    if "__" in name:
        name = name.split("__", 1)[1]
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


def _split_curve_name(name: str) -> tuple[str | None, str]:
    if "__" not in name:
        return None, name
    model, strategy = name.split("__", 1)
    return model, strategy


def _strategy_from_name(name: str) -> str:
    return _split_curve_name(name)[1]


def _model_from_name(name: str) -> str | None:
    return _split_curve_name(name)[0]


def _stable_index(text: str, modulo: int) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % modulo


def _strategy_color(strategy_name: str) -> str:
    if strategy_name in STRATEGY_COLORS:
        return STRATEGY_COLORS[strategy_name]
    return STRATEGY_COLOR_CYCLE[_stable_index(strategy_name, len(STRATEGY_COLOR_CYCLE))]


def _sort_curve_names(names: list[str]) -> list[str]:
    if any("__" in name and name.split("__", 1)[0] in MODEL_ORDER for name in names):
        return sorted(
            names,
            key=lambda n: (
                MODEL_ORDER.get(n.split("__", 1)[0], 99) if "__" in n else 98,
                FAMILY_ORDER.get(_family_from_name(n), 99),
                n.split("__", 1)[1] if "__" in n else n,
            ),
        )
    return sorted(names, key=lambda n: (FAMILY_ORDER.get(_family_from_name(n), 99), n))


def _group_curve_names(names: list[str]) -> list[tuple[str, list[str]]]:
    ordered = _sort_curve_names(names)
    if any("__" in name and name.split("__", 1)[0] in MODEL_ORDER for name in ordered):
        groups: dict[str, list[str]] = {}
        for name in ordered:
            model = name.split("__", 1)[0] if "__" in name else "other"
            groups.setdefault(model, []).append(name)
        return sorted(groups.items(), key=lambda item: (MODEL_ORDER.get(item[0], 99), item[0]))
    groups: dict[str, list[str]] = {}
    for name in ordered:
        groups.setdefault(_family_from_name(name), []).append(name)
    return sorted(groups.items(), key=lambda item: (FAMILY_ORDER.get(item[0], 99), item[0]))


def _style_for_name(name: str, variant_idx: int, style_mode: str = "auto") -> tuple[str, str, float, float]:
    if name == "benchmark_000300_sh_weight" or "000300" in name:
        return "#111111", "", 2.8, 1.0
    if name == "benchmark_equal_weight_universe":
        return "#666666", "8,4", 2.4, 0.95
    if "__" in name:
        model, strategy_name = name.split("__", 1)
        if style_mode == "model":
            color = MODEL_COLORS.get(model, STRATEGY_COLOR_CYCLE[_stable_index(model, len(STRATEGY_COLOR_CYCLE))])
            dash = MODEL_DASHES.get(model, "")
            stroke_width = 2.8 if model in {"label5d_final", "final"} else 2.5
            return color, dash, stroke_width, 0.98
        if style_mode in {"strategy", "unique"}:
            color = _strategy_color(strategy_name)
            dash = MODEL_DASHES.get(model, "")
            stroke_width = 2.7 if not dash else 2.4
            return color, dash, stroke_width, 0.97
        if model in MODEL_COLORS:
            color = _strategy_color(strategy_name)
            dash = MODEL_DASHES.get(model, STRATEGY_DASHES.get(strategy_name, ""))
            stroke_width = 2.8 if model == "label5d_final" else 2.4
            return color, dash, stroke_width, 0.96
    family = _family_from_name(name)
    color = _family_variant_color(family, variant_idx)
    dash = FAMILY_DASHES[variant_idx % len(FAMILY_DASHES)]
    stroke_width = 2.4 if variant_idx == 0 else 2.0
    return color, dash, stroke_width, 0.94


def _family_from_strategy(strategy: str) -> str:
    if strategy.startswith("benchmark"):
        return "benchmark"
    return strategy


def _metric_lookup(metric_rows: list[dict[str, Any]] | pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if metric_rows is None:
        return {}
    df = metric_rows if isinstance(metric_rows, pd.DataFrame) else pd.DataFrame(metric_rows)
    if df.empty or "name" not in df.columns:
        return {}
    return {str(row["name"]): row.to_dict() for _, row in df.iterrows()}


def _metric_suffix(name: str, metrics: dict[str, dict[str, Any]], final_equity: float) -> str:
    row = metrics.get(name, {})
    sharpe = row.get("sharpe")
    max_dd = row.get("max_drawdown")
    turnover_value = row.get("avg_turnover")
    parts = [f"净值 {final_equity:.2f}"]
    if sharpe is not None and not pd.isna(sharpe):
        parts.append(f"S {float(sharpe):.2f}")
    if max_dd is not None and not pd.isna(max_dd):
        parts.append(f"DD {float(max_dd):.1%}")
    if turnover_value is not None and not pd.isna(turnover_value):
        parts.append(f"换手 {float(turnover_value):.2f}")
    return " · ".join(parts)


def _curve_display_name(name: str, label_context: str = "auto") -> str:
    model, strategy = _split_curve_name(name)
    if name == "benchmark_equal_weight_universe":
        return "等权股票池"
    if name == "benchmark_000300_sh_weight" or "000300" in name:
        return "沪深300权重"
    if model is None:
        return name
    model_label = MODEL_LABELS.get(model, model)
    if label_context == "model":
        return model_label
    if label_context == "strategy":
        return strategy
    return f"{model_label} / {strategy}"


def _adjust_label_positions(items: list[dict[str, Any]], y_min: float, y_max: float, min_gap: float = 17.0) -> list[dict[str, Any]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: float(item["y"]))
    for idx in range(1, len(ordered)):
        if float(ordered[idx]["label_y"]) < float(ordered[idx - 1]["label_y"]) + min_gap:
            ordered[idx]["label_y"] = float(ordered[idx - 1]["label_y"]) + min_gap
    overflow = float(ordered[-1]["label_y"]) - y_max
    if overflow > 0:
        for item in ordered:
            item["label_y"] = float(item["label_y"]) - overflow
    for idx in range(len(ordered) - 2, -1, -1):
        if float(ordered[idx]["label_y"]) > float(ordered[idx + 1]["label_y"]) - min_gap:
            ordered[idx]["label_y"] = float(ordered[idx + 1]["label_y"]) - min_gap
    underflow = y_min - float(ordered[0]["label_y"])
    if underflow > 0:
        for item in ordered:
            item["label_y"] = float(item["label_y"]) + underflow
    return ordered


def plot_comparison(
    curves: dict[str, pd.DataFrame],
    out_path: str | Path,
    title: str,
    log_scale: bool = True,
    *,
    style_mode: str = "auto",
    label_context: str = "auto",
    direct_labels: bool = True,
    metric_rows: list[dict[str, Any]] | pd.DataFrame | None = None,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    series = {name: curve for name, curve in curves.items() if not curve.empty}
    if not series:
        out.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return

    grouped_names = _group_curve_names(list(series))
    show_direct_labels = bool(direct_labels and len(series) <= 18)
    legend_rows = len(series) + (0 if show_direct_labels else len(grouped_names))
    width, height = 1480, max(680, 132 + 22 * legend_rows)
    left, right, top, bottom = 92, 470 if show_direct_labels else 360, 50, 78
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
    metrics = _metric_lookup(metric_rows)
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

    ordered_names = _sort_curve_names(list(series))
    end_items: list[dict[str, Any]] = []
    for variant_idx, name in enumerate(ordered_names):
        curve = series[name]
        color, dash, stroke_width, opacity = _style_for_name(name, variant_idx, style_mode=style_mode)
        points = " ".join(f"{sx(d):.2f},{sy(float(e)):.2f}" for d, e in zip(curve["trade_date"], curve["equity"]))
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        parts.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='{stroke_width:.1f}' "
            f"stroke-linejoin='round' stroke-linecap='round' opacity='{opacity:.2f}' points='{points}'{dash_attr}/>"
        )
        last = curve.iloc[-1]
        end_x = sx(last["trade_date"])
        end_y = sy(float(last["equity"]))
        parts.append(f"<circle cx='{end_x:.2f}' cy='{end_y:.2f}' r='2.8' fill='{color}' opacity='{opacity:.2f}'/>")
        if show_direct_labels:
            final_equity = float(last["equity"])
            label = f"{_curve_display_name(name, label_context)} · {_metric_suffix(name, metrics, final_equity)}"
            end_items.append(
                {
                    "name": name,
                    "x": end_x,
                    "y": end_y,
                    "label_y": end_y,
                    "label": label,
                    "color": color,
                    "dash": dash,
                    "stroke_width": stroke_width,
                }
            )

    label_x = left + plot_w + 32
    if show_direct_labels:
        for item in _adjust_label_positions(end_items, top + 12, top + plot_h - 8):
            dash_attr = f" stroke-dasharray='{item['dash']}'" if item["dash"] else ""
            parts.append(
                f"<line x1='{float(item['x']) + 4:.2f}' y1='{float(item['y']):.2f}' "
                f"x2='{label_x - 8:.2f}' y2='{float(item['label_y']):.2f}' "
                f"stroke='{item['color']}' stroke-width='1.2' opacity='0.65'{dash_attr}/>"
            )
            parts.append(
                f"<line x1='{label_x:.2f}' y1='{float(item['label_y']) - 4:.2f}' "
                f"x2='{label_x + 28:.2f}' y2='{float(item['label_y']) - 4:.2f}' "
                f"stroke='{item['color']}' stroke-width='{max(2.4, float(item['stroke_width'])):.1f}' "
                f"stroke-linecap='round'{dash_attr}/>"
            )
            parts.append(
                f"<text x='{label_x + 36:.2f}' y='{float(item['label_y']):.2f}' "
                f"font-size='11' font-family='Arial' fill='#222'>{escape(str(item['label']))}</text>"
            )
    else:
        legend_y = top + 18
        for family, names in grouped_names:
            header_color = MODEL_COLORS.get(family, FAMILY_COLORS.get(family, "#444444")) if family != "benchmark" else "#222222"
            header_label = MODEL_LABELS.get(family, FAMILY_LABELS.get(family, family))
            parts.append(
                f"<text x='{left + plot_w + 25}' y='{legend_y}' font-size='11' font-weight='700' "
                f"font-family='Arial' fill='{header_color}'>{escape(header_label)}</text>"
            )
            legend_y += 18
            for variant_idx, name in enumerate(names):
                curve = series[name]
                color, dash, stroke_width, _ = _style_for_name(name, variant_idx, style_mode=style_mode)
                dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
                last = curve.iloc[-1]
                label = f"{_curve_display_name(name, label_context)} · {_metric_suffix(name, metrics, float(last['equity']))}"
                parts.append(
                    f"<line x1='{left + plot_w + 25}' y1='{legend_y - 4}' x2='{left + plot_w + 58}' y2='{legend_y - 4}' "
                    f"stroke='{color}' stroke-width='{max(2.4, stroke_width):.1f}' stroke-linecap='round'{dash_attr}/>"
                )
                parts.append(f"<text x='{left + plot_w + 66}' y='{legend_y}' font-size='11' font-family='Arial'>{escape(label)}</text>")
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
    plot_comparison(
        _subset_curves(curves, overview),
        overview_path,
        f"{title_prefix} overview",
        log_scale=log_scale,
        style_mode="unique",
        label_context="auto",
        metric_rows=rows,
    )
    plot_paths["overview"] = str(overview_path)

    top_path = out / "equity_top_valid_sharpe.svg"
    plot_comparison(
        _subset_curves(curves, benchmarks + top_valid),
        top_path,
        f"{title_prefix} top valid Sharpe",
        log_scale=log_scale,
        style_mode="unique",
        label_context="auto",
        metric_rows=rows,
    )
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
            plot_comparison(
                _subset_curves(curves, benchmarks + names),
                path,
                f"{title_prefix} {family}",
                log_scale=log_scale,
                style_mode="unique",
                label_context="auto",
                metric_rows=rows,
            )
            plot_paths[f"family_{family}"] = str(path)

        strategy_dir = out / "plots_by_strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        for strategy, g in df[~df["strategy"].astype(str).str.startswith("benchmark")].groupby("strategy", sort=True):
            names = g["name"].astype(str).tolist()
            path = strategy_dir / f"{strategy}.svg"
            plot_comparison(
                _subset_curves(curves, benchmarks + names),
                path,
                f"{title_prefix} {strategy}",
                log_scale=log_scale,
                style_mode="model",
                label_context="model",
                metric_rows=rows,
            )
            plot_paths[f"strategy_{strategy}"] = str(path)

        variant_df = df[~df["strategy"].astype(str).str.startswith("benchmark")].copy()
        variant_df["_variant"] = variant_df["name"].astype(str).map(_strategy_from_name)
        variant_dir = out / "plots_by_variant"
        variant_dir.mkdir(parents=True, exist_ok=True)
        for variant, g in variant_df.groupby("_variant", sort=True):
            names = g["name"].astype(str).tolist()
            path = variant_dir / f"{variant}.svg"
            plot_comparison(
                _subset_curves(curves, benchmarks + names),
                path,
                f"{title_prefix} {variant}",
                log_scale=log_scale,
                style_mode="model",
                label_context="model",
                metric_rows=rows,
            )
            plot_paths[f"variant_{variant}"] = str(path)

        if "model" in df.columns:
            model_dir = out / "plots_by_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_df = df[~df["strategy"].astype(str).str.startswith("benchmark")]
            for model, g in model_df.groupby("model", sort=True):
                names = g["name"].astype(str).tolist()
                path = model_dir / f"{model}.svg"
                plot_comparison(
                    _subset_curves(curves, benchmarks + names),
                    path,
                    f"{title_prefix} {model}",
                    log_scale=log_scale,
                    style_mode="strategy",
                    label_context="strategy",
                    metric_rows=rows,
                )
                plot_paths[f"model_{model}"] = str(path)

        key_names = benchmarks + top_valid[: min(12, len(top_valid))]
        key_path = out / "equity_key_model_strategy_matrix.svg"
        plot_comparison(
            _subset_curves(curves, key_names),
            key_path,
            f"{title_prefix} key model strategy comparison",
            log_scale=log_scale,
            style_mode="unique",
            label_context="auto",
            metric_rows=rows,
        )
        plot_paths["key_model_strategy_matrix"] = str(key_path)

    debug_path = out / "equity_all_debug.svg"
    plot_comparison(
        curves,
        debug_path,
        f"{title_prefix} all strategies debug",
        log_scale=log_scale,
        style_mode="unique",
        label_context="auto",
        direct_labels=False,
        metric_rows=rows,
    )
    plot_paths["all_debug"] = str(debug_path)
    return plot_paths
