from __future__ import annotations

import argparse
import json
import warnings
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipelines.run_strategy_backtest import (
    DEFAULT_MODEL_REGISTRY,
    load_model_registry,
    registered_model_names,
    resolve_feature_set,
    resolve_prediction_path,
)
from src.strategy import build_strategy_grid, load_prediction_data, merge_feature_columns, merge_trade_constraint_columns, run_strategy
from src.utils import DEFAULT_STRATEGY_REGISTRY, load_registry, make_run_dir, resolve_strategy_run, write_run_metadata


RISK_PROFILES: dict[str, dict[str, Any]] = {
    "none": {
        "label": "无风控",
        "overrides": {},
    },
    "market_mild": {
        "label": "市场压力温和降仓",
        "overrides": {
            "apply_market_stress_deleveraging": True,
            "market_window": 5,
            "market_stress_threshold": -0.05,
            "market_stress_lag": 2,
            "stress_gross_exposure": 0.75,
        },
    },
    "market_medium": {
        "label": "市场压力中等降仓",
        "overrides": {
            "apply_market_stress_deleveraging": True,
            "market_window": 5,
            "market_stress_threshold": -0.04,
            "market_stress_lag": 2,
            "stress_gross_exposure": 0.65,
        },
    },
    "dd_mild": {
        "label": "组合回撤温和降仓",
        "overrides": {
            "apply_drawdown_control": True,
            "drawdown_warning_threshold": -0.12,
            "drawdown_warning_exposure": 0.75,
            "drawdown_cut_threshold": -0.18,
            "drawdown_cut_exposure": 0.50,
            "drawdown_stop_threshold": -0.25,
            "drawdown_stop_exposure": 0.35,
        },
    },
    "dd_medium": {
        "label": "组合回撤中等降仓",
        "overrides": {
            "apply_drawdown_control": True,
            "drawdown_warning_threshold": -0.10,
            "drawdown_warning_exposure": 0.65,
            "drawdown_cut_threshold": -0.15,
            "drawdown_cut_exposure": 0.40,
            "drawdown_stop_threshold": -0.22,
            "drawdown_stop_exposure": 0.25,
        },
    },
    "combined_mild": {
        "label": "市场+回撤温和",
        "overrides": {
            "apply_market_stress_deleveraging": True,
            "market_window": 5,
            "market_stress_threshold": -0.05,
            "market_stress_lag": 2,
            "stress_gross_exposure": 0.75,
            "apply_drawdown_control": True,
            "drawdown_warning_threshold": -0.12,
            "drawdown_warning_exposure": 0.75,
            "drawdown_cut_threshold": -0.18,
            "drawdown_cut_exposure": 0.50,
            "drawdown_stop_threshold": -0.25,
            "drawdown_stop_exposure": 0.35,
        },
    },
    "combined_medium": {
        "label": "市场+回撤中等",
        "overrides": {
            "apply_market_stress_deleveraging": True,
            "market_window": 5,
            "market_stress_threshold": -0.04,
            "market_stress_lag": 2,
            "stress_gross_exposure": 0.65,
            "apply_drawdown_control": True,
            "drawdown_warning_threshold": -0.10,
            "drawdown_warning_exposure": 0.65,
            "drawdown_cut_threshold": -0.15,
            "drawdown_cut_exposure": 0.40,
            "drawdown_stop_threshold": -0.22,
            "drawdown_stop_exposure": 0.25,
        },
    },
    "combined_hard": {
        "label": "市场+回撤强降仓",
        "overrides": {
            "apply_market_stress_deleveraging": True,
            "market_window": 5,
            "market_stress_threshold": -0.03,
            "market_stress_lag": 2,
            "stress_gross_exposure": 0.50,
            "apply_drawdown_control": True,
            "drawdown_warning_threshold": -0.08,
            "drawdown_warning_exposure": 0.50,
            "drawdown_cut_threshold": -0.12,
            "drawdown_cut_exposure": 0.25,
            "drawdown_stop_threshold": -0.18,
            "drawdown_stop_exposure": 0.20,
        },
    },
}


DEFAULT_PROFILE_NAMES = ["none", "market_mild", "market_medium", "dd_mild", "dd_medium", "combined_mild", "combined_medium", "combined_hard"]


def resolve_risk_profiles(names: list[str] | None = None) -> list[dict[str, Any]]:
    selected = names or list(DEFAULT_PROFILE_NAMES)
    unknown = sorted(set(selected) - set(RISK_PROFILES))
    if unknown:
        raise ValueError(f"unknown risk profile(s): {unknown}; available: {sorted(RISK_PROFILES)}")
    return [{"name": name, **RISK_PROFILES[name]} for name in selected]


def _filter_grid(grid: list[tuple[str, Any]], strategies: list[str] | None, parser: argparse.ArgumentParser) -> list[tuple[str, Any]]:
    if not strategies:
        return grid
    selected = set(str(name) for name in strategies)
    known = {name for name, _ in grid}
    unknown = sorted(selected - known)
    if unknown:
        parser.error("unknown strategy value(s): " + ", ".join(unknown) + "; known strategies: " + ", ".join(sorted(known)))
    return [(name, cfg) for name, cfg in grid if name in selected]


def add_risk_return_score(
    metrics: pd.DataFrame,
    *,
    max_drawdown_limit: float = -0.25,
    min_avg_gross_exposure: float = 0.45,
    max_avg_turnover: float = 0.65,
) -> pd.DataFrame:
    out = metrics.copy()
    if out.empty:
        return out
    dd_abs_limit = abs(float(max_drawdown_limit))
    dd_abs = out["max_drawdown"].abs().astype(float)
    out["passes_drawdown_limit"] = out["max_drawdown"].astype(float) >= float(max_drawdown_limit)
    out["drawdown_violation"] = (dd_abs - dd_abs_limit).clip(lower=0.0)
    out["turnover_violation"] = (out["avg_turnover"].astype(float) - float(max_avg_turnover)).clip(lower=0.0)
    gross = out.get("avg_gross_exposure", pd.Series(1.0, index=out.index)).fillna(1.0).astype(float)
    out["exposure_violation"] = (float(min_avg_gross_exposure) - gross).clip(lower=0.0)
    out["risk_return_score"] = (
        out["sharpe"].astype(float)
        + 0.50 * out["annual_return"].astype(float)
        - 6.00 * out["drawdown_violation"]
        - 0.80 * out["turnover_violation"]
        - 1.00 * out["exposure_violation"]
    )
    return out


def select_valid_risk_return(metrics: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    scored = add_risk_return_score(metrics)
    valid = scored[scored["split"].astype(str) == "valid"].copy()
    test = scored[scored["split"].astype(str) == "test"].copy()
    if valid.empty:
        return pd.DataFrame()
    valid = valid.sort_values(
        ["passes_drawdown_limit", "risk_return_score", "total_return", "sharpe", "max_drawdown"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    ).head(top_n)
    key = ["model", "variant", "risk_profile"]
    valid_cols = key + [
        "risk_profile_label",
        "strategy",
        "risk_return_score",
        "passes_drawdown_limit",
        "final_equity",
        "total_return",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "avg_gross_exposure",
        "market_stress_days",
        "drawdown_control_days",
    ]
    valid = valid[[col for col in valid_cols if col in valid.columns]].rename(
        columns={
            col: f"{col}_valid"
            for col in valid_cols
            if col not in {"model", "variant", "risk_profile", "risk_profile_label", "strategy"} and col in valid.columns
        }
    )
    if test.empty:
        return valid
    test_cols = key + [
        "final_equity",
        "total_return",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "avg_gross_exposure",
        "market_stress_days",
        "drawdown_control_days",
    ]
    test = test[[col for col in test_cols if col in test.columns]].rename(
        columns={col: f"{col}_test" for col in test_cols if col not in key and col in test.columns}
    )
    return valid.merge(test, on=key, how="left")


def _is_dominated(row: pd.Series, candidates: pd.DataFrame) -> bool:
    better_or_equal = (
        (candidates["total_return"] >= row["total_return"])
        & (candidates["max_drawdown"] >= row["max_drawdown"])
        & (candidates["sharpe"] >= row["sharpe"])
    )
    strictly_better = (
        (candidates["total_return"] > row["total_return"])
        | (candidates["max_drawdown"] > row["max_drawdown"])
        | (candidates["sharpe"] > row["sharpe"])
    )
    return bool((better_or_equal & strictly_better).any())


def pareto_frontier(metrics: pd.DataFrame, split: str = "valid") -> pd.DataFrame:
    scored = add_risk_return_score(metrics)
    sub = scored[scored["split"].astype(str) == split].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["is_pareto"] = [not _is_dominated(row, sub) for _, row in sub.iterrows()]
    return sub[sub["is_pareto"]].sort_values(["max_drawdown", "total_return"], ascending=[False, False], kind="mergesort")


def _metric_text(value: object, pct: bool = False) -> str:
    if value is None or pd.isna(value):
        return ""
    number = float(value)
    if pct:
        return f"{number:.1%}"
    return f"{number:.3f}"


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "没有结果。\n"
    show = df[[col for col in cols if col in df.columns]].copy()
    pct_cols = [col for col in show.columns if col != "risk_return_score_valid" and any(key in col for key in ["return", "drawdown"])]
    for col in show.columns:
        if col in pct_cols:
            show[col] = show[col].map(lambda x: _metric_text(x, pct=True))
        elif pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(_metric_text)
    lines = ["| " + " | ".join(show.columns) + " |", "| " + " | ".join("---" for _ in show.columns) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
    return "\n".join(lines) + "\n"


def _scatter_color(profile: str) -> str:
    colors = {
        "none": "#111111",
        "market_mild": "#1f77b4",
        "market_medium": "#4e79a7",
        "dd_mild": "#59a14f",
        "dd_medium": "#8cd17d",
        "combined_mild": "#f28e2b",
        "combined_medium": "#e15759",
        "combined_hard": "#b07aa1",
    }
    return colors.get(profile, "#777777")


def write_risk_return_scatter(metrics: pd.DataFrame, out_root: str | Path, split: str = "valid") -> str:
    root = Path(out_root)
    path = root / "plots" / f"{split}_risk_return_scatter.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    scored = add_risk_return_score(metrics)
    sub = scored[scored["split"].astype(str) == split].copy()
    if sub.empty:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return str(path)
    width, height = 1120, 720
    left, right, top, bottom = 90, 280, 56, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    x = sub["max_drawdown"].abs().astype(float)
    y = sub["total_return"].astype(float)
    size = sub["avg_turnover"].fillna(0.0).astype(float)
    x_min, x_max = max(0.0, float(x.min()) - 0.03), float(x.max()) + 0.03
    y_min, y_max = min(0.0, float(y.min()) - 0.15), float(y.max()) + 0.15

    def sx(value: float) -> float:
        return left + (value - x_min) / max(1e-12, x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - y_min) / max(1e-12, y_max - y_min) * plot_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#fbfaf7'/>",
        f"<text x='28' y='34' font-size='22' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif' font-weight='700'>{escape(split)} 回撤-收益权衡</text>",
        f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='white' stroke='#d7d7d7'/>",
    ]
    for tick in [i / 100 for i in range(int(x_min * 100), int(x_max * 100) + 1, 5)]:
        px = sx(tick)
        parts.append(f"<line x1='{px:.1f}' y1='{top}' x2='{px:.1f}' y2='{top + plot_h}' stroke='#eeeeee'/>")
        parts.append(f"<text x='{px:.1f}' y='{top + plot_h + 24}' text-anchor='middle' font-size='11' font-family='Arial'>{tick:.0%}</text>")
    y_tick_count = 6
    for i in range(y_tick_count + 1):
        tick = y_min + (y_max - y_min) * i / y_tick_count
        py = sy(tick)
        parts.append(f"<line x1='{left}' y1='{py:.1f}' x2='{left + plot_w}' y2='{py:.1f}' stroke='#eeeeee'/>")
        parts.append(f"<text x='{left - 10}' y='{py + 4:.1f}' text-anchor='end' font-size='11' font-family='Arial'>{tick:.0%}</text>")
    parts.append(f"<text x='{left + plot_w / 2}' y='{height - 28}' text-anchor='middle' font-size='13' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif'>最大回撤绝对值，越左越好</text>")
    parts.append(f"<text transform='translate(24 {top + plot_h / 2}) rotate(-90)' text-anchor='middle' font-size='13' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif'>总收益，越高越好</text>")
    for _, row in sub.iterrows():
        radius = 4.0 + min(8.0, max(0.0, float(size.loc[row.name])) * 7.0)
        profile = str(row["risk_profile"])
        color = _scatter_color(profile)
        px, py = sx(abs(float(row["max_drawdown"]))), sy(float(row["total_return"]))
        label = f"{row['model']} / {row['variant']} / {profile}"
        parts.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{radius:.1f}' fill='{color}' fill-opacity='0.78' stroke='#202124' stroke-width='0.6'><title>{escape(label)}</title></circle>")
    legend_x = left + plot_w + 28
    parts.append(f"<text x='{legend_x}' y='{top}' font-size='14' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif' font-weight='700'>风控配置</text>")
    for idx, profile in enumerate(sorted(sub["risk_profile"].astype(str).unique())):
        y0 = top + 26 + idx * 24
        parts.append(f"<circle cx='{legend_x + 8}' cy='{y0 - 4}' r='6' fill='{_scatter_color(profile)}'/>")
        parts.append(f"<text x='{legend_x + 22}' y='{y0}' font-size='12' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif'>{escape(profile)}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path)


def _write_report(out_root: Path, summary: dict[str, Any], selected: pd.DataFrame, frontier: pd.DataFrame) -> str:
    lines = [
        "# 风控收益回撤权衡报告",
        "",
        "## 协议",
        "",
        "- valid 集用于选择风控参数，test 集只做外推验证。",
        f"- valid 最大回撤约束：`{float(summary['max_drawdown_limit']):.1%}`。",
        "- 综合分会惩罚超过回撤上限、换手过高、平均仓位过低。",
        "",
        "## valid 综合分排名",
        "",
        _markdown_table(
            selected.head(12),
            [
                "model",
                "variant",
                "risk_profile",
                "risk_return_score_valid",
                "total_return_valid",
                "max_drawdown_valid",
                "sharpe_valid",
                "avg_gross_exposure_valid",
                "total_return_test",
                "max_drawdown_test",
                "sharpe_test",
            ],
        ),
        "",
        "## valid Pareto 前沿",
        "",
        _markdown_table(
            frontier.head(16),
            ["model", "variant", "risk_profile", "total_return", "max_drawdown", "sharpe", "avg_gross_exposure", "avg_turnover"],
        ),
        "",
        "## 图表",
        "",
    ]
    for key, value in summary.get("plots", {}).items():
        rel = Path(value).relative_to(out_root).as_posix()
        lines.append(f"- {key}：`{rel}`")
    path = out_root / "risk_sweep_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run_cli() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/strategy")
    parser.add_argument("--run-name", default="strategy_risk_sweep")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--strategy-registry", default=DEFAULT_STRATEGY_REGISTRY)
    parser.add_argument("--strategy-run", default=None)
    parser.add_argument("--model-registry", default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", choices=["valid", "test"], default=None)
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--risk-profiles", nargs="+", default=None)
    parser.add_argument("--max-drawdown-limit", type=float, default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--execution-price-model", choices=["close_to_close", "close_with_slippage"], default=None)
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--return-col", default=None)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--feature-path", default=None)
    parser.add_argument("--feature-columns", nargs="+", default=None)
    parser.add_argument("--no-feature-merge", action="store_true")
    parser.add_argument("--trade-constraints-path", default=None)
    parser.add_argument("--min-amount-mean-20", type=float, default=None)
    parser.add_argument("--no-trade-constraints", action="store_true")
    args = parser.parse_args()

    strategy_cfg: dict[str, Any] = {}
    if args.strategy_run:
        try:
            strategy_registry = load_registry(args.strategy_registry)
            strategy_cfg = resolve_strategy_run(strategy_registry, args.strategy_run, source=args.strategy_registry)
        except ValueError as exc:
            parser.error(str(exc))
        args.out_root = args.out_root if args.out_root != "outputs/strategy" else str(strategy_cfg.get("out_root", args.out_root))
        if args.run_name == "strategy_risk_sweep":
            args.run_name = str(strategy_cfg.get("run_name", args.strategy_run))
        args.model_registry = str(strategy_cfg.get("model_registry", args.model_registry))
        args.models = args.models or list(strategy_cfg.get("models", []))
        args.splits = args.splits or list(strategy_cfg.get("splits", []))
        args.strategies = args.strategies or list(strategy_cfg.get("strategies", [])) or None
        args.risk_profiles = args.risk_profiles or list(strategy_cfg.get("risk_profiles", [])) or None
        args.max_drawdown_limit = args.max_drawdown_limit if args.max_drawdown_limit is not None else strategy_cfg.get("max_drawdown_limit")
        args.feature_set = args.feature_set or strategy_cfg.get("feature_set")
        args.transaction_cost_bps = args.transaction_cost_bps if args.transaction_cost_bps is not None else strategy_cfg.get("transaction_cost_bps")
        args.slippage_bps = args.slippage_bps if args.slippage_bps is not None else strategy_cfg.get("slippage_bps")
        args.execution_price_model = args.execution_price_model or strategy_cfg.get("execution_price_model")
        args.score_col = args.score_col or strategy_cfg.get("score_col")
        args.return_col = args.return_col or strategy_cfg.get("return_col")
        trade_constraints = strategy_cfg.get("trade_constraints", {}) if isinstance(strategy_cfg.get("trade_constraints", {}), dict) else {}
        if not args.no_trade_constraints and trade_constraints.get("enabled", False):
            args.trade_constraints_path = args.trade_constraints_path or trade_constraints.get("path")
            args.min_amount_mean_20 = (
                args.min_amount_mean_20 if args.min_amount_mean_20 is not None else trade_constraints.get("min_amount_mean_20")
            )

    args.models = args.models or ["final", "lgb_top40"]
    args.splits = args.splits or ["valid", "test"]
    args.transaction_cost_bps = float(5.0 if args.transaction_cost_bps is None else args.transaction_cost_bps)
    args.slippage_bps = float(0.0 if args.slippage_bps is None else args.slippage_bps)
    args.execution_price_model = args.execution_price_model or ("close_with_slippage" if args.slippage_bps > 0 else "close_to_close")
    args.min_amount_mean_20 = float(0.0 if args.min_amount_mean_20 is None else args.min_amount_mean_20)
    args.max_drawdown_limit = float(-0.25 if args.max_drawdown_limit is None else args.max_drawdown_limit)
    args.score_col = args.score_col or "pred"
    args.return_col = args.return_col or "label_1d"
    args.feature_set = args.feature_set or "risk_default"
    use_trade_constraints = bool(args.trade_constraints_path) and not args.no_trade_constraints
    try:
        risk_profiles = resolve_risk_profiles(args.risk_profiles)
        registry = load_model_registry(args.model_registry)
        unknown_models = sorted(set(args.models) - set(registered_model_names(registry)))
        if unknown_models:
            parser.error(
                "unknown --models value(s): "
                + ", ".join(unknown_models)
                + "; registered models: "
                + ", ".join(registered_model_names(registry))
            )
        feature_path: str | None = None
        feature_columns: list[str] = []
        if not args.no_feature_merge:
            feature_path, feature_columns = resolve_feature_set(registry, args.feature_set)
            if args.feature_path:
                feature_path = args.feature_path
            if args.feature_columns:
                feature_columns = list(args.feature_columns)
    except ValueError as exc:
        parser.error(str(exc))

    sample_grid = build_strategy_grid(
        cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        execution_price_model=args.execution_price_model,
        enforce_buy_constraints=use_trade_constraints,
    )
    sample_grid = _filter_grid(sample_grid, args.strategies, parser)
    strategies = [name for name, _ in sample_grid]

    out_root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    write_run_metadata(
        out_root,
        command="strategy-risk-sweep",
        args=args,
        inputs={"strategy_run": args.strategy_run, "strategy_config": strategy_cfg},
        registry_paths=[args.model_registry, args.strategy_registry],
    )

    rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    for model_name in args.models:
        for split in args.splits:
            pred_path = resolve_prediction_path(registry, model_name, split)
            print(json.dumps({"stage": "load", "model": model_name, "split": split, "path": pred_path}, ensure_ascii=False), flush=True)
            df = load_prediction_data(pred_path, score_col=args.score_col, return_col=args.return_col)
            if not args.no_feature_merge:
                df = merge_feature_columns(df, feature_path, feature_columns)
            if use_trade_constraints:
                df, constraint_stats = merge_trade_constraint_columns(
                    df,
                    args.trade_constraints_path,
                    min_amount_mean_20=args.min_amount_mean_20,
                )
                constraint_rows.append({"model": model_name, "split": split, **constraint_stats})
            for profile in risk_profiles:
                grid = build_strategy_grid(
                    cost_bps=args.transaction_cost_bps,
                    slippage_bps=args.slippage_bps,
                    execution_price_model=args.execution_price_model,
                    enforce_buy_constraints=use_trade_constraints,
                    config_overrides=profile["overrides"],
                )
                grid = _filter_grid(grid, strategies, parser)
                for variant, cfg in grid:
                    cfg = cfg.__class__(
                        **{
                            **cfg.__dict__,
                            "score_col": args.score_col,
                            "return_col": args.return_col,
                            "enforce_buy_constraints": use_trade_constraints,
                        }
                    )
                    result = run_strategy(df, cfg, name=variant)
                    metrics = dict(result["metrics"])
                    metrics["name"] = f"{model_name}__{variant}__{profile['name']}"
                    metrics["model"] = model_name
                    metrics["split"] = split
                    metrics["variant"] = variant
                    metrics["risk_profile"] = profile["name"]
                    metrics["risk_profile_label"] = profile["label"]
                    metrics["pred_path"] = pred_path
                    rows.append(metrics)
                    print(
                        json.dumps(
                            {
                                "model": model_name,
                                "split": split,
                                "strategy": variant,
                                "risk_profile": profile["name"],
                                "sharpe": metrics.get("sharpe"),
                                "max_drawdown": metrics.get("max_drawdown"),
                                "total_return": metrics.get("total_return"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    metrics_df = add_risk_return_score(pd.DataFrame(rows), max_drawdown_limit=args.max_drawdown_limit)
    metrics_path = out_root / "risk_sweep_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    selected = select_valid_risk_return(metrics_df, top_n=24)
    selected_path = out_root / "risk_sweep_selected.csv"
    selected.to_csv(selected_path, index=False)
    frontier = pareto_frontier(metrics_df, split="valid")
    frontier_path = out_root / "risk_sweep_pareto_valid.csv"
    frontier.to_csv(frontier_path, index=False)
    plots = {
        "valid_risk_return_scatter": write_risk_return_scatter(metrics_df, out_root, split="valid"),
        "test_risk_return_scatter": write_risk_return_scatter(metrics_df, out_root, split="test"),
    }
    summary: dict[str, Any] = {
        "out_root": str(out_root),
        "run_name": args.run_name,
        "model_registry": args.model_registry,
        "models": args.models,
        "splits": args.splits,
        "strategies": strategies,
        "risk_profiles": [profile["name"] for profile in risk_profiles],
        "max_drawdown_limit": float(args.max_drawdown_limit),
        "transaction_cost_bps": float(args.transaction_cost_bps),
        "slippage_bps": float(args.slippage_bps),
        "score_col": args.score_col,
        "return_col": args.return_col,
        "feature_set": None if args.no_feature_merge else args.feature_set,
        "feature_path": feature_path,
        "feature_columns": feature_columns,
        "trade_constraints": {
            "enabled": use_trade_constraints,
            "path": args.trade_constraints_path,
            "min_amount_mean_20": float(args.min_amount_mean_20),
        },
        "metrics_csv": str(metrics_path),
        "selected_csv": str(selected_path),
        "pareto_valid_csv": str(frontier_path),
        "plots": plots,
    }
    if constraint_rows:
        constraint_path = out_root / "trade_constraint_summary.csv"
        pd.DataFrame(constraint_rows).to_csv(constraint_path, index=False)
        summary["trade_constraint_summary_csv"] = str(constraint_path)
    summary["report_md"] = _write_report(out_root, summary, selected, frontier)
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"saved_summary": str(out_root / "summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
