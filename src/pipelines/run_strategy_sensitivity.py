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


DEFAULT_TOTAL_COST_GRID = [5.0, 10.0, 20.0, 50.0]


def parse_total_cost_grid(values: Any | None) -> list[float]:
    if values is None:
        return list(DEFAULT_TOTAL_COST_GRID)
    if isinstance(values, (str, int, float)):
        values = [values]
    parsed: list[float] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            if part:
                parsed.append(float(part))
    if not parsed:
        raise ValueError("total cost grid is empty")
    seen: set[float] = set()
    out: list[float] = []
    for value in parsed:
        if value < 0:
            raise ValueError("total cost bps must be non-negative")
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def split_total_cost(total_cost_bps: float, base_transaction_cost_bps: float) -> tuple[float, float]:
    transaction_cost_bps = min(float(base_transaction_cost_bps), float(total_cost_bps))
    slippage_bps = max(0.0, float(total_cost_bps) - transaction_cost_bps)
    return transaction_cost_bps, slippage_bps


def select_best_by_valid_cost(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    required = {"split", "total_cost_bps", "model", "variant", "sharpe", "max_drawdown", "avg_turnover"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics missing required columns: {sorted(missing)}")
    valid = metrics[metrics["split"].astype(str) == "valid"].copy()
    test = metrics[metrics["split"].astype(str) == "test"].copy()
    if valid.empty:
        return pd.DataFrame()
    valid = valid.sort_values(
        ["total_cost_bps", "sharpe", "max_drawdown", "total_return", "avg_turnover"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    best_valid = valid.groupby("total_cost_bps", as_index=False, sort=False).head(1)
    keep_cols = [
        "total_cost_bps",
        "model",
        "variant",
        "strategy",
        "final_equity",
        "total_return",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "avg_n_holdings",
    ]
    best_valid = best_valid[[col for col in keep_cols if col in best_valid.columns]].copy()
    best_valid = best_valid.rename(columns={col: f"{col}_valid" for col in best_valid.columns if col not in {"total_cost_bps", "model", "variant"}})
    if test.empty:
        return best_valid
    test_cols = ["total_cost_bps", "model", "variant", "final_equity", "total_return", "annual_return", "sharpe", "max_drawdown", "avg_turnover", "avg_n_holdings"]
    test = test[[col for col in test_cols if col in test.columns]].rename(
        columns={col: f"{col}_test" for col in test_cols if col not in {"total_cost_bps", "model", "variant"} and col in test.columns}
    )
    return best_valid.merge(test, on=["total_cost_bps", "model", "variant"], how="left")


def _fmt_cost(value: float) -> str:
    return f"{float(value):g}"


def _fmt_metric(value: object, metric: str) -> str:
    if value is None or pd.isna(value):
        return ""
    number = float(value)
    if metric in {"total_return", "annual_return", "max_drawdown"}:
        return f"{number * 100:.1f}%"
    return f"{number:.2f}"


def _heat_color(value: float, lo: float, hi: float, *, higher_is_better: bool = True) -> str:
    if hi <= lo:
        ratio = 0.5
    else:
        ratio = (float(value) - lo) / (hi - lo)
    ratio = max(0.0, min(1.0, ratio))
    if not higher_is_better:
        ratio = 1.0 - ratio
    red = round(244 - 122 * ratio)
    green = round(242 - 58 * (1.0 - ratio))
    blue = round(240 - 142 * ratio)
    return f"rgb({red},{green},{blue})"


def _wrap_label(text: str, max_chars: int = 24, max_lines: int = 3) -> list[str]:
    parts = str(text).replace("__", "_").split("_")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}_{part}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = part
        if len(lines) >= max_lines - 1:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines or [str(text)[:max_chars]]


def _write_heatmap_svg(pivot: pd.DataFrame, path: Path, *, title: str, metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [float(v) for v in pivot.index]
    cols = list(pivot.columns)
    cell_w = 162
    cell_h = 42
    left = 92
    top = 118
    width = left + cell_w * max(1, len(cols)) + 26
    height = top + cell_h * max(1, len(rows)) + 44
    values = pivot.to_numpy(dtype=float)
    finite = values[pd.notna(values)]
    lo = float(finite.min()) if len(finite) else 0.0
    hi = float(finite.max()) if len(finite) else 1.0
    higher_is_better = metric != "avg_turnover"
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#fbfaf7'/>",
        f"<text x='24' y='32' font-size='20' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif' font-weight='700' fill='#202124'>{escape(title)}</text>",
        "<text x='24' y='56' font-size='12' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif' fill='#5f6368'>行是总交易成本 bps，列是模型/策略；颜色越绿表示该指标越好。</text>",
    ]
    for j, col in enumerate(cols):
        x = left + j * cell_w + cell_w / 2
        label = str(col)
        model, _, variant = label.partition(" / ")
        parts.append(f"<text x='{x:.1f}' y='80' text-anchor='middle' font-size='11' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif' fill='#202124'>")
        parts.append(f"<tspan x='{x:.1f}' dy='0' font-weight='700'>{escape(model)}</tspan>")
        for line in _wrap_label(variant or label):
            parts.append(f"<tspan x='{x:.1f}' dy='14'>{escape(line)}</tspan>")
        parts.append("</text>")
    for i, cost in enumerate(rows):
        y = top + i * cell_h
        parts.append(f"<text x='{left - 18}' y='{y + 27}' text-anchor='end' font-size='13' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif' fill='#202124'>{_fmt_cost(cost)}</text>")
        for j, col in enumerate(cols):
            x = left + j * cell_w
            value = pivot.iloc[i, j]
            if pd.isna(value):
                fill = "#eeeeee"
                text = ""
            else:
                fill = _heat_color(float(value), lo, hi, higher_is_better=higher_is_better)
                text = _fmt_metric(value, metric)
            parts.append(f"<rect x='{x}' y='{y}' width='{cell_w - 4}' height='{cell_h - 4}' rx='4' fill='{fill}' stroke='#ffffff'/>")
            parts.append(f"<text x='{x + (cell_w - 4) / 2:.1f}' y='{y + 25}' text-anchor='middle' font-size='13' font-family='Arial,Noto Sans CJK SC,Microsoft YaHei,sans-serif' fill='#202124'>{escape(text)}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_sensitivity_heatmaps(
    metrics: pd.DataFrame,
    out_root: str | Path,
    *,
    metric_names: tuple[str, ...] = ("sharpe", "total_return", "max_drawdown", "avg_turnover"),
) -> dict[str, str]:
    root = Path(out_root)
    plot_dir = root / "plots"
    paths: dict[str, str] = {}
    if metrics.empty:
        return paths
    metrics = metrics.copy()
    metrics["display_name"] = metrics["model"].astype(str) + " / " + metrics["variant"].astype(str)
    order = (
        metrics[["model", "variant", "display_name"]]
        .drop_duplicates()
        .sort_values(["model", "variant"], kind="mergesort")["display_name"]
        .tolist()
    )
    for split in sorted(metrics["split"].dropna().astype(str).unique()):
        sub = metrics[metrics["split"].astype(str) == split]
        for metric in metric_names:
            if metric not in sub.columns:
                continue
            pivot = sub.pivot_table(index="total_cost_bps", columns="display_name", values=metric, aggfunc="max")
            pivot = pivot.reindex(index=sorted(pivot.index.astype(float)), columns=[col for col in order if col in pivot.columns])
            path = plot_dir / f"{split}_{metric}.svg"
            title = f"{split} {metric} 成本敏感性矩阵"
            _write_heatmap_svg(pivot, path, title=title, metric=metric)
            paths[f"{split}_{metric}"] = str(path)
    return paths


def _metric_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "没有结果。\n"
    cols = ["total_cost_bps", "model", "variant", "sharpe_valid", "max_drawdown_valid", "avg_turnover_valid", "sharpe_test", "max_drawdown_test", "avg_turnover_test"]
    show = df[[col for col in cols if col in df.columns]].copy()
    for col in show.columns:
        if col.startswith("max_drawdown"):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.2%}")
        elif col.startswith("sharpe") or col.startswith("avg_turnover"):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    lines = ["| " + " | ".join(show.columns) + " |", "| " + " | ".join("---" for _ in show.columns) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown_report(out_root: Path, summary: dict[str, Any], selected: pd.DataFrame) -> Path:
    lines = [
        "# 策略成本敏感性报告",
        "",
        "## 回测协议",
        "",
        f"- 总成本档位：{', '.join(_fmt_cost(v) + ' bps' for v in summary['total_cost_bps_grid'])}",
        f"- 基础交易成本上限：{float(summary['base_transaction_cost_bps']):.2f} bps。",
        "- 当总成本高于基础交易成本时，差额记为额外滑点。",
        f"- 买入约束：{'启用' if summary['trade_constraints']['enabled'] else '未启用'}。",
        "- valid 集选择参数，test 集只做外推验证。",
        "",
        "## valid 选择结果",
        "",
        _metric_table(selected),
        "",
        "## 图表",
        "",
    ]
    for key, path in summary.get("plots", {}).items():
        rel = Path(path).relative_to(out_root).as_posix()
        lines.append(f"- {key}：`{rel}`")
    report_path = out_root / "sensitivity_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _filter_grid(grid: list[tuple[str, Any]], strategies: list[str] | None, parser: argparse.ArgumentParser) -> list[tuple[str, Any]]:
    if not strategies:
        return grid
    selected = set(str(name) for name in strategies)
    known = {name for name, _ in grid}
    unknown = sorted(selected - known)
    if unknown:
        parser.error("unknown strategy value(s): " + ", ".join(unknown) + "; known strategies: " + ", ".join(sorted(known)))
    return [(name, cfg) for name, cfg in grid if name in selected]


def run_cli() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/strategy")
    parser.add_argument("--run-name", default="strategy_sensitivity")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--strategy-registry", default=DEFAULT_STRATEGY_REGISTRY)
    parser.add_argument("--strategy-run", default=None)
    parser.add_argument("--model-registry", default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", choices=["valid", "test"], default=None)
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--total-cost-bps-grid", nargs="+", default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=None)
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
        if args.run_name == "strategy_sensitivity":
            args.run_name = str(strategy_cfg.get("run_name", args.strategy_run))
        args.model_registry = str(strategy_cfg.get("model_registry", args.model_registry))
        args.models = args.models or list(strategy_cfg.get("models", []))
        args.splits = args.splits or list(strategy_cfg.get("splits", []))
        args.strategies = args.strategies or list(strategy_cfg.get("strategies", [])) or None
        args.total_cost_bps_grid = args.total_cost_bps_grid or strategy_cfg.get("total_cost_bps_grid")
        args.feature_set = args.feature_set or strategy_cfg.get("feature_set")
        args.transaction_cost_bps = args.transaction_cost_bps if args.transaction_cost_bps is not None else strategy_cfg.get("transaction_cost_bps")
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
    args.min_amount_mean_20 = float(0.0 if args.min_amount_mean_20 is None else args.min_amount_mean_20)
    args.score_col = args.score_col or "pred"
    args.return_col = args.return_col or "label_1d"
    args.feature_set = args.feature_set or "risk_default"
    use_trade_constraints = bool(args.trade_constraints_path) and not args.no_trade_constraints
    try:
        total_cost_grid = parse_total_cost_grid(args.total_cost_bps_grid)
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

    sample_fee, sample_slippage = split_total_cost(total_cost_grid[0], args.transaction_cost_bps)
    sample_grid = build_strategy_grid(
        cost_bps=sample_fee,
        slippage_bps=sample_slippage,
        execution_price_model="close_with_slippage" if sample_slippage > 0 else "close_to_close",
        enforce_buy_constraints=use_trade_constraints,
    )
    sample_grid = _filter_grid(sample_grid, args.strategies, parser)
    strategies = [name for name, _ in sample_grid]

    out_root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    write_run_metadata(
        out_root,
        command="strategy-sensitivity",
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
            for total_cost_bps in total_cost_grid:
                transaction_cost_bps, slippage_bps = split_total_cost(total_cost_bps, args.transaction_cost_bps)
                execution_price_model = "close_with_slippage" if slippage_bps > 0 else "close_to_close"
                grid = build_strategy_grid(
                    cost_bps=transaction_cost_bps,
                    slippage_bps=slippage_bps,
                    execution_price_model=execution_price_model,
                    enforce_buy_constraints=use_trade_constraints,
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
                    metrics["name"] = f"{model_name}__{variant}"
                    metrics["model"] = model_name
                    metrics["split"] = split
                    metrics["variant"] = variant
                    metrics["pred_path"] = pred_path
                    metrics["scenario"] = f"total_cost_{_fmt_cost(total_cost_bps)}bps"
                    metrics["base_transaction_cost_bps"] = float(args.transaction_cost_bps)
                    rows.append(metrics)
                    print(
                        json.dumps(
                            {
                                "model": model_name,
                                "split": split,
                                "strategy": variant,
                                "total_cost_bps": total_cost_bps,
                                "sharpe": metrics.get("sharpe"),
                                "max_drawdown": metrics.get("max_drawdown"),
                                "avg_turnover": metrics.get("avg_turnover"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_root / "sensitivity_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    selected = select_best_by_valid_cost(metrics_df)
    selected_path = out_root / "best_by_valid_cost.csv"
    selected.to_csv(selected_path, index=False)
    plots = write_sensitivity_heatmaps(metrics_df, out_root)
    summary: dict[str, Any] = {
        "out_root": str(out_root),
        "run_name": args.run_name,
        "model_registry": args.model_registry,
        "models": args.models,
        "splits": args.splits,
        "strategies": strategies,
        "total_cost_bps_grid": total_cost_grid,
        "base_transaction_cost_bps": float(args.transaction_cost_bps),
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
        "best_by_valid_cost_csv": str(selected_path),
        "plots": plots,
    }
    if constraint_rows:
        constraint_path = out_root / "trade_constraint_summary.csv"
        pd.DataFrame(constraint_rows).to_csv(constraint_path, index=False)
        summary["trade_constraint_summary_csv"] = str(constraint_path)
    report_path = _write_markdown_report(out_root, summary, selected)
    summary["report_md"] = str(report_path)
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"saved_summary": str(out_root / "summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
