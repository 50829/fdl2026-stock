from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from src.strategy import (
    align_benchmark_to_dates,
    build_equal_weight_benchmark,
    build_index_weight_benchmark,
    build_strategy_grid,
    load_prediction_data,
    load_price_benchmark,
    merge_feature_columns,
    merge_trade_constraint_columns,
    run_strategy,
    write_strategy_outputs,
    write_split_plots,
)
from src.strategy.reporting import write_report_artifacts
from src.utils import DEFAULT_STRATEGY_REGISTRY, load_registry, make_run_dir, read_yaml, resolve_strategy_run, write_run_metadata


DEFAULT_MODEL_REGISTRY = "configs/registry/models.yaml"


def load_model_registry(path: str | Path = DEFAULT_MODEL_REGISTRY) -> dict[str, Any]:
    registry = read_yaml(path)
    models = registry.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"model registry `{path}` must define a non-empty `models` mapping")
    feature_sets = registry.get("feature_sets", {})
    if feature_sets is not None and not isinstance(feature_sets, dict):
        raise ValueError(f"model registry `{path}` has invalid `feature_sets`; expected a mapping")
    return registry


def registered_model_names(registry: dict[str, Any]) -> list[str]:
    return sorted(str(name) for name in registry["models"])


def resolve_prediction_path(registry: dict[str, Any], model_name: str, split: str) -> str:
    models = registry["models"]
    if model_name not in models:
        choices = ", ".join(registered_model_names(registry))
        raise ValueError(f"unknown model `{model_name}`; registered models: {choices}")
    model_cfg = models[model_name]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"model `{model_name}` must be a mapping")
    predictions = model_cfg.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError(f"model `{model_name}` must define a `predictions` mapping")
    if split not in predictions:
        choices = ", ".join(sorted(str(name) for name in predictions))
        raise ValueError(f"model `{model_name}` has no split `{split}`; available splits: {choices}")
    return str(predictions[split])


def resolve_feature_set(registry: dict[str, Any], feature_set: str) -> tuple[str, list[str]]:
    feature_sets = registry.get("feature_sets", {})
    if not isinstance(feature_sets, dict) or feature_set not in feature_sets:
        choices = ", ".join(sorted(str(name) for name in feature_sets)) or "<none>"
        raise ValueError(f"unknown feature set `{feature_set}`; registered feature sets: {choices}")
    cfg = feature_sets[feature_set]
    if not isinstance(cfg, dict):
        raise ValueError(f"feature set `{feature_set}` must be a mapping")
    path = cfg.get("path") or cfg.get("feature_path")
    columns = cfg.get("columns")
    if not path:
        raise ValueError(f"feature set `{feature_set}` must define `path`")
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise ValueError(f"feature set `{feature_set}` must define `columns` as a list of strings")
    return str(path), list(columns)


def _select_best(valid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(valid_rows)
    if df.empty:
        return {}
    df = df[~df["strategy"].astype(str).str.startswith("benchmark")]
    if df.empty:
        return {}
    ranked = df.sort_values(
        ["sharpe", "max_drawdown", "total_return", "avg_turnover"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return ranked.iloc[0].to_dict()


def _metric_table(rows: list[dict[str, Any]], top_n: int | None = None) -> str:
    if not rows:
        return "_No rows._\n"
    df = pd.DataFrame(rows)
    cols = ["name", "strategy", "total_return", "annual_return", "sharpe", "max_drawdown", "avg_turnover", "avg_n_holdings"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df = df.sort_values(["sharpe", "total_return"], ascending=[False, False], kind="mergesort")
    if top_n is not None:
        df = df.head(top_n)
    for col in ["total_return", "annual_return", "sharpe", "max_drawdown", "avg_turnover", "avg_n_holdings"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    headers = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def _write_report(out_root: Path, summary: dict[str, Any], all_rows: list[dict[str, Any]], benchmark_note: str) -> None:
    constraints = summary.get("trade_constraints", {}) if isinstance(summary.get("trade_constraints", {}), dict) else {}
    risk_controls = summary.get("risk_controls", {}) if isinstance(summary.get("risk_controls", {}), dict) else {}
    lines = [
        "# 策略回测报告",
        "",
        "## 回测协议",
        "",
        "- 选股信号只使用模型 `pred`。",
        "- 已实现收益 `label_1d` 只用于事后收益计算和历史风险估计。",
        f"- 成交假设：`{summary.get('execution_price_model', 'close_to_close')}`。",
        f"- 基础交易成本：`{float(summary.get('transaction_cost_bps', 0.0)):.2f} bps`。",
        f"- 额外滑点成本：`{float(summary.get('slippage_bps', 0.0)):.2f} bps`。",
        f"- 买入约束：{'启用' if constraints.get('enabled') else '未启用'}。",
        f"- 市场压力降仓：{'启用' if risk_controls.get('apply_market_stress_deleveraging') else '未启用'}。",
        f"- 组合回撤控制：{'启用' if risk_controls.get('apply_drawdown_control') else '未启用'}。",
        "- 权益曲线图默认使用 log10 净值坐标。",
        "- 主报告按概览、valid Sharpe、策略族、具体策略参数和模型视角拆分。",
        f"- 基准：{benchmark_note}",
        "",
    ]
    reporting = summary.get("reporting", {}) if isinstance(summary.get("reporting", {}), dict) else {}
    if reporting:
        lines.extend(
            [
                "## 报告文件",
                "",
                f"- HTML 报告：`{reporting.get('report_html', '')}`",
                f"- 指标长表：`{reporting.get('metrics_long', '')}`",
                f"- 权益长表：`{reporting.get('equity_long', '')}`",
                "",
            ]
        )
    df = pd.DataFrame(all_rows)
    for model_name, model_info in summary["models"].items():
        lines.extend([f"## {model_name}", ""])
        for split, split_info in model_info.items():
            if df.empty:
                rows: list[dict[str, Any]] = []
            else:
                rows = df[(df["model"] == model_name) & (df["split"] == split)].to_dict("records")
            lines.extend(
                [
                    f"### {split}",
                    "",
                    f"- 指标 CSV：`{split_info['metrics_csv']}`",
                    f"- 概览图：`{split_info['plots']['overview']}`",
                    f"- valid Sharpe 图：`{split_info['plots']['top_valid_sharpe']}`",
                    f"- 全量调试图：`{split_info['plots']['all_debug']}`",
                    "",
                    _metric_table(rows, top_n=10),
                    "",
                ]
            )
    (out_root / "strategy_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_cli() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/strategy")
    parser.add_argument("--run-name", default="strategy_backtest")
    parser.add_argument("--no-timestamp", action="store_true", help="Write to <out-root>/<run-name> instead of timestamping the run directory.")
    parser.add_argument("--strategy-registry", default=DEFAULT_STRATEGY_REGISTRY)
    parser.add_argument("--strategy-run", default=None, help="Load defaults from configs/registry/strategies.yaml.")
    parser.add_argument("--model-registry", default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", choices=["valid", "test"], default=None)
    parser.add_argument("--strategies", nargs="+", default=None, help="Optional strategy variant names to run from the grid.")
    parser.add_argument("--transaction-cost-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--execution-price-model", choices=["close_to_close", "close_with_slippage"], default=None)
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--return-col", default=None)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--feature-path", default=None, help="Override the feature path from --feature-set.")
    parser.add_argument("--feature-columns", nargs="+", default=None, help="Override feature columns from --feature-set.")
    parser.add_argument("--no-feature-merge", action="store_true")
    parser.add_argument("--trade-constraints-path", default=None, help="Optional universe parquet with in_universe/is_st/passes_liquidity fields.")
    parser.add_argument("--min-amount-mean-20", type=float, default=None)
    parser.add_argument("--no-trade-constraints", action="store_true")
    parser.add_argument("--market-stress-deleveraging", action="store_true")
    parser.add_argument("--market-window", type=int, default=None)
    parser.add_argument("--market-stress-threshold", type=float, default=None)
    parser.add_argument("--market-stress-lag", type=int, default=None)
    parser.add_argument("--stress-gross-exposure", type=float, default=None)
    parser.add_argument("--drawdown-control", action="store_true")
    parser.add_argument("--drawdown-warning-threshold", type=float, default=None)
    parser.add_argument("--drawdown-warning-exposure", type=float, default=None)
    parser.add_argument("--drawdown-cut-threshold", type=float, default=None)
    parser.add_argument("--drawdown-cut-exposure", type=float, default=None)
    parser.add_argument("--drawdown-stop-threshold", type=float, default=None)
    parser.add_argument("--drawdown-stop-exposure", type=float, default=None)
    parser.add_argument("--benchmark-path", default=None, help="Optional CSV/parquet index benchmark with trade_date and close/equity/return.")
    parser.add_argument("--benchmark-name", default="benchmark_index")
    parser.add_argument("--index-weight-path", default="data/raw/index_weight.zip")
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--no-index-weight-benchmark", action="store_true")
    parser.add_argument("--no-equal-weight-benchmark", action="store_true")
    parser.add_argument("--linear-scale", action="store_true", help="Use linear equity scale for comparison SVGs.")
    args = parser.parse_args()
    strategy_cfg: dict[str, Any] = {}
    if args.strategy_run:
        try:
            strategy_registry = load_registry(args.strategy_registry)
            strategy_cfg = resolve_strategy_run(strategy_registry, args.strategy_run, source=args.strategy_registry)
        except ValueError as exc:
            parser.error(str(exc))
        args.out_root = args.out_root if args.out_root != "outputs/strategy" else str(strategy_cfg.get("out_root", args.out_root))
        args.run_name = args.run_name if args.run_name != "strategy_backtest" else str(strategy_cfg.get("run_name", args.run_name))
        args.model_registry = str(strategy_cfg.get("model_registry", args.model_registry))
        args.models = args.models or list(strategy_cfg.get("models", []))
        args.splits = args.splits or list(strategy_cfg.get("splits", []))
        args.strategies = args.strategies or list(strategy_cfg.get("strategies", [])) or None
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
        risk_controls = strategy_cfg.get("risk_controls", {}) if isinstance(strategy_cfg.get("risk_controls", {}), dict) else {}
        args.market_stress_deleveraging = args.market_stress_deleveraging or bool(risk_controls.get("market_stress_deleveraging", False))
        args.market_window = args.market_window if args.market_window is not None else risk_controls.get("market_window")
        args.market_stress_threshold = args.market_stress_threshold if args.market_stress_threshold is not None else risk_controls.get("market_stress_threshold")
        args.market_stress_lag = args.market_stress_lag if args.market_stress_lag is not None else risk_controls.get("market_stress_lag")
        args.stress_gross_exposure = args.stress_gross_exposure if args.stress_gross_exposure is not None else risk_controls.get("stress_gross_exposure")
        args.drawdown_control = args.drawdown_control or bool(risk_controls.get("drawdown_control", False))
        args.drawdown_warning_threshold = (
            args.drawdown_warning_threshold if args.drawdown_warning_threshold is not None else risk_controls.get("drawdown_warning_threshold")
        )
        args.drawdown_warning_exposure = (
            args.drawdown_warning_exposure if args.drawdown_warning_exposure is not None else risk_controls.get("drawdown_warning_exposure")
        )
        args.drawdown_cut_threshold = args.drawdown_cut_threshold if args.drawdown_cut_threshold is not None else risk_controls.get("drawdown_cut_threshold")
        args.drawdown_cut_exposure = args.drawdown_cut_exposure if args.drawdown_cut_exposure is not None else risk_controls.get("drawdown_cut_exposure")
        args.drawdown_stop_threshold = args.drawdown_stop_threshold if args.drawdown_stop_threshold is not None else risk_controls.get("drawdown_stop_threshold")
        args.drawdown_stop_exposure = args.drawdown_stop_exposure if args.drawdown_stop_exposure is not None else risk_controls.get("drawdown_stop_exposure")
        benchmarks = strategy_cfg.get("benchmarks", {}) if isinstance(strategy_cfg.get("benchmarks", {}), dict) else {}
        index_weight = benchmarks.get("index_weight", {}) if isinstance(benchmarks.get("index_weight", {}), dict) else {}
        if args.index_weight_path == "data/raw/index_weight.zip":
            args.index_weight_path = str(index_weight.get("weight_path", args.index_weight_path))
        if args.index_code == "000300.SH":
            args.index_code = str(index_weight.get("index_code", args.index_code))

    args.models = args.models or ["final", "lgb_top40"]
    args.splits = args.splits or ["valid", "test"]
    args.transaction_cost_bps = float(5.0 if args.transaction_cost_bps is None else args.transaction_cost_bps)
    args.slippage_bps = float(0.0 if args.slippage_bps is None else args.slippage_bps)
    args.execution_price_model = args.execution_price_model or ("close_with_slippage" if args.slippage_bps > 0 else "close_to_close")
    args.min_amount_mean_20 = float(0.0 if args.min_amount_mean_20 is None else args.min_amount_mean_20)
    args.score_col = args.score_col or "pred"
    args.return_col = args.return_col or "label_1d"
    args.feature_set = args.feature_set or "risk_default"
    use_trade_constraints = bool(args.trade_constraints_path) and not args.no_trade_constraints
    risk_control_overrides = {
        "apply_market_stress_deleveraging": bool(args.market_stress_deleveraging),
        "apply_drawdown_control": bool(args.drawdown_control),
    }
    for arg_name, cfg_name in [
        ("market_window", "market_window"),
        ("market_stress_threshold", "market_stress_threshold"),
        ("market_stress_lag", "market_stress_lag"),
        ("stress_gross_exposure", "stress_gross_exposure"),
        ("drawdown_warning_threshold", "drawdown_warning_threshold"),
        ("drawdown_warning_exposure", "drawdown_warning_exposure"),
        ("drawdown_cut_threshold", "drawdown_cut_threshold"),
        ("drawdown_cut_exposure", "drawdown_cut_exposure"),
        ("drawdown_stop_threshold", "drawdown_stop_threshold"),
        ("drawdown_stop_exposure", "drawdown_stop_exposure"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            risk_control_overrides[cfg_name] = value

    try:
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

    out_root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    write_run_metadata(
        out_root,
        command="strategy-backtest",
        args=args,
        inputs={"strategy_run": args.strategy_run, "strategy_config": strategy_cfg},
        registry_paths=[args.model_registry, args.strategy_registry],
    )
    grid = build_strategy_grid(cost_bps=args.transaction_cost_bps)
    if use_trade_constraints or args.slippage_bps or args.execution_price_model != "close_to_close":
        grid = build_strategy_grid(
            cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            execution_price_model=args.execution_price_model,
            enforce_buy_constraints=use_trade_constraints,
            config_overrides=risk_control_overrides,
        )
    elif risk_control_overrides:
        grid = build_strategy_grid(cost_bps=args.transaction_cost_bps, config_overrides=risk_control_overrides)
    if args.strategies:
        selected = set(str(name) for name in args.strategies)
        known = {name for name, _ in grid}
        unknown = sorted(selected - known)
        if unknown:
            parser.error("unknown --strategies value(s): " + ", ".join(unknown) + "; known strategies: " + ", ".join(sorted(known)))
        grid = [(name, cfg) for name, cfg in grid if name in selected]
    summary: dict[str, Any] = {
        "out_root": str(out_root),
        "out_parent": args.out_root,
        "run_name": args.run_name,
        "timestamped": not args.no_timestamp,
        "model_registry": args.model_registry,
        "transaction_cost_bps": float(args.transaction_cost_bps),
        "slippage_bps": float(args.slippage_bps),
        "total_cost_bps": float(args.transaction_cost_bps + args.slippage_bps),
        "execution_price_model": args.execution_price_model,
        "trade_constraints": {
            "enabled": use_trade_constraints,
            "path": args.trade_constraints_path,
            "min_amount_mean_20": float(args.min_amount_mean_20),
        },
        "risk_controls": risk_control_overrides,
        "strategies": [name for name, _ in grid],
        "score_col": args.score_col,
        "return_col": args.return_col,
        "feature_set": None if args.no_feature_merge else args.feature_set,
        "feature_path": feature_path,
        "feature_columns": feature_columns,
        "plot_scale": "linear" if args.linear_scale else "log",
        "benchmark_path": args.benchmark_path,
        "index_weight_path": args.index_weight_path,
        "index_code": args.index_code,
        "models": {},
    }
    all_rows: list[dict[str, Any]] = []
    valid_rows_by_model: dict[str, list[dict[str, Any]]] = {}
    aggregate_rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in args.splits}
    aggregate_curves_by_split: dict[str, dict[str, pd.DataFrame]] = {split: {} for split in args.splits}
    aggregate_benchmarks_seen: dict[str, set[str]] = {split: set() for split in args.splits}
    constraint_rows: list[dict[str, Any]] = []
    benchmark_notes: list[str] = []
    if args.benchmark_path:
        benchmark_notes.append(f"external benchmark `{args.benchmark_name}` from `{args.benchmark_path}`")
    index_weight_path = Path(args.index_weight_path)
    use_index_weight = not args.no_index_weight_benchmark and index_weight_path.exists()
    if use_index_weight:
        benchmark_notes.append(f"index weight benchmark `{args.index_code}` from `{args.index_weight_path}`")
    elif not args.no_index_weight_benchmark:
        benchmark_notes.append(f"index weight benchmark skipped because `{args.index_weight_path}` does not exist")
    if not args.no_equal_weight_benchmark:
        benchmark_notes.append("equal-weight universe baseline from prediction file")
    if use_trade_constraints:
        benchmark_notes.append(f"buy constraints from `{args.trade_constraints_path}`")
    benchmark_note = "; ".join(benchmark_notes) if benchmark_notes else "none"

    for model_name in args.models:
        summary["models"][model_name] = {}
        for split in args.splits:
            pred_path = resolve_prediction_path(registry, model_name, split)
            print(json.dumps({"stage": "load", "model": model_name, "split": split, "path": pred_path}, ensure_ascii=False), flush=True)
            df = load_prediction_data(pred_path, score_col=args.score_col, return_col=args.return_col)
            if not args.no_feature_merge:
                df = merge_feature_columns(df, feature_path, feature_columns)
            constraint_stats: dict[str, object] | None = None
            if use_trade_constraints:
                df, constraint_stats = merge_trade_constraint_columns(
                    df,
                    args.trade_constraints_path,
                    min_amount_mean_20=args.min_amount_mean_20,
                )
                constraint_rows.append({"model": model_name, "split": split, **constraint_stats})
            split_rows: list[dict[str, Any]] = []
            curves: dict[str, pd.DataFrame] = {}
            benchmark_rows: list[dict[str, Any]] = []
            if args.benchmark_path:
                benchmark = load_price_benchmark(args.benchmark_path, args.benchmark_name)
                benchmark = align_benchmark_to_dates(benchmark, df["trade_date"].unique().tolist())
                benchmark_dir = out_root / model_name / split / args.benchmark_name
                write_strategy_outputs(benchmark, benchmark_dir)
                bm_metrics = dict(benchmark["metrics"])
                bm_metrics["model"] = model_name
                bm_metrics["split"] = split
                bm_metrics["pred_path"] = pred_path
                benchmark_rows.append(bm_metrics)
                curves[args.benchmark_name] = benchmark["curve"]
                if args.benchmark_name not in aggregate_benchmarks_seen[split]:
                    aggregate_rows_by_split[split].append(dict(bm_metrics))
                    aggregate_curves_by_split[split][args.benchmark_name] = benchmark["curve"]
                    aggregate_benchmarks_seen[split].add(args.benchmark_name)
            if use_index_weight:
                benchmark = build_index_weight_benchmark(
                    df,
                    weight_path=args.index_weight_path,
                    index_code=args.index_code,
                    return_col=args.return_col,
                )
                benchmark_name = str(benchmark["metrics"]["name"])
                benchmark_dir = out_root / model_name / split / benchmark_name
                write_strategy_outputs(benchmark, benchmark_dir)
                bm_metrics = dict(benchmark["metrics"])
                bm_metrics["model"] = model_name
                bm_metrics["split"] = split
                bm_metrics["pred_path"] = pred_path
                benchmark_rows.append(bm_metrics)
                curves[benchmark_name] = benchmark["curve"]
                if benchmark_name not in aggregate_benchmarks_seen[split]:
                    aggregate_rows_by_split[split].append(dict(bm_metrics))
                    aggregate_curves_by_split[split][benchmark_name] = benchmark["curve"]
                    aggregate_benchmarks_seen[split].add(benchmark_name)
            if not args.no_equal_weight_benchmark:
                benchmark = build_equal_weight_benchmark(df, return_col=args.return_col)
                benchmark_dir = out_root / model_name / split / "benchmark_equal_weight_universe"
                write_strategy_outputs(benchmark, benchmark_dir)
                bm_metrics = dict(benchmark["metrics"])
                bm_metrics["model"] = model_name
                bm_metrics["split"] = split
                bm_metrics["pred_path"] = pred_path
                benchmark_rows.append(bm_metrics)
                curves["benchmark_equal_weight_universe"] = benchmark["curve"]
                if "benchmark_equal_weight_universe" not in aggregate_benchmarks_seen[split]:
                    aggregate_rows_by_split[split].append(dict(bm_metrics))
                    aggregate_curves_by_split[split]["benchmark_equal_weight_universe"] = benchmark["curve"]
                    aggregate_benchmarks_seen[split].add("benchmark_equal_weight_universe")
            for exp_name, cfg in grid:
                cfg = cfg.__class__(
                    **{
                        **cfg.__dict__,
                        "score_col": args.score_col,
                        "return_col": args.return_col,
                        "enforce_buy_constraints": use_trade_constraints,
                    }
                )
                result = run_strategy(df, cfg, name=exp_name)
                exp_dir = out_root / model_name / split / exp_name
                write_strategy_outputs(result, exp_dir)
                metrics = dict(result["metrics"])
                metrics["model"] = model_name
                metrics["split"] = split
                metrics["pred_path"] = pred_path
                split_rows.append(metrics)
                all_rows.append(metrics)
                curves[exp_name] = result["curve"]
                aggregate_name = f"{model_name}__{exp_name}"
                aggregate_metrics = dict(metrics)
                aggregate_metrics["name"] = aggregate_name
                aggregate_rows_by_split[split].append(aggregate_metrics)
                aggregate_curves_by_split[split][aggregate_name] = result["curve"]
                print(json.dumps({"model": model_name, "split": split, "strategy": exp_name, "metrics": metrics}, ensure_ascii=False), flush=True)
            split_rows.extend(benchmark_rows)
            all_rows.extend(benchmark_rows)
            metrics_df = pd.DataFrame(split_rows)
            split_dir = out_root / model_name / split
            split_dir.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(split_dir / "strategy_metrics.csv", index=False)
            if split == "valid":
                valid_rows_by_model[model_name] = list(split_rows)
            plot_paths = write_split_plots(
                curves,
                split_rows,
                split_dir,
                f"{model_name} {split} strategy equity",
                log_scale=not args.linear_scale,
                valid_rows=valid_rows_by_model.get(model_name),
            )
            summary["models"][model_name][split] = {
                "pred_path": pred_path,
                "rows": int(len(df)),
                "metrics_csv": str(split_dir / "strategy_metrics.csv"),
                "plots": plot_paths,
                "trade_constraints": constraint_stats,
                "best_by_valid_protocol": _select_best(split_rows) if split == "valid" else None,
            }

    summary["aggregate"] = {}
    valid_aggregate_rows = aggregate_rows_by_split.get("valid")
    for split in args.splits:
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = aggregate_rows_by_split[split]
        pd.DataFrame(rows).to_csv(split_dir / "strategy_metrics.csv", index=False)
        plots = write_split_plots(
            aggregate_curves_by_split[split],
            rows,
            split_dir,
            f"{split} label1d vs label5d strategy equity",
            log_scale=not args.linear_scale,
            valid_rows=valid_aggregate_rows,
        )
        summary["aggregate"][split] = {
            "metrics_csv": str(split_dir / "strategy_metrics.csv"),
            "plots": plots,
            "best_by_valid_protocol": _select_best(rows) if split == "valid" else None,
        }

    summary["reporting"] = write_report_artifacts(
        out_root,
        aggregate_rows_by_split,
        aggregate_curves_by_split,
        benchmark_note=benchmark_note,
        title=args.run_name,
    )
    if constraint_rows:
        constraint_path = out_root / "trade_constraint_summary.csv"
        pd.DataFrame(constraint_rows).to_csv(constraint_path, index=False)
        summary["trade_constraint_summary_csv"] = str(constraint_path)

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_report(out_root, summary, all_rows, benchmark_note)
    print(json.dumps({"saved_summary": str(out_root / "summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
