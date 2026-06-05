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
    run_strategy,
    write_strategy_outputs,
    write_split_plots,
)
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
    lines = [
        "# Strategy Backtest Report",
        "",
        "## Protocol",
        "",
        "- Selection signal: model `pred` only.",
        "- Realized `label_1d` is used for ex-post returns and historical risk estimation.",
        "- Equity comparison plots use log10 equity scale by default.",
        "- Main report uses split plots: overview, top valid Sharpe, plots by strategy family, and all-strategies debug.",
        f"- Benchmark: {benchmark_note}",
        "",
    ]
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
                    f"- Metrics CSV: `{split_info['metrics_csv']}`",
                    f"- Overview plot: `{split_info['plots']['overview']}`",
                    f"- Top valid Sharpe plot: `{split_info['plots']['top_valid_sharpe']}`",
                    f"- All-strategies debug plot: `{split_info['plots']['all_debug']}`",
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
    parser.add_argument("--transaction-cost-bps", type=float, default=None)
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--return-col", default=None)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--feature-path", default=None, help="Override the feature path from --feature-set.")
    parser.add_argument("--feature-columns", nargs="+", default=None, help="Override feature columns from --feature-set.")
    parser.add_argument("--no-feature-merge", action="store_true")
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
        args.feature_set = args.feature_set or strategy_cfg.get("feature_set")
        args.transaction_cost_bps = args.transaction_cost_bps if args.transaction_cost_bps is not None else strategy_cfg.get("transaction_cost_bps")
        args.score_col = args.score_col or strategy_cfg.get("score_col")
        args.return_col = args.return_col or strategy_cfg.get("return_col")
        benchmarks = strategy_cfg.get("benchmarks", {}) if isinstance(strategy_cfg.get("benchmarks", {}), dict) else {}
        index_weight = benchmarks.get("index_weight", {}) if isinstance(benchmarks.get("index_weight", {}), dict) else {}
        if args.index_weight_path == "data/raw/index_weight.zip":
            args.index_weight_path = str(index_weight.get("weight_path", args.index_weight_path))
        if args.index_code == "000300.SH":
            args.index_code = str(index_weight.get("index_code", args.index_code))

    args.models = args.models or ["final", "lgb_top40"]
    args.splits = args.splits or ["valid", "test"]
    args.transaction_cost_bps = float(5.0 if args.transaction_cost_bps is None else args.transaction_cost_bps)
    args.score_col = args.score_col or "pred"
    args.return_col = args.return_col or "label_1d"
    args.feature_set = args.feature_set or "risk_default"

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
    summary: dict[str, Any] = {
        "out_root": str(out_root),
        "out_parent": args.out_root,
        "run_name": args.run_name,
        "timestamped": not args.no_timestamp,
        "model_registry": args.model_registry,
        "transaction_cost_bps": float(args.transaction_cost_bps),
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
    benchmark_note = "; ".join(benchmark_notes) if benchmark_notes else "none"

    for model_name in args.models:
        summary["models"][model_name] = {}
        for split in args.splits:
            pred_path = resolve_prediction_path(registry, model_name, split)
            print(json.dumps({"stage": "load", "model": model_name, "split": split, "path": pred_path}, ensure_ascii=False), flush=True)
            df = load_prediction_data(pred_path, score_col=args.score_col, return_col=args.return_col)
            if not args.no_feature_merge:
                df = merge_feature_columns(df, feature_path, feature_columns)
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
            for exp_name, cfg in grid:
                cfg = cfg.__class__(**{**cfg.__dict__, "score_col": args.score_col, "return_col": args.return_col})
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
                "best_by_valid_protocol": _select_best(split_rows) if split == "valid" else None,
            }

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_report(out_root, summary, all_rows, benchmark_note)
    print(json.dumps({"saved_summary": str(out_root / "summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
