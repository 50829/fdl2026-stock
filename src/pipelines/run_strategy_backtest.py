from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
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


DEFAULT_PREDS = {
    "final": {
        "valid": "outputs/models/sdd_final_model_handoff/valid/valid_pred.parquet",
        "test": "outputs/models/sdd_final_model_handoff/test/test_pred.parquet",
    },
    "lgb_top40": {
        "valid": "outputs/models/sdd_feature_selection/lightgbm_top40/lightgbm/valid/valid_pred.parquet",
        "test": "outputs/models/sdd_feature_selection/lightgbm_top40/lightgbm/test/test_pred.parquet",
    },
}


FEATURE_COLUMNS = [
    "log_total_mv__cs_rank",
    "log_amount__cs_rank",
    "volatility_20__cs_rank",
    "turnover_rate__cs_rank",
]


def _timestamped_out_root(base: str, run_name: str, no_timestamp: bool) -> Path:
    base_path = Path(base)
    if no_timestamp:
        return base_path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path / f"{run_name}_{stamp}"


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
    parser.add_argument("--no-timestamp", action="store_true", help="Write directly to --out-root without creating a timestamped run folder.")
    parser.add_argument("--models", nargs="+", choices=sorted(DEFAULT_PREDS), default=["final", "lgb_top40"])
    parser.add_argument("--splits", nargs="+", choices=["valid", "test"], default=["valid", "test"])
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--score-col", default="pred")
    parser.add_argument("--return-col", default="label_1d")
    parser.add_argument("--feature-path", default="data/processed/features.parquet")
    parser.add_argument("--no-feature-merge", action="store_true")
    parser.add_argument("--benchmark-path", default=None, help="Optional CSV/parquet index benchmark with trade_date and close/equity/return.")
    parser.add_argument("--benchmark-name", default="benchmark_index")
    parser.add_argument("--index-weight-path", default="data/raw/index_weight.zip")
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--no-index-weight-benchmark", action="store_true")
    parser.add_argument("--no-equal-weight-benchmark", action="store_true")
    parser.add_argument("--linear-scale", action="store_true", help="Use linear equity scale for comparison SVGs.")
    args = parser.parse_args()

    out_root = _timestamped_out_root(args.out_root, args.run_name, args.no_timestamp)
    grid = build_strategy_grid(cost_bps=args.transaction_cost_bps)
    summary: dict[str, Any] = {
        "out_root": str(out_root),
        "out_parent": args.out_root,
        "run_name": args.run_name,
        "timestamped": not args.no_timestamp,
        "transaction_cost_bps": float(args.transaction_cost_bps),
        "score_col": args.score_col,
        "return_col": args.return_col,
        "feature_path": None if args.no_feature_merge else args.feature_path,
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
            pred_path = DEFAULT_PREDS[model_name][split]
            print(json.dumps({"stage": "load", "model": model_name, "split": split, "path": pred_path}, ensure_ascii=False), flush=True)
            df = load_prediction_data(pred_path, score_col=args.score_col, return_col=args.return_col)
            if not args.no_feature_merge:
                df = merge_feature_columns(df, args.feature_path, FEATURE_COLUMNS)
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
