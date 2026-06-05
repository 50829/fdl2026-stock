from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.strategy.plotting import write_split_plots
from src.strategy.reporting import load_existing_aggregate_outputs, write_report_artifacts


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Existing strategy backtest run directory.")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to refresh, for example valid test.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--benchmark-note", default="")
    parser.add_argument("--linear-scale", action="store_true")
    parser.add_argument("--no-refresh-plots", action="store_true", help="Only rewrite long tables and HTML.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows_by_split, curves_by_split = load_existing_aggregate_outputs(run_dir, splits=args.splits)
    if not rows_by_split:
        parser.error(f"no aggregate strategy_metrics.csv files found under `{run_dir}`")

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if not args.no_refresh_plots:
        summary.setdefault("aggregate", {})
        valid_rows = rows_by_split.get("valid")
        for split, rows in rows_by_split.items():
            plots = write_split_plots(
                curves_by_split.get(split, {}),
                rows,
                run_dir / split,
                f"{split} label1d vs label5d strategy equity",
                log_scale=not args.linear_scale,
                valid_rows=valid_rows,
            )
            summary["aggregate"].setdefault(split, {})["plots"] = plots

    report_paths = write_report_artifacts(
        run_dir,
        rows_by_split,
        curves_by_split,
        benchmark_note=args.benchmark_note,
        title=args.title or run_dir.name,
    )
    summary["reporting"] = report_paths
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"reporting": report_paths}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
