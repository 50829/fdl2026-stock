from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    module: str
    help: str


COMMANDS: dict[str, Command] = {
    "preprocess": Command("src.data.preprocess", "Build processed parquet data from raw daily files."),
    "feature-meta": Command("src.data.feature_meta", "Build or inspect feature metadata."),
    "train": Command("src.train", "Train a torch model from a YAML config."),
    "predict": Command("src.predict", "Generate torch model predictions from a YAML config."),
    "gru": Command("src.model_experiments.run_e0_e1", "Run MLP/GRU baseline experiments."),
    "gru-ablation": Command("src.model_experiments.run_ablation", "Run GRU/TCN ablation experiments."),
    "gbdt": Command("src.model_experiments.run_gbdt", "Train and evaluate LightGBM/XGBoost baselines."),
    "gbdt-walkforward": Command("src.model_experiments.run_gbdt_walkforward", "Run GBDT walk-forward validation."),
    "fusion": Command("src.model_experiments.run_fusion_methods", "Run stacking/residual-rank/tree-neural fusion experiments."),
    "residual-mlp": Command("src.model_experiments.run_residual_mlp", "Run LightGBM residual MLP experiments."),
    "prediction-ensemble": Command("src.model_experiments.run_prediction_ensemble", "Run simple prediction/rank ensemble grids."),
    "report-fusion": Command("src.model_experiments.run_report_fusion", "Generate report-grade label1d prediction fusion files."),
    "nsntk-inspired": Command("src.model_experiments.run_nsntk_inspired_experiments", "Run time decay, EMA, seed ensemble, score smoothing, and IC stability experiments."),
    "seq-len-fusion": Command("src.model_experiments.run_seq_len_ablation_fusion", "Run GRU/TCN sequence length ablations and tree fusion backtests."),
    "gcn-propagation": Command("src.model_experiments.run_gcn_graph_propagation", "Run industry-graph propagation baselines for stock scores."),
    "gcn-rolling-corr": Command("src.model_experiments.run_gcn_rolling_corr", "Run rolling return-correlation graph propagation baselines for stock scores."),
    "tree-residual-deep": Command("src.model_experiments.run_tree_residual_deep", "Train a deep residual model behind tree ensemble scores."),
    "oof-tree-residual-deep": Command("src.model_experiments.run_oof_tree_residual_deep", "Train a strict OOF tree residual deep model."),
    "rolling-eval": Command("src.model_experiments.run_rolling_tranche_eval", "Evaluate a prediction file with rolling tranche metrics."),
    "backtest-sensitivity": Command("src.model_experiments.run_backtest_sensitivity", "Run backtest parameter sensitivity analysis."),
    "final-handoff": Command("src.pipelines.make_final_handoff", "Reproduce final residual-rank model handoff predictions."),
    "final-report-figures": Command("src.pipelines.plot_final_report_figures", "Plot report figures from completed model and strategy outputs."),
    "final-strategy-curves": Command("src.pipelines.plot_final_strategy_curves", "Plot final strategy time-return curves for the report."),
    "final-report-tables": Command("src.pipelines.make_final_report_tables", "Build final report tables from completed experiment outputs."),
    "final-strategy-report-tables": Command("src.pipelines.make_final_strategy_report_tables", "Build final strategy report tables from completed strategy outputs."),
    "strategy-backtest": Command("src.pipelines.run_strategy_backtest", "Run strategy grid backtests from registered prediction files."),
    "strategy-report": Command("src.pipelines.run_strategy_report", "Refresh strategy plots, long tables, and HTML reports from a completed run."),
    "strategy-sensitivity": Command("src.pipelines.run_strategy_sensitivity", "Run cost sensitivity matrices for registered strategy runs."),
    "strategy-risk-sweep": Command("src.pipelines.run_strategy_risk_sweep", "Run risk-control return/drawdown sweeps for registered strategy runs."),
    "live-rank": Command("src.pipelines.live_rank", "Generate a live ranking file for one decision date."),
    "label1d-window-ablation": Command("src.pipelines.run_label1d_window_ablation", "Run label1d LightGBM feature-window ablations."),
    "label1d-window-walkforward": Command("src.pipelines.run_label1d_window_walkforward", "Run label1d expanding walk-forward checks for window ablations."),
    "plot-label1d-window-walkforward": Command("src.pipelines.plot_label1d_window_walkforward", "Plot label1d expanding walk-forward equity curves."),
    "normalize-outputs": Command("src.pipelines.normalize_outputs", "Normalize local output directory names."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments",
        description="Canonical experiment entry point. Pass command-specific options after the command.",
    )
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def print_command_list() -> None:
    width = max(len(name) for name in COMMANDS)
    print("Available commands:")
    for name in sorted(COMMANDS):
        print(f"  {name:<{width}}  {COMMANDS[name].help}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        build_parser().print_help()
        print()
        print_command_list()
        return

    command = argv[0]
    if command not in COMMANDS:
        build_parser().error(f"unknown command: {command}")

    module = COMMANDS[command].module
    old_argv = sys.argv[:]
    try:
        sys.argv = [f"python -m src.experiments {command}", *argv[1:]]
        mod = importlib.import_module(module)
        mod.run_cli()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
