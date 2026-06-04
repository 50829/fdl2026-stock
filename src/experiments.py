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
    "gru": Command("src.models.sdd.run_e0_e1", "Run MLP/GRU baseline experiments."),
    "gru-ablation": Command("src.models.sdd.run_ablation", "Run GRU/TCN ablation experiments."),
    "gbdt": Command("src.models.sdd.run_gbdt", "Train and evaluate LightGBM/XGBoost baselines."),
    "gbdt-walkforward": Command("src.models.sdd.run_gbdt_walkforward", "Run GBDT walk-forward validation."),
    "fusion": Command("src.models.sdd.run_fusion_methods", "Run stacking/residual-rank/tree-neural fusion experiments."),
    "residual-mlp": Command("src.models.sdd.run_residual_mlp", "Run LightGBM residual MLP experiments."),
    "prediction-ensemble": Command("src.models.sdd.run_prediction_ensemble", "Run simple prediction/rank ensemble grids."),
    "rolling-eval": Command("src.models.sdd.run_rolling_tranche_eval", "Evaluate a prediction file with rolling tranche metrics."),
    "backtest-sensitivity": Command("src.models.sdd.run_backtest_sensitivity", "Run backtest parameter sensitivity analysis."),
    "final-handoff": Command("src.pipelines.make_final_handoff", "Reproduce final residual-rank model handoff predictions."),
    "strategy-backtest": Command("src.pipelines.run_strategy_backtest", "Run strategy grid backtests from registered prediction files."),
    "live-rank": Command("src.pipelines.live_rank", "Generate a live ranking file for one decision date."),
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
