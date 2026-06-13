from __future__ import annotations

import importlib

from src.experiments import COMMANDS


def test_new_canonical_commands_are_registered() -> None:
    for name in [
        "train",
        "predict",
        "strategy-backtest",
        "strategy-report",
        "strategy-sensitivity",
        "strategy-risk-sweep",
        "live-rank",
        "label1d-window-ablation",
        "label1d-window-walkforward",
        "plot-label1d-window-walkforward",
        "normalize-outputs",
        "report-fusion",
        "nsntk-inspired",
        "seq-len-fusion",
        "final-report-figures",
        "final-strategy-curves",
        "final-report-tables",
        "final-strategy-report-tables",
        "gcn-propagation",
        "gcn-rolling-corr",
        "tree-residual-deep",
        "oof-tree-residual-deep",
    ]:
        assert name in COMMANDS


def test_registered_commands_expose_run_cli() -> None:
    for name in [
        "preprocess",
        "gbdt",
        "train",
        "predict",
        "strategy-backtest",
        "strategy-report",
        "strategy-sensitivity",
        "strategy-risk-sweep",
        "live-rank",
        "label1d-window-ablation",
        "label1d-window-walkforward",
        "plot-label1d-window-walkforward",
        "normalize-outputs",
        "report-fusion",
        "nsntk-inspired",
        "seq-len-fusion",
        "final-report-figures",
        "final-strategy-curves",
        "final-report-tables",
        "final-strategy-report-tables",
        "gcn-propagation",
        "gcn-rolling-corr",
        "tree-residual-deep",
        "oof-tree-residual-deep",
    ]:
        module = importlib.import_module(COMMANDS[name].module)
        assert callable(getattr(module, "run_cli", None)), name
