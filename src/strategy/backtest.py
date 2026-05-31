from __future__ import annotations

from .benchmarks import (
    align_benchmark_to_dates,
    build_equal_weight_benchmark,
    build_index_weight_benchmark,
    load_index_weight_data,
    load_price_benchmark,
)
from .config import StrategyBacktestConfig
from .data import load_prediction_data, prepare_maps
from .engine import run_strategy, target_holdings
from .grid import build_strategy_grid
from .io import write_strategy_outputs
from .metrics import max_drawdown, metrics_from_curve, sharpe
from .plotting import plot_comparison, write_split_plots

__all__ = [
    "StrategyBacktestConfig",
    "align_benchmark_to_dates",
    "build_equal_weight_benchmark",
    "build_index_weight_benchmark",
    "build_strategy_grid",
    "load_index_weight_data",
    "load_prediction_data",
    "load_price_benchmark",
    "max_drawdown",
    "metrics_from_curve",
    "plot_comparison",
    "prepare_maps",
    "run_strategy",
    "sharpe",
    "target_holdings",
    "write_strategy_outputs",
    "write_split_plots",
]
