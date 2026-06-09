from __future__ import annotations

import pandas as pd

from src.pipelines.run_strategy_sensitivity import parse_total_cost_grid, select_best_by_valid_cost, split_total_cost, write_sensitivity_heatmaps


def test_parse_total_cost_grid_accepts_lists_and_commas() -> None:
    assert parse_total_cost_grid(["5", "10,20", 50]) == [5.0, 10.0, 20.0, 50.0]
    assert split_total_cost(20.0, 5.0) == (5.0, 15.0)
    assert split_total_cost(3.0, 5.0) == (3.0, 0.0)


def test_select_best_by_valid_cost_uses_valid_then_attaches_test() -> None:
    metrics = pd.DataFrame(
        [
            {"split": "valid", "total_cost_bps": 5.0, "model": "m1", "variant": "a", "strategy": "s", "sharpe": 1.0, "max_drawdown": -0.2, "total_return": 0.2, "avg_turnover": 0.3},
            {"split": "valid", "total_cost_bps": 5.0, "model": "m2", "variant": "b", "strategy": "s", "sharpe": 2.0, "max_drawdown": -0.3, "total_return": 0.4, "avg_turnover": 0.5},
            {"split": "test", "total_cost_bps": 5.0, "model": "m2", "variant": "b", "strategy": "s", "sharpe": 1.5, "max_drawdown": -0.1, "total_return": 0.3, "avg_turnover": 0.4},
        ]
    )

    selected = select_best_by_valid_cost(metrics)

    assert selected.loc[0, "model"] == "m2"
    assert selected.loc[0, "variant"] == "b"
    assert selected.loc[0, "sharpe_valid"] == 2.0
    assert selected.loc[0, "sharpe_test"] == 1.5


def test_write_sensitivity_heatmaps_creates_svg_files(tmp_path) -> None:
    metrics = pd.DataFrame(
        [
            {"split": "valid", "total_cost_bps": 5.0, "model": "m1", "variant": "a", "sharpe": 1.0, "total_return": 0.2, "max_drawdown": -0.2, "avg_turnover": 0.3},
            {"split": "valid", "total_cost_bps": 10.0, "model": "m1", "variant": "a", "sharpe": 0.8, "total_return": 0.1, "max_drawdown": -0.25, "avg_turnover": 0.3},
        ]
    )

    paths = write_sensitivity_heatmaps(metrics, tmp_path, metric_names=("sharpe",))

    path = tmp_path / "plots" / "valid_sharpe.svg"
    assert paths["valid_sharpe"] == str(path)
    assert path.exists()
    assert "成本敏感性矩阵" in path.read_text(encoding="utf-8")
