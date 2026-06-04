from __future__ import annotations

import math

import pandas as pd

from src.evaluation import (
    BacktestConfig,
    backtest_config_from_cfg,
    backtest_rolling_tranche,
    backtest_topk,
    evaluate_prediction_scores,
)


def make_pred_frame() -> pd.DataFrame:
    rows = []
    for d in ["20250101", "20250102", "20250103", "20250106"]:
        for i in range(5):
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": f"{i:06d}.SZ",
                    "pred": float(5 - i),
                    "label_1d": 0.001 * (i + 1),
                    "label_5d": 0.01 * (i + 1),
                }
            )
    return pd.DataFrame(rows)


def test_backtest_config_from_cfg_reads_legacy_backtest_keys() -> None:
    cfg = {
        "backtest": {
            "mode": "rolling_tranche",
            "n_hold": 30,
            "k_rotate": 3,
            "step_days": 2,
            "tranche_size": 6,
            "hold_days": 4,
            "daily_return_col": "label_1d",
            "transaction_cost_bps": 7.5,
        }
    }

    out = backtest_config_from_cfg(cfg)

    assert out.mode == "rolling_tranche"
    assert out.n_hold == 30
    assert out.k_rotate == 3
    assert out.step_days == 2
    assert out.tranche_size == 6
    assert out.hold_days == 4
    assert out.transaction_cost_bps == 7.5


def test_topk_backtest_uses_strategy_engine_and_keeps_legacy_metric_names() -> None:
    metrics = backtest_topk(make_pred_frame(), "label_5d", BacktestConfig(n_hold=2, k_rotate=1, step_days=2))

    assert metrics["bt_mode"] == "topk"
    assert metrics["bt_periods"] == 2
    assert metrics["bt_step_days"] == 2
    assert metrics["bt_n_hold"] == 2
    assert metrics["bt_k_rotate"] == 1
    assert math.isfinite(float(metrics["bt_total_return"]))
    assert "bt_avg_turnover" in metrics


def test_rolling_tranche_backtest_uses_strategy_engine_and_keeps_legacy_metric_names() -> None:
    metrics = backtest_rolling_tranche(
        make_pred_frame(),
        BacktestConfig(tranche_size=1, hold_days=3, daily_return_col="label_1d"),
    )

    assert metrics["bt_mode"] == "rolling_tranche"
    assert metrics["bt_periods"] == 4
    assert metrics["bt_tranche_size"] == 1
    assert metrics["bt_hold_days"] == 3
    assert metrics["bt_target_active"] == 3
    assert metrics["bt_daily_return_col"] == "label_1d"
    assert math.isfinite(float(metrics["bt_total_return"]))
    assert "bt_avg_active_positions" in metrics


def test_evaluate_prediction_scores_combines_ic_topk_and_rolling_metrics() -> None:
    metrics = evaluate_prediction_scores(
        make_pred_frame(),
        label_col="label_5d",
        raw_return_col="label_5d",
        daily_return_col="label_1d",
        topk_cfg=BacktestConfig(n_hold=2, k_rotate=1, step_days=2),
        rolling_cfg=BacktestConfig(tranche_size=1, hold_days=3, daily_return_col="label_1d"),
    )

    assert metrics["label_col"] == "label_5d"
    assert metrics["raw_return_col"] == "label_5d"
    assert "ic_mean" in metrics
    assert "bt_total_return" in metrics
    assert "rolling_bt_total_return" in metrics
