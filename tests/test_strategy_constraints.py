from __future__ import annotations

import math

import pandas as pd

from src.strategy import StrategyBacktestConfig, merge_trade_constraint_columns, run_strategy


def test_buy_constraints_filter_new_buys_and_slippage_hits_net_return() -> None:
    df = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240102", "20240103", "20240103"],
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "pred": [0.9, 0.5, 0.9, 0.1],
            "label_1d": [0.10, 0.02, 0.00, 0.00],
            "is_buyable": [False, True, True, True],
        }
    )
    cfg = StrategyBacktestConfig(
        strategy="topk_drop",
        topk=1,
        drop=1,
        transaction_cost_bps=5.0,
        slippage_bps=20.0,
        execution_price_model="close_with_slippage",
        enforce_buy_constraints=True,
    )

    result = run_strategy(df, cfg, name="demo")

    first_day_trades = result["trades"][result["trades"]["trade_date"] == "20240102"]
    assert first_day_trades["ts_code"].tolist() == ["000002.SZ"]
    first = result["curve"].iloc[0]
    assert math.isclose(float(first["gross_return"]), 0.02, rel_tol=1e-9)
    assert math.isclose(float(first["fee_cost"]), 0.0005, rel_tol=1e-9)
    assert math.isclose(float(first["slippage_cost"]), 0.0020, rel_tol=1e-9)
    assert math.isclose(float(first["transaction_cost"]), 0.0025, rel_tol=1e-9)
    assert math.isclose(float(first["net_return"]), 0.0175, rel_tol=1e-9)
    assert result["metrics"]["slippage_bps"] == 20.0
    assert result["metrics"]["execution_price_model"] == "close_with_slippage"


def test_merge_trade_constraint_columns_builds_buyable_flag(tmp_path) -> None:
    pred = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240102", "20240102"],
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "pred": [0.3, 0.2, 0.1],
            "label_1d": [0.01, 0.02, 0.03],
        }
    )
    universe = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240102"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "in_universe": [True, True],
            "is_st": [False, True],
            "passes_liquidity": [True, True],
            "amount_mean_20": [100000000.0, 100000000.0],
        }
    )
    path = tmp_path / "universe.parquet"
    universe.to_parquet(path, index=False)

    out, stats = merge_trade_constraint_columns(pred, path)

    flags = dict(zip(out["ts_code"], out["is_buyable"]))
    assert flags == {"000001.SZ": True, "000002.SZ": False, "000003.SZ": False}
    assert stats["matched_rows"] == 2
    assert stats["buyable_rows"] == 1
    assert "not_st" in stats["rules"]


def test_drawdown_control_scales_exposure_after_prior_loss() -> None:
    df = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "pred": [1.0, 1.0],
            "label_1d": [-0.10, 0.10],
        }
    )
    cfg = StrategyBacktestConfig(
        strategy="topk_drop",
        topk=1,
        drop=0,
        transaction_cost_bps=0.0,
        apply_drawdown_control=True,
        drawdown_warning_threshold=-0.08,
        drawdown_warning_exposure=0.50,
        drawdown_cut_threshold=-0.50,
        drawdown_stop_threshold=-0.90,
    )

    curve = run_strategy(df, cfg, name="drawdown_control")["curve"]

    assert math.isclose(float(curve.iloc[0]["gross_exposure"]), 1.0, rel_tol=1e-9)
    assert math.isclose(float(curve.iloc[0]["equity"]), 0.90, rel_tol=1e-9)
    assert math.isclose(float(curve.iloc[1]["portfolio_drawdown_pre"]), -0.10, rel_tol=1e-9)
    assert math.isclose(float(curve.iloc[1]["gross_exposure"]), 0.50, rel_tol=1e-9)
    assert math.isclose(float(curve.iloc[1]["gross_return"]), 0.05, rel_tol=1e-9)


def test_market_stress_deleveraging_uses_prior_market_return() -> None:
    df = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "pred": [1.0, 1.0],
            "label_1d": [-0.05, 0.10],
        }
    )
    cfg = StrategyBacktestConfig(
        strategy="topk_drop",
        topk=1,
        drop=0,
        transaction_cost_bps=0.0,
        apply_market_stress_deleveraging=True,
        market_window=1,
        market_stress_lag=0,
        market_stress_threshold=-0.02,
        stress_gross_exposure=0.40,
    )

    curve = run_strategy(df, cfg, name="market_control")["curve"]

    assert bool(curve.iloc[1]["market_stressed"])
    assert math.isclose(float(curve.iloc[1]["market_stress_return"]), -0.05, rel_tol=1e-9)
    assert math.isclose(float(curve.iloc[1]["gross_exposure"]), 0.40, rel_tol=1e-9)
    assert math.isclose(float(curve.iloc[1]["gross_return"]), 0.04, rel_tol=1e-9)
