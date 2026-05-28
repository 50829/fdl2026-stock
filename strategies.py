from __future__ import annotations

from typing import Callable

import pandas as pd

from backtest import BacktestResult, backtest_topk_rotate


def build_strategy(cfg: dict) -> Callable[[pd.DataFrame, int], BacktestResult]:
    bt_cfg = cfg.get("backtest", {})
    s_cfg = bt_cfg.get("strategy", {})
    name = str(s_cfg.get("name", "topk_rotate")).strip()
    params = s_cfg.get("params", {}) or {}

    if name == "topk_rotate":
        n_hold = int(params.get("n_hold", 20))
        k_rotate = int(params.get("k_rotate", 5))
        transaction_cost_bps = float(params.get("transaction_cost_bps", 0.0))
        trading_days_per_year = int(params.get("trading_days_per_year", 252))

        def _run(pred_df: pd.DataFrame, horizon: int) -> BacktestResult:
            return backtest_topk_rotate(
                pred_df,
                n_hold=n_hold,
                k_rotate=k_rotate,
                horizon=int(horizon),
                transaction_cost_bps=transaction_cost_bps,
                trading_days_per_year=trading_days_per_year,
            )

        return _run

    raise ValueError(f"Unknown backtest strategy: {name}")

