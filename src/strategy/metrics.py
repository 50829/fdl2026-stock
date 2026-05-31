from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return math.nan
    peak = np.maximum.accumulate(equity)
    return float((equity / (peak + 1e-12) - 1.0).min())


def sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    if len(returns) < 2:
        return math.nan
    sd = float(np.std(returns, ddof=1))
    if sd <= 0:
        return math.nan
    return float(np.mean(returns) / sd * math.sqrt(periods_per_year))


def metrics_from_curve(
    curve: pd.DataFrame,
    name: str,
    strategy: str,
    trading_days_per_year: int = 252,
    transaction_cost_bps: float = 0.0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    returns_np = curve["net_return"].to_numpy(dtype=np.float64) if "net_return" in curve else np.empty(0)
    equity_np = curve["equity"].to_numpy(dtype=np.float64) if "equity" in curve else np.empty(0)
    periods = int(len(curve))
    years = max(1e-12, periods / trading_days_per_year)
    final_equity = float(equity_np[-1]) if len(equity_np) else 1.0
    return {
        "name": name,
        "strategy": strategy,
        "periods": periods,
        "start_date": str(curve["trade_date"].iloc[0]) if periods else None,
        "end_date": str(curve["trade_date"].iloc[-1]) if periods else None,
        "final_equity": final_equity,
        "total_return": final_equity - 1.0,
        "annual_return": float(final_equity ** (1.0 / years) - 1.0),
        "sharpe": sharpe(returns_np, trading_days_per_year),
        "max_drawdown": max_drawdown(equity_np),
        "avg_turnover": float(curve["turnover"].mean()) if "turnover" in curve and periods else 0.0,
        "avg_n_holdings": float(curve["n_holdings"].mean()) if "n_holdings" in curve and periods else math.nan,
        "transaction_cost_bps": float(transaction_cost_bps),
        "config": config or {},
    }
