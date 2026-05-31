from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .config import StrategyBacktestConfig
from .data import prepare_maps
from .metrics import metrics_from_curve
from .strategies import defensive_rank_buffer, rank_buffer, risk_balanced_tail, risk_budget_rank_buffer, risk_filtered_rank_buffer, rolling_tranche, topk_drop
from .utils import equal_weights, turnover


def target_holdings(
    holdings: dict[str, int],
    day: pd.DataFrame,
    ret_panel: pd.DataFrame,
    date: str,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], dict[str, float], list[dict[str, Any]]]:
    if cfg.strategy == "rolling_tranche":
        next_holdings, trades = rolling_tranche(holdings, day, cfg)
        return next_holdings, equal_weights(sorted(next_holdings)), trades
    if cfg.strategy == "topk_drop":
        next_holdings, trades = topk_drop(holdings, day, cfg)
        return next_holdings, equal_weights(sorted(next_holdings)), trades
    if cfg.strategy == "rank_buffer":
        next_holdings, trades = rank_buffer(holdings, day, cfg)
        return next_holdings, equal_weights(sorted(next_holdings)), trades
    if cfg.strategy == "defensive_rank_buffer":
        return defensive_rank_buffer(holdings, day, ret_panel, date, cfg)
    if cfg.strategy == "risk_balanced_tail":
        return risk_balanced_tail(holdings, day, ret_panel, date, cfg)
    if cfg.strategy == "risk_filtered_rank_buffer":
        return risk_filtered_rank_buffer(holdings, day, ret_panel, date, cfg)
    if cfg.strategy == "risk_budget_rank_buffer":
        return risk_budget_rank_buffer(holdings, day, ret_panel, date, cfg)
    raise ValueError(f"unknown strategy: {cfg.strategy}")


def run_strategy(df: pd.DataFrame, cfg: StrategyBacktestConfig, name: str | None = None) -> dict[str, Any]:
    dates, day_map, ret_panel = prepare_maps(df, cfg)
    holdings: dict[str, int] = {}
    prev_weights: dict[str, float] = {}
    equity = 1.0
    curve_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []

    for date in dates:
        day = day_map[date]
        holdings, weights, trades = target_holdings(holdings, day, ret_panel, date, cfg)
        day_turnover = turnover(prev_weights, weights)
        cost = day_turnover * cfg.transaction_cost_bps / 10000.0
        returns = day[cfg.return_col].reindex(weights.keys()).fillna(0.0)
        gross_ret = float(sum(weights[c] * float(returns.get(c, 0.0)) for c in weights))
        net_ret = gross_ret - cost
        equity *= 1.0 + net_ret
        curve_rows.append(
            {
                "trade_date": date,
                "gross_return": gross_ret,
                "transaction_cost": cost,
                "net_return": net_ret,
                "turnover": day_turnover,
                "equity": equity,
                "n_holdings": len(weights),
            }
        )
        for tr in trades:
            trade_rows.append({"trade_date": date, "strategy": name or cfg.strategy, **tr})
        for code, weight in sorted(weights.items()):
            holding_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "weight": weight,
                    "holding_days": holdings.get(code, 0),
                    "score": float(day.at[code, cfg.score_col]) if code in day.index else math.nan,
                    "rank": int(day.at[code, "rank"]) if code in day.index else -1,
                }
            )
        holdings = {c: age + 1 for c, age in holdings.items()}
        prev_weights = weights

    curve = pd.DataFrame(curve_rows)
    metrics = metrics_from_curve(
        curve,
        name=name or cfg.strategy,
        strategy=cfg.strategy,
        trading_days_per_year=cfg.trading_days_per_year,
        transaction_cost_bps=cfg.transaction_cost_bps,
        config=cfg.__dict__,
    )
    return {
        "metrics": metrics,
        "curve": curve,
        "trades": pd.DataFrame(trade_rows),
        "holdings": pd.DataFrame(holding_rows),
    }
