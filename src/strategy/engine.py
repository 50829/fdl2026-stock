from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .config import StrategyBacktestConfig
from .data import prepare_maps
from .metrics import metrics_from_curve
from .strategies import defensive_rank_buffer, rank_buffer, risk_balanced_tail, risk_budget_rank_buffer, risk_filtered_rank_buffer, rolling_tranche, topk_drop
from .utils import equal_weights, fixed_slot_weights, turnover


def _historical_market_return(ret_panel: pd.DataFrame, date: str, cfg: StrategyBacktestConfig) -> float:
    hist = ret_panel.loc[ret_panel.index < date].mean(axis=1)
    lag = max(0, int(cfg.market_stress_lag))
    if lag:
        hist = hist.iloc[:-lag]
    hist = hist.tail(max(1, int(cfg.market_window)))
    if len(hist) < max(1, int(cfg.market_window)):
        return math.nan
    return float((1.0 + hist).prod() - 1.0)


def _market_exposure_limit(ret_panel: pd.DataFrame, date: str, cfg: StrategyBacktestConfig) -> tuple[float, bool, float]:
    if not cfg.apply_market_stress_deleveraging:
        return 1.0, False, math.nan
    market_return = _historical_market_return(ret_panel, date, cfg)
    stressed = not math.isnan(market_return) and market_return <= cfg.market_stress_threshold
    if stressed:
        return max(0.0, min(1.0, float(cfg.stress_gross_exposure))), True, market_return
    return 1.0, False, market_return


def _drawdown_exposure_limit(equity: float, peak_equity: float, cfg: StrategyBacktestConfig) -> tuple[float, float]:
    if not cfg.apply_drawdown_control:
        return 1.0, 0.0
    if peak_equity <= 0:
        return 1.0, 0.0
    drawdown = float(equity / peak_equity - 1.0)
    levels = [
        (cfg.drawdown_stop_threshold, cfg.drawdown_stop_exposure),
        (cfg.drawdown_cut_threshold, cfg.drawdown_cut_exposure),
        (cfg.drawdown_warning_threshold, cfg.drawdown_warning_exposure),
    ]
    for threshold, exposure in sorted(levels, key=lambda item: item[0]):
        if drawdown <= float(threshold):
            return max(0.0, min(1.0, float(exposure))), drawdown
    return 1.0, drawdown


def _limit_gross_exposure(weights: dict[str, float], limit: float) -> dict[str, float]:
    gross = float(sum(abs(float(w)) for w in weights.values()))
    limit = max(0.0, min(1.0, float(limit)))
    if gross <= 0.0 or gross <= limit:
        return dict(weights)
    scale = limit / gross
    return {code: float(weight) * scale for code, weight in weights.items()}


def target_holdings(
    holdings: dict[str, int],
    day: pd.DataFrame,
    ret_panel: pd.DataFrame,
    date: str,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], dict[str, float], list[dict[str, Any]]]:
    if cfg.strategy == "rolling_tranche":
        next_holdings, trades = rolling_tranche(holdings, day, cfg)
        return next_holdings, fixed_slot_weights(sorted(next_holdings), cfg.target_positions), trades
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
    peak_equity = 1.0
    curve_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []

    for date in dates:
        day = day_map[date]
        holdings, weights, trades = target_holdings(holdings, day, ret_panel, date, cfg)
        raw_gross_exposure = float(sum(abs(float(w)) for w in weights.values()))
        market_limit, market_stressed, market_return = _market_exposure_limit(ret_panel, date, cfg)
        drawdown_limit, portfolio_drawdown_pre = _drawdown_exposure_limit(equity, peak_equity, cfg)
        exposure_limit = min(market_limit, drawdown_limit)
        weights = _limit_gross_exposure(weights, exposure_limit)
        gross_exposure = float(sum(abs(float(w)) for w in weights.values()))
        day_turnover = turnover(prev_weights, weights)
        fee_cost = day_turnover * cfg.transaction_cost_bps / 10000.0
        slippage_cost = day_turnover * cfg.slippage_bps / 10000.0
        cost = fee_cost + slippage_cost
        returns = day[cfg.return_col].reindex(weights.keys()).fillna(0.0)
        gross_ret = float(sum(weights[c] * float(returns.get(c, 0.0)) for c in weights))
        net_ret = gross_ret - cost
        equity *= 1.0 + net_ret
        peak_equity = max(peak_equity, equity)
        curve_rows.append(
            {
                "trade_date": date,
                "gross_return": gross_ret,
                "transaction_cost": cost,
                "fee_cost": fee_cost,
                "slippage_cost": slippage_cost,
                "total_cost": cost,
                "net_return": net_ret,
                "turnover": day_turnover,
                "equity": equity,
                "n_holdings": len(weights),
                "raw_gross_exposure": raw_gross_exposure,
                "gross_exposure": gross_exposure,
                "exposure_limit": exposure_limit,
                "market_exposure_limit": market_limit,
                "drawdown_exposure_limit": drawdown_limit,
                "market_stressed": market_stressed,
                "market_stress_return": market_return,
                "portfolio_drawdown_pre": portfolio_drawdown_pre,
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
        slippage_bps=cfg.slippage_bps,
        execution_price_model=cfg.execution_price_model,
        config=cfg.__dict__,
    )
    return {
        "metrics": metrics,
        "curve": curve,
        "trades": pd.DataFrame(trade_rows),
        "holdings": pd.DataFrame(holding_rows),
    }
