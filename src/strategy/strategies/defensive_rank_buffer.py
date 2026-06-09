from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import StrategyBacktestConfig
from ..utils import buyable_mask, drop_missing


SIZE_COL = "log_total_mv__cs_rank"
AMOUNT_COL = "log_amount__cs_rank"
VOL_COL = "volatility_20__cs_rank"


def _eligible(day: pd.DataFrame, cfg: StrategyBacktestConfig) -> pd.Series:
    ok = pd.Series(True, index=day.index)
    if SIZE_COL in day.columns:
        ok &= day[SIZE_COL].fillna(-1.0) >= cfg.min_size_rank
    if AMOUNT_COL in day.columns:
        ok &= day[AMOUNT_COL].fillna(-1.0) >= cfg.min_amount_rank
    if VOL_COL in day.columns:
        ok &= day[VOL_COL].fillna(1.0) <= cfg.max_volatility_rank
    return ok


def _market_stressed(ret_panel: pd.DataFrame, date: str, cfg: StrategyBacktestConfig) -> bool:
    hist = ret_panel.loc[ret_panel.index < date].mean(axis=1).tail(cfg.market_window)
    if len(hist) < cfg.market_window:
        return False
    cumulative = float((1.0 + hist).prod() - 1.0)
    return cumulative <= cfg.market_stress_threshold


def _capped_equal_weights(codes: list[str], max_weight: float, gross_exposure: float) -> dict[str, float]:
    if not codes:
        return {}
    raw_weight = min(float(max_weight), float(gross_exposure) / len(codes))
    weights = pd.Series(raw_weight, index=codes, dtype=float)
    return {str(c): float(w) for c, w in weights.items()}


def defensive_rank_buffer(
    holdings: dict[str, int],
    day: pd.DataFrame,
    ret_panel: pd.DataFrame,
    date: str,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], dict[str, float], list[dict[str, Any]]]:
    current = drop_missing(holdings, day)
    eligible = _eligible(day, cfg)
    stressed = _market_stressed(ret_panel, date, cfg)
    gross_exposure = cfg.stress_gross_exposure if stressed else 1.0

    next_holdings: dict[str, int] = {}
    sells: list[tuple[str, str]] = []
    for code, age in current.items():
        rank = int(day.at[code, "rank"]) if code in day.index else len(day) + 1
        is_eligible = bool(eligible.get(code, False))
        should_sell = False
        reason = ""
        if age >= cfg.min_hold_days and not is_eligible:
            should_sell = True
            reason = "size_liquidity_vol_filter_exit"
        elif age >= cfg.max_hold_days and rank > cfg.buy_rank:
            should_sell = True
            reason = "max_hold_rank_check"
        elif age >= cfg.min_hold_days and rank > cfg.sell_rank:
            should_sell = True
            reason = "rank_buffer_exit"
        if should_sell:
            sells.append((code, reason))
        else:
            next_holdings[code] = age

    if current:
        sells = sells[: cfg.max_stock_updates]
    for code, _ in sells:
        next_holdings.pop(code, None)

    eligible_day = day[eligible & buyable_mask(day, cfg)].sort_values(cfg.score_col, ascending=False, kind="mergesort")
    strict_buy_pool = eligible_day[eligible_day["rank"] <= cfg.buy_rank]
    buy_pool = [str(c) for c in strict_buy_pool.index if str(c) not in next_holdings]
    if not current:
        buy_pool += [str(c) for c in eligible_day.index if str(c) not in next_holdings and str(c) not in set(buy_pool)]
    buys = buy_pool[: max(0, cfg.target_positions - len(next_holdings))]
    for code in buys:
        next_holdings[code] = 0

    weights = _capped_equal_weights(sorted(next_holdings), cfg.max_position_weight, gross_exposure)
    trades = [{"action": "sell", "ts_code": c, "reason": r} for c, r in sells]
    buy_reason = "defensive_fill_stressed" if stressed else "defensive_fill"
    trades.extend({"action": "buy", "ts_code": c, "reason": buy_reason} for c in buys)
    return next_holdings, weights, trades
