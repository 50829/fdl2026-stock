from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import StrategyBacktestConfig
from ..utils import buyable_day, drop_missing


def _historical_vol(ret_panel: pd.DataFrame, date: str, codes: list[str], cfg: StrategyBacktestConfig) -> pd.Series:
    codes = list(dict.fromkeys(str(c) for c in codes))
    hist = ret_panel.loc[ret_panel.index < date].tail(cfg.risk_window)
    if len(hist) < 2 or not codes:
        return pd.Series(1.0, index=codes)
    cols = [c for c in codes if c in hist.columns]
    vol = hist[cols].std(ddof=1).replace([np.inf, -np.inf], np.nan)
    fill = float(vol.median()) if vol.notna().any() else 1.0
    return vol.reindex(codes).fillna(fill).clip(lower=1e-6)


def _capped_inverse_vol_weights(vol: pd.Series, max_weight: float) -> dict[str, float]:
    if vol.empty:
        return {}
    raw = (1.0 / vol.astype(float).clip(lower=1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(raw.sum()) <= 0:
        raw = pd.Series(1.0, index=vol.index)
    weights = raw / float(raw.sum())
    cap = max(float(max_weight), 1.0 / len(weights))
    for _ in range(10):
        over = weights > cap
        if not bool(over.any()):
            break
        excess = float((weights[over] - cap).sum())
        weights[over] = cap
        under = ~over
        under_sum = float(weights[under].sum())
        if under_sum <= 0:
            break
        weights[under] += weights[under] / under_sum * excess
    weights = weights / float(weights.sum())
    return {str(c): float(w) for c, w in weights.items()}


def risk_budget_rank_buffer(
    holdings: dict[str, int],
    day: pd.DataFrame,
    ret_panel: pd.DataFrame,
    date: str,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], dict[str, float], list[dict[str, Any]]]:
    current = drop_missing(holdings, day)
    score = day[cfg.score_col].astype(float)
    buy_score = buyable_day(day, cfg)[cfg.score_col].astype(float)
    buy_ranked_codes = [str(c) for c in buy_score.sort_values(ascending=False).index]
    buy_ranked_set = set(buy_ranked_codes)
    candidate_pool = list(dict.fromkeys(buy_ranked_codes[: max(cfg.risk_candidate_count, cfg.target_positions)] + list(current)))
    vol = _historical_vol(ret_panel, date, candidate_pool + list(current), cfg)
    alpha_pct = score.reindex(candidate_pool).rank(method="average", pct=True).fillna(0.0)
    vol_pct = vol.reindex(candidate_pool).rank(method="average", pct=True).fillna(0.5)
    utility = alpha_pct - cfg.volatility_penalty * vol_pct
    utility = utility.sort_values(ascending=False, kind="mergesort")
    utility_rank = {str(c): i + 1 for i, c in enumerate(utility.index)}

    next_holdings: dict[str, int] = {}
    sell_candidates: list[tuple[str, str, int]] = []
    for code, age in current.items():
        rank = int(day.at[code, "rank"]) if code in day.index else len(day) + 1
        util_rank = utility_rank.get(code, len(day) + 1)
        should_sell = False
        reason = ""
        if age >= cfg.max_hold_days and (rank > cfg.buy_rank or util_rank > cfg.risk_keep_count):
            should_sell = True
            reason = "max_hold_rank_or_vol_budget"
        elif age >= cfg.min_hold_days and rank > cfg.sell_rank:
            should_sell = True
            reason = "rank_buffer_exit"
        elif age >= cfg.min_hold_days and util_rank > cfg.risk_keep_count and rank > cfg.buy_rank:
            should_sell = True
            reason = "vol_budget_exit"
        if should_sell:
            sell_candidates.append((code, reason, max(rank, util_rank)))
        else:
            next_holdings[code] = age

    sell_candidates = sorted(sell_candidates, key=lambda item: item[2], reverse=True)
    sell_limit = min(cfg.max_stock_updates, len(sell_candidates)) if current else 0
    sells = sell_candidates[:sell_limit]
    for code, _, _ in sells:
        next_holdings.pop(code, None)
    for code, _, _ in sell_candidates[sell_limit:]:
        next_holdings[code] = current[code]

    buy_pool = [
        str(c)
        for c in utility.index
        if str(c) in buy_ranked_set and str(c) not in next_holdings and int(day.at[str(c), "rank"]) <= cfg.buy_rank
    ]
    if not current:
        buy_pool += [str(c) for c in utility.index if str(c) in buy_ranked_set and str(c) not in next_holdings and str(c) not in set(buy_pool)]
    buys = buy_pool[: max(0, cfg.target_positions - len(next_holdings))]
    for code in buys:
        next_holdings[code] = 0

    held_codes = sorted(next_holdings)
    weights = _capped_inverse_vol_weights(vol.reindex(held_codes).fillna(float(vol.median())), cfg.max_position_weight)
    trades = [{"action": "sell", "ts_code": c, "reason": r} for c, r, _ in sells]
    trades.extend({"action": "buy", "ts_code": c, "reason": "risk_budget_rank_buffer_fill"} for c in buys)
    return next_holdings, weights, trades
