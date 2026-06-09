from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import StrategyBacktestConfig
from ..risk import risk_score_from_history
from ..utils import buyable_day, drop_missing, equal_weights


def risk_filtered_rank_buffer(
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
    candidate_pool = list(dict.fromkeys(buy_ranked_codes[: max(cfg.risk_candidate_count, cfg.target_positions)] + list(current)))
    core = buy_ranked_codes[: min(cfg.core_count, len(buy_ranked_codes))]
    risk = risk_score_from_history(ret_panel, date, core, candidate_pool, cfg)
    low_risk = list(risk.sort_values(ascending=True).head(min(cfg.risk_keep_count, len(risk))).index)
    low_risk_set = set(str(c) for c in low_risk)

    next_holdings: dict[str, int] = {}
    sell_candidates: list[tuple[str, str, int]] = []
    for code, age in current.items():
        rank = int(day.at[code, "rank"]) if code in day.index else len(day) + 1
        risk_excluded = code not in low_risk_set
        should_sell = False
        reason = ""
        if age >= cfg.max_hold_days and (rank > cfg.buy_rank or risk_excluded):
            should_sell = True
            reason = "max_hold_rank_or_risk"
        elif age >= cfg.min_hold_days and rank > cfg.sell_rank:
            should_sell = True
            reason = "rank_buffer_exit"
        elif age >= cfg.min_hold_days and risk_excluded and rank > cfg.buy_rank:
            should_sell = True
            reason = "risk_filter_exit"
        if should_sell:
            sell_candidates.append((code, reason, rank))
        else:
            next_holdings[code] = age

    sell_candidates = sorted(sell_candidates, key=lambda item: item[2], reverse=True)
    sell_limit = min(cfg.max_stock_updates, len(sell_candidates)) if current else 0
    sells = sell_candidates[:sell_limit]
    for code, _, _ in sells:
        next_holdings.pop(code, None)
    for code, _, _ in sell_candidates[sell_limit:]:
        next_holdings[code] = current[code]

    strict_buy_pool = [
        code
        for code in buy_ranked_codes
        if code in low_risk_set and code not in next_holdings and int(day.at[code, "rank"]) <= cfg.buy_rank
    ]
    fallback_buy_pool = [code for code in buy_ranked_codes if code in low_risk_set and code not in next_holdings]
    broad_buy_pool = [code for code in buy_ranked_codes if code not in next_holdings]
    buy_pool = strict_buy_pool + [c for c in fallback_buy_pool if c not in set(strict_buy_pool)]
    if not current:
        buy_pool += [c for c in broad_buy_pool if c not in set(buy_pool)]
    buys = buy_pool[: max(0, cfg.target_positions - len(next_holdings))]
    for code in buys:
        next_holdings[code] = 0

    trades = [{"action": "sell", "ts_code": c, "reason": r} for c, r, _ in sells]
    trades.extend({"action": "buy", "ts_code": c, "reason": "risk_filtered_rank_buffer_fill"} for c in buys)
    return next_holdings, equal_weights(sorted(next_holdings)), trades
