from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import StrategyBacktestConfig
from ..utils import buyable_day, drop_missing, top_codes


def rank_buffer(
    holdings: dict[str, int],
    day: pd.DataFrame,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    current = drop_missing(holdings, day)
    next_holdings: dict[str, int] = {}
    sells: list[tuple[str, str]] = []
    for code, age in current.items():
        rank = int(day.at[code, "rank"]) if code in day.index else len(day) + 1
        should_sell = False
        reason = ""
        if age >= cfg.max_hold_days and rank > cfg.buy_rank:
            should_sell = True
            reason = "max_hold_rank_check"
        elif age >= cfg.min_hold_days and rank > cfg.sell_rank:
            should_sell = True
            reason = "rank_buffer_exit"
        if should_sell:
            sells.append((code, reason))
        else:
            next_holdings[code] = age

    buy_day = buyable_day(day, cfg)
    buy_pool = buy_day[buy_day["rank"] <= cfg.buy_rank]
    buys = [str(c) for c in buy_pool.index if str(c) not in next_holdings][: max(0, cfg.target_positions - len(next_holdings))]
    if len(next_holdings) + len(buys) < cfg.target_positions:
        extra = top_codes(buy_day, cfg.target_positions - len(next_holdings) - len(buys), exclude=set(next_holdings) | set(buys))
        buys.extend(extra)
    for code in buys:
        next_holdings[code] = 0
    trades = [{"action": "sell", "ts_code": c, "reason": r} for c, r in sells]
    trades.extend({"action": "buy", "ts_code": c, "reason": "rank_buffer_fill"} for c in buys)
    return next_holdings, trades
