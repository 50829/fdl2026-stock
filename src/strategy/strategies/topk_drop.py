from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import StrategyBacktestConfig
from ..utils import drop_missing, top_codes


def topk_drop(
    holdings: dict[str, int],
    day: pd.DataFrame,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    current = drop_missing(holdings, day)
    if not current:
        buys = top_codes(day, cfg.topk)
        return {c: 0 for c in buys}, [{"action": "buy", "ts_code": c, "reason": "initial_topk"} for c in buys]
    current_codes = list(current)
    held_rank = day.reindex(current_codes)["rank"].fillna(len(day) + 1).sort_values(ascending=False)
    sells = [str(c) for c in held_rank.head(min(cfg.drop, len(held_rank))).index]
    after_sell = {c: age for c, age in current.items() if c not in set(sells)}
    need = max(0, cfg.topk - len(after_sell))
    buys = top_codes(day, need, exclude=set(after_sell))
    next_holdings = dict(after_sell)
    for code in buys:
        next_holdings[code] = 0
    trades = [{"action": "sell", "ts_code": c, "reason": "drop_worst_rank"} for c in sells]
    trades.extend({"action": "buy", "ts_code": c, "reason": "topk_refill"} for c in buys)
    return next_holdings, trades
