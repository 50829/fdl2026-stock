from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import StrategyBacktestConfig
from ..utils import drop_missing, top_codes


def rolling_tranche(
    holdings: dict[str, int],
    day: pd.DataFrame,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    next_holdings = {c: age for c, age in drop_missing(holdings, day).items() if age < cfg.hold_days}
    sold = sorted(set(holdings) - set(next_holdings))
    daily_buy = int(cfg.daily_buy or max(1, round(cfg.target_positions / max(1, cfg.hold_days))))
    buy_slots = max(0, min(daily_buy, cfg.target_positions - len(next_holdings)))
    buys = top_codes(day, buy_slots, exclude=set(next_holdings))
    for code in buys:
        next_holdings[code] = 0
    trades = [{"action": "sell", "ts_code": c, "reason": "expired_or_missing"} for c in sold]
    trades.extend({"action": "buy", "ts_code": c, "reason": "daily_tranche"} for c in buys)
    return next_holdings, trades
