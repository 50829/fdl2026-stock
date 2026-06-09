from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import StrategyBacktestConfig
from ..risk import risk_score_from_history
from ..utils import buyable_day, drop_missing, score_weights


def risk_balanced_tail(
    holdings: dict[str, int],
    day: pd.DataFrame,
    ret_panel: pd.DataFrame,
    date: str,
    cfg: StrategyBacktestConfig,
) -> tuple[dict[str, int], dict[str, float], list[dict[str, Any]]]:
    score = day[cfg.score_col].astype(float)
    buy_score = buyable_day(day, cfg)[cfg.score_col].astype(float)
    core = [str(c) for c in buy_score.sort_values(ascending=False).head(cfg.core_count).index]
    candidates = [str(c) for c in buy_score.sort_values(ascending=False).index if str(c) not in set(core)]
    candidate_pool = candidates[: max(cfg.tail_risk_candidates, cfg.tail_count)]
    risk = risk_score_from_history(ret_panel, date, core, candidate_pool, cfg)
    low_risk_pool = list(risk.sort_values(ascending=True).head(min(cfg.tail_risk_candidates, len(risk))).index)
    tail = [str(c) for c in buy_score.reindex(low_risk_pool).sort_values(ascending=False).head(cfg.tail_count).index]
    target_codes = core + tail
    current = drop_missing(holdings, day)
    current_codes = set(current)
    target_set = set(target_codes)
    additions = [str(c) for c in buy_score.reindex(list(target_set - current_codes)).sort_values(ascending=False).index]
    removals = [str(c) for c in score.reindex(list(current_codes - target_set)).sort_values(ascending=True).index]
    if current:
        sell_count = min(cfg.max_stock_updates, len(removals))
        next_codes = current_codes - set(removals[:sell_count])
        buy_count = min(len(additions), max(0, len(target_codes) - len(next_codes)))
        next_codes = next_codes | set(additions[:buy_count])
    else:
        sell_count = 0
        buy_count = len(additions)
        next_codes = set(target_codes)

    core_kept = [c for c in core if c in next_codes]
    tail_kept = [c for c in sorted(next_codes - set(core_kept))]
    weights: dict[str, float] = {}
    if core_kept and tail_kept:
        core_total = cfg.core_weight
        tail_total = 1.0 - cfg.core_weight
    elif core_kept:
        core_total = 1.0
        tail_total = 0.0
    else:
        core_total = 0.0
        tail_total = 1.0
    weights.update(score_weights(score.reindex(core_kept).dropna(), core_total))
    tail_score = score.reindex(tail_kept).rank(method="average", pct=True).fillna(0.5)
    tail_risk = risk.reindex(tail_kept).rank(method="average", pct=True).fillna(0.5)
    weights.update(score_weights(0.7 * tail_score + 0.3 * (1.0 - tail_risk), tail_total))
    next_holdings = {c: current.get(c, 0) for c in weights}
    trades = [{"action": "sell", "ts_code": c, "reason": "risk_tail_rebalance"} for c in removals[:sell_count]]
    trades.extend({"action": "buy", "ts_code": c, "reason": "risk_tail_rebalance"} for c in additions[:buy_count])
    return next_holdings, weights, trades
