from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyBacktestConfig:
    strategy: str
    score_col: str = "pred"
    return_col: str = "label_1d"
    transaction_cost_bps: float = 5.0
    trading_days_per_year: int = 252
    target_positions: int = 20
    hold_days: int = 5
    daily_buy: int = 0
    topk: int = 20
    drop: int = 2
    buy_rank: int = 30
    sell_rank: int = 100
    min_hold_days: int = 2
    max_hold_days: int = 10
    core_count: int = 30
    tail_count: int = 70
    core_weight: float = 0.9
    tail_risk_candidates: int = 140
    risk_window: int = 60
    max_stock_updates: int = 25
    risk_candidate_count: int = 150
    risk_keep_count: int = 80
    volatility_penalty: float = 0.35
    max_position_weight: float = 0.08
