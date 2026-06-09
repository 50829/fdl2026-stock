from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyBacktestConfig:
    strategy: str
    score_col: str = "pred"
    return_col: str = "label_1d"
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 0.0
    execution_price_model: str = "close_to_close"
    enforce_buy_constraints: bool = False
    buyable_col: str = "is_buyable"
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
    min_size_rank: float = -0.35
    min_amount_rank: float = -0.35
    max_volatility_rank: float = 0.70
    market_window: int = 5
    market_stress_threshold: float = -0.08
    market_stress_lag: int = 2
    stress_gross_exposure: float = 0.55
    apply_market_stress_deleveraging: bool = False
    apply_drawdown_control: bool = False
    drawdown_warning_threshold: float = -0.08
    drawdown_warning_exposure: float = 0.50
    drawdown_cut_threshold: float = -0.12
    drawdown_cut_exposure: float = 0.25
    drawdown_stop_threshold: float = -0.18
    drawdown_stop_exposure: float = 0.20
