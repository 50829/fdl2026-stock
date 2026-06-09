from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import StrategyBacktestConfig


def build_strategy_grid(
    cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    execution_price_model: str = "close_to_close",
    enforce_buy_constraints: bool = False,
    config_overrides: dict[str, Any] | None = None,
) -> list[tuple[str, StrategyBacktestConfig]]:
    rows: list[tuple[str, StrategyBacktestConfig]] = []
    base_kwargs = {
        "transaction_cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "execution_price_model": execution_price_model,
        "enforce_buy_constraints": enforce_buy_constraints,
    }
    if config_overrides:
        base_kwargs.update(config_overrides)
    base = StrategyBacktestConfig(strategy="rolling_tranche", **base_kwargs)
    for target, hold in [(10, 5), (20, 3), (20, 5), (20, 10), (30, 5)]:
        cfg = replace(base, target_positions=target, hold_days=hold, daily_buy=max(1, round(target / hold)))
        rows.append((f"rolling_p{target}_h{hold}", cfg))

    base = StrategyBacktestConfig(strategy="topk_drop", **base_kwargs)
    for topk, drop in [(20, 1), (20, 2), (20, 3), (20, 5), (30, 3)]:
        rows.append((f"topk{topk}_drop{drop}", replace(base, topk=topk, drop=drop)))

    base = StrategyBacktestConfig(strategy="rank_buffer", **base_kwargs)
    for target, buy, sell, min_hold, max_hold in [(20, 30, 100, 2, 10), (20, 50, 100, 2, 10), (30, 50, 150, 2, 10)]:
        rows.append(
            (
                f"rankbuf_p{target}_b{buy}_s{sell}_min{min_hold}_max{max_hold}",
                replace(base, target_positions=target, buy_rank=buy, sell_rank=sell, min_hold_days=min_hold, max_hold_days=max_hold),
            )
        )

    base = StrategyBacktestConfig(strategy="defensive_rank_buffer", **base_kwargs)
    for target, buy, sell, min_size, min_amt, max_vol, stress_exp in [
        (20, 60, 180, -0.35, -0.35, 0.70, 0.55),
        (20, 80, 220, -0.20, -0.20, 0.60, 0.50),
        (30, 100, 260, -0.20, -0.20, 0.65, 0.60),
    ]:
        rows.append(
            (
                f"defensive_p{target}_b{buy}_s{sell}_size{int(min_size * 100)}_amt{int(min_amt * 100)}",
                replace(
                    base,
                    target_positions=target,
                    buy_rank=buy,
                    sell_rank=sell,
                    min_hold_days=3,
                    max_hold_days=15,
                    max_stock_updates=4,
                    min_size_rank=min_size,
                    min_amount_rank=min_amt,
                    max_volatility_rank=max_vol,
                    stress_gross_exposure=stress_exp,
                    max_position_weight=0.08 if target == 20 else 0.06,
                ),
            )
        )

    rows.append(
        (
            "risk_tail_core30_tail70",
            StrategyBacktestConfig(strategy="risk_balanced_tail", **base_kwargs),
        )
    )

    base = StrategyBacktestConfig(strategy="risk_filtered_rank_buffer", **base_kwargs)
    for target, candidate, keep, buy, sell, min_hold, max_hold, max_updates in [
        (20, 100, 70, 50, 120, 3, 10, 4),
        (30, 150, 80, 60, 150, 3, 10, 5),
        (30, 100, 70, 50, 120, 5, 15, 4),
        (20, 150, 80, 40, 100, 3, 10, 4),
    ]:
        rows.append(
            (
                f"riskbuf_p{target}_top{candidate}_keep{keep}_b{buy}_s{sell}_min{min_hold}_max{max_hold}",
                replace(
                    base,
                    target_positions=target,
                    risk_candidate_count=candidate,
                    risk_keep_count=keep,
                    buy_rank=buy,
                    sell_rank=sell,
                    min_hold_days=min_hold,
                    max_hold_days=max_hold,
                    max_stock_updates=max_updates,
                ),
            )
        )
    base = StrategyBacktestConfig(strategy="risk_budget_rank_buffer", **base_kwargs)
    for target, candidate, keep, buy, sell, min_hold, max_hold, max_updates, penalty, cap in [
        (20, 150, 80, 40, 120, 3, 12, 4, 0.25, 0.08),
        (20, 200, 100, 50, 150, 5, 15, 3, 0.35, 0.08),
        (30, 200, 120, 60, 180, 5, 15, 4, 0.35, 0.06),
    ]:
        rows.append(
            (
                f"riskbudget_p{target}_top{candidate}_keep{keep}_b{buy}_s{sell}_pen{int(penalty * 100)}",
                replace(
                    base,
                    target_positions=target,
                    risk_candidate_count=candidate,
                    risk_keep_count=keep,
                    buy_rank=buy,
                    sell_rank=sell,
                    min_hold_days=min_hold,
                    max_hold_days=max_hold,
                    max_stock_updates=max_updates,
                    volatility_penalty=penalty,
                    max_position_weight=cap,
                ),
            )
        )
    return rows
