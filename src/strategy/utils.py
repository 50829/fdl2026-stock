from __future__ import annotations

import pandas as pd


def equal_weights(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {c: w for c in codes}


def fixed_slot_weights(codes: list[str], target_positions: int) -> dict[str, float]:
    if not codes or target_positions <= 0:
        return {}
    w = 1.0 / int(target_positions)
    return {c: w for c in codes}


def score_weights(scores: pd.Series, total_weight: float) -> dict[str, float]:
    if scores.empty or total_weight <= 0:
        return {}
    ranks = scores.rank(method="average", pct=True).astype(float).clip(lower=0.0) + 1e-6
    denom = float(ranks.sum())
    if denom <= 0:
        return {str(c): total_weight / len(ranks) for c in ranks.index}
    return {str(c): total_weight * float(v) / denom for c, v in ranks.items()}


def turnover(prev: dict[str, float], new: dict[str, float]) -> float:
    codes = set(prev) | set(new)
    return float(sum(abs(float(new.get(c, 0.0)) - float(prev.get(c, 0.0))) for c in codes))


def top_codes(day: pd.DataFrame, n: int, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    return [str(c) for c in day.index if str(c) not in exclude][: max(0, int(n))]


def drop_missing(holdings: dict[str, int], day: pd.DataFrame) -> dict[str, int]:
    available = set(str(c) for c in day.index)
    return {c: age for c, age in holdings.items() if c in available}
