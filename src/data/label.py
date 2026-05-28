"""Label construction utilities."""

from __future__ import annotations

import pandas as pd


def add_forward_return_labels(df: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """Build labels from tradable future windows.

    label_h[t] = close[t + h + 1] / close[t + 1] - 1
    The t row can only be used after t close; the earliest buy is t+1.
    """
    horizons = horizons or [1, 5]
    out = df[["trade_date", "ts_code", "close"]].sort_values(["ts_code", "trade_date"]).copy()
    grouped = out.groupby("ts_code")["close"]
    buy_price = grouped.shift(-1)
    labels = out[["trade_date", "ts_code"]].copy()
    for horizon in horizons:
        sell_price = grouped.shift(-(horizon + 1))
        labels[f"label_{horizon}d"] = sell_price / buy_price - 1
    return labels


def add_cross_section_label_rank(labels: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add date-wise rank-normalized labels in [-1, 1]."""
    out = labels.copy()
    for col in columns:
        ranked = out[col].groupby(out["trade_date"]).rank(pct=True)
        out[f"{col}__cs_rank"] = ranked * 2 - 1
    return out
