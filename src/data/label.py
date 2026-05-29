"""Label construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_return_labels(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    trade_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Build labels from tradable future windows.

    label_h[t] = close[t + h + 1] / close[t + 1] - 1
    The t row can only be used after t close; the earliest buy is t+1.
    Offsets are based on the global trading calendar, not each stock's next
    available row, so suspensions or missing rows do not silently skip days.
    """
    horizons = horizons or [1, 5]
    prices = df[["trade_date", "ts_code", "close"]].copy()
    prices["trade_date"] = prices["trade_date"].astype(str)
    prices["ts_code"] = prices["ts_code"].astype(str)

    calendar = sorted(str(d) for d in (trade_dates or prices["trade_date"].unique().tolist()))
    date_to_idx = {date: idx for idx, date in enumerate(calendar)}
    idx_to_date = dict(enumerate(calendar))

    labels = prices[["trade_date", "ts_code"]].copy()
    date_idx = labels["trade_date"].map(date_to_idx)
    labels["buy_date"] = (date_idx + 1).map(idx_to_date)

    buy_prices = prices.rename(columns={"trade_date": "buy_date", "close": "buy_close"})
    labels = labels.merge(buy_prices[["buy_date", "ts_code", "buy_close"]], on=["buy_date", "ts_code"], how="left")

    for horizon in horizons:
        sell_col = f"sell_date_{horizon}d"
        labels[sell_col] = (date_idx + horizon + 1).map(idx_to_date)
        sell_prices = prices.rename(columns={"trade_date": sell_col, "close": f"sell_close_{horizon}d"})
        labels = labels.merge(
            sell_prices[[sell_col, "ts_code", f"sell_close_{horizon}d"]],
            on=[sell_col, "ts_code"],
            how="left",
        )
        labels[f"label_{horizon}d"] = labels[f"sell_close_{horizon}d"] / labels["buy_close"] - 1

    price_cols = ["buy_close"] + [f"sell_close_{horizon}d" for horizon in horizons]
    labels = labels.drop(columns=price_cols)
    label_cols = [f"label_{horizon}d" for horizon in horizons]
    labels[label_cols] = labels[label_cols].replace([np.inf, -np.inf], np.nan)
    return labels


def add_cross_section_label_rank(labels: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add date-wise rank-normalized labels in [-1, 1]."""
    out = labels.copy()
    for col in columns:
        ranked = out[col].groupby(out["trade_date"]).rank(pct=True)
        out[f"{col}__cs_rank"] = ranked * 2 - 1
    return out


def add_market_excess_labels(labels: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add equal-weight market excess labels within each decision-date pool."""
    out = labels.copy()
    for col in columns:
        market_ret = out[col].groupby(out["trade_date"]).transform("mean")
        out[f"{col}_excess"] = out[col] - market_ret
    return out
