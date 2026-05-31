from __future__ import annotations

import pandas as pd


def topk_by_date(pred_df: pd.DataFrame, k: int, score_col: str = "pred") -> pd.DataFrame:
    required = {"trade_date", "ts_code", score_col}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction frame missing required columns: {sorted(missing)}")
    return (
        pred_df.sort_values(["trade_date", score_col], ascending=[True, False], kind="mergesort")
        .groupby("trade_date", sort=False)
        .head(int(k))
        .reset_index(drop=True)
    )


def rolling_daily_buy_list(
    pred_df: pd.DataFrame,
    target_positions: int = 20,
    hold_days: int = 5,
    score_col: str = "pred",
) -> pd.DataFrame:
    daily_buy_n = max(1, int(target_positions) // max(1, int(hold_days)))
    return topk_by_date(pred_df, k=daily_buy_n, score_col=score_col)


__all__ = ["rolling_daily_buy_list", "topk_by_date"]
