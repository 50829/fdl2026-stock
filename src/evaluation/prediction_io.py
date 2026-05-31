from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_PRED_COLUMNS = ["trade_date", "ts_code", "pred"]
FINAL_PRED_COLUMNS = [
    "trade_date",
    "ts_code",
    "pred",
    "final_pred",
    "pred_lgb",
    "pred_xgb",
    "residual_rank_pred",
    "alpha",
]


def prediction_frame(
    df: pd.DataFrame,
    pred: np.ndarray,
    label_cols: list[str] | None = None,
    pred_col: str = "pred",
) -> pd.DataFrame:
    cols = ["trade_date", "ts_code"] + [c for c in (label_cols or []) if c in df.columns]
    out = df[cols].copy()
    out[pred_col] = pred.astype(np.float32, copy=False)
    return out


def load_prediction_frame(path: str | Path, pred_name: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(BASE_PRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file {path} missing required columns: {sorted(missing)}")
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    if pred_name:
        df = df.rename(columns={"pred": f"pred_{pred_name}"})
    return df


def save_prediction_frame(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
