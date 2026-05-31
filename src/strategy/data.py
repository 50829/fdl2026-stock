from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import StrategyBacktestConfig


def load_prediction_data(path: str | Path, score_col: str = "pred", return_col: str = "label_1d") -> pd.DataFrame:
    if score_col == return_col or str(score_col).startswith("label_"):
        raise ValueError(
            f"score_col={score_col!r} would use realized label/return data as the selection signal"
        )
    df = pd.read_parquet(path)
    required = {"trade_date", "ts_code", score_col, return_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df[["trade_date", "ts_code", score_col, return_col]].copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["ts_code"] = out["ts_code"].astype(str)
    out[score_col] = out[score_col].astype("float32")
    out[return_col] = out[return_col].astype("float32")
    out = out.dropna(subset=[score_col, return_col])
    return out.sort_values(["trade_date", score_col], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def merge_feature_columns(
    df: pd.DataFrame,
    feature_path: str | Path,
    columns: list[str],
) -> pd.DataFrame:
    path = Path(feature_path)
    if not path.exists() or not columns:
        return df
    required = ["trade_date", "ts_code"]
    features = pd.read_parquet(path, columns=required + columns)
    features["trade_date"] = features["trade_date"].astype(str)
    features["ts_code"] = features["ts_code"].astype(str)
    out = df.merge(features, on=["trade_date", "ts_code"], how="left")
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype("float32")
    return out


def prepare_maps(df: pd.DataFrame, cfg: StrategyBacktestConfig) -> tuple[list[str], dict[str, pd.DataFrame], pd.DataFrame]:
    dates = sorted(df["trade_date"].unique().tolist())
    day_map: dict[str, pd.DataFrame] = {}
    rows = []
    for d, g in df.groupby("trade_date", sort=True):
        day = g.sort_values(cfg.score_col, ascending=False, kind="mergesort").copy()
        day["rank"] = np.arange(1, len(day) + 1, dtype=np.int32)
        day = day.set_index("ts_code", drop=False)
        day_map[str(d)] = day
        rows.append(day[["trade_date", "ts_code", cfg.return_col]].reset_index(drop=True))
    ret_panel = pd.concat(rows, ignore_index=True).pivot(index="trade_date", columns="ts_code", values=cfg.return_col).sort_index()
    return dates, day_map, ret_panel
