from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.strategy.metrics import max_drawdown, sharpe as _strategy_sharpe


def ic_metrics(pred_df: pd.DataFrame, label_col: str) -> dict[str, float | int]:
    if pred_df.empty:
        return {"ic_mean": math.nan, "ic_std": math.nan, "icir": math.nan, "ic_days": 0}

    rows = []
    for d, g in pred_df.groupby("trade_date", sort=True):
        g = g.dropna(subset=["pred", label_col])
        if len(g) < 3:
            continue
        pred_rank = g["pred"].rank(method="average").to_numpy(dtype=np.float64)
        label_rank = g[label_col].rank(method="average").to_numpy(dtype=np.float64)
        if np.allclose(pred_rank, pred_rank[0]) or np.allclose(label_rank, label_rank[0]):
            continue
        ic = float(np.corrcoef(pred_rank, label_rank)[0, 1])
        if np.isfinite(ic):
            rows.append({"trade_date": str(d), "ic": float(ic), "n": int(len(g))})
    ic_df = pd.DataFrame(rows)
    if ic_df.empty:
        return {"ic_mean": math.nan, "ic_std": math.nan, "icir": math.nan, "ic_days": 0}
    ic_mean = float(ic_df["ic"].mean())
    ic_std = float(ic_df["ic"].std(ddof=0))
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": float(ic_mean / (ic_std + 1e-12)),
        "ic_days": int(len(ic_df)),
    }


def prediction_metrics(pred_df: pd.DataFrame, label_col: str, raw_return_col: str | None = None) -> dict[str, object]:
    n = int(len(pred_df))
    if n and label_col in pred_df.columns:
        diff = pred_df["pred"].to_numpy(dtype=np.float64) - pred_df[label_col].to_numpy(dtype=np.float64)
        mse = float(np.mean(diff * diff))
    else:
        mse = math.nan
    metrics: dict[str, object] = {
        "samples": n,
        "mse": mse,
        "label_col": label_col,
    }
    if raw_return_col is not None:
        metrics["raw_return_col"] = raw_return_col
    metrics.update(ic_metrics(pred_df, label_col=label_col))
    return metrics


def sharpe_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    return _strategy_sharpe(returns, periods_per_year)
