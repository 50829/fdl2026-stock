from __future__ import annotations

import pandas as pd

from .backtest import BacktestConfig, backtest_rolling_tranche, backtest_topk
from .metrics import prediction_metrics


def evaluate_prediction_scores(
    pred_df: pd.DataFrame,
    label_col: str,
    raw_return_col: str | None = None,
    daily_return_col: str | None = None,
    topk_cfg: BacktestConfig | None = None,
    rolling_cfg: BacktestConfig | None = None,
    rolling_prefix: str = "rolling_",
) -> dict[str, object]:
    metrics = prediction_metrics(pred_df, label_col=label_col, raw_return_col=raw_return_col)

    if raw_return_col:
        metrics.update(backtest_topk(pred_df, return_col=raw_return_col, cfg=topk_cfg or BacktestConfig()))

    if daily_return_col:
        cfg = rolling_cfg or BacktestConfig(mode="rolling_tranche", daily_return_col=daily_return_col)
        rolling = backtest_rolling_tranche(pred_df, cfg=cfg)
        metrics.update({f"{rolling_prefix}{k}": v for k, v in rolling.items()})

    return metrics
