from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyBacktestConfig


def risk_score_from_history(
    ret_panel: pd.DataFrame,
    date: str,
    core: list[str],
    candidates: list[str],
    cfg: StrategyBacktestConfig,
) -> pd.Series:
    hist = ret_panel.loc[ret_panel.index < date].tail(cfg.risk_window)
    if len(hist) < 2 or not candidates:
        return pd.Series(0.5, index=candidates)
    core_cols = [c for c in core if c in hist.columns]
    cand_cols = [c for c in candidates if c in hist.columns]
    if not core_cols or not cand_cols:
        return pd.Series(0.5, index=candidates)
    core_ret = hist[core_cols].mean(axis=1)
    ch = hist[cand_cols]
    vol = ch.std(ddof=1).replace([np.inf, -np.inf], np.nan)
    corr = ch.corrwith(core_ret).replace([np.inf, -np.inf], np.nan)
    downside = ch.clip(upper=0.0).std(ddof=1).replace([np.inf, -np.inf], np.nan)

    def pct(s: pd.Series, fill: float) -> pd.Series:
        s = s.fillna(s.median() if s.notna().any() else fill)
        return s.rank(method="average", pct=True).fillna(0.5)

    risk = 0.5 * pct(corr, 0.0) + 0.3 * pct(vol, 0.0) + 0.2 * pct(downside, 0.0)
    return risk.reindex(candidates).fillna(0.5)
