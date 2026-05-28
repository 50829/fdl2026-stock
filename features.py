from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ts_code", "trade_date"], kind="mergesort")

    def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        if "ts_code" not in g.columns:
            g["ts_code"] = str(getattr(g, "name", ""))
        pre_close = g["pre_close"] if "pre_close" in g.columns else g["close"].shift(1)
        g["ret_1"] = g["close"] / (pre_close + 1e-12) - 1.0
        g["log_ret_1"] = np.log1p(g["ret_1"].clip(lower=-0.999))
        g["hl_range"] = (g["high"] - g["low"]) / (pre_close + 1e-12)
        g["oc_range"] = (g["close"] - g["open"]) / (g["open"] + 1e-12)
        g["vol_chg"] = g["vol"].pct_change()
        g["amount_chg"] = g["amount"].pct_change() if "amount" in g.columns else np.nan
        g["ma_5"] = g["close"].rolling(5, min_periods=5).mean()
        g["ma_10"] = g["close"].rolling(10, min_periods=10).mean()
        g["ma_20"] = g["close"].rolling(20, min_periods=20).mean()
        g["mom_5"] = g["close"].pct_change(5)
        g["mom_10"] = g["close"].pct_change(10)
        g["rsi_14"] = _rsi(g["close"], 14)
        g["macd"] = _macd(g["close"])
        g["close_to_ma20"] = g["close"] / (g["ma_20"] + 1e-12) - 1.0
        return g

    gb = df.groupby("ts_code", group_keys=False, sort=False)
    try:
        return gb.apply(_per_stock, include_groups=False)
    except TypeError:
        return gb.apply(_per_stock)


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(n, min_periods=n).mean()
    roll_down = down.rolling(n, min_periods=n).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd - signal_line


def add_future_return_label(df: pd.DataFrame, horizon: int = 1, price_col: str = "close") -> tuple[pd.DataFrame, str]:
    horizon = int(horizon)
    label_col = f"label_ret_{horizon}d"

    def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        if "ts_code" not in g.columns:
            g["ts_code"] = str(getattr(g, "name", ""))
        future = g[price_col].shift(-horizon)
        g[label_col] = future / (g[price_col] + 1e-12) - 1.0
        return g

    gb = df.groupby("ts_code", group_keys=False, sort=False)
    try:
        out = gb.apply(_per_stock, include_groups=False)
    except TypeError:
        out = gb.apply(_per_stock)
    return out, label_col


def default_feature_cols() -> list[str]:
    return [
        "ret_1",
        "log_ret_1",
        "hl_range",
        "oc_range",
        "vol_chg",
        "amount_chg",
        "ma_5",
        "ma_10",
        "ma_20",
        "mom_5",
        "mom_10",
        "rsi_14",
        "macd",
        "close_to_ma20",
    ]


def select_feature_and_label_cols(df: pd.DataFrame, label_col: str, extra_feature_cols: list[str] | None = None) -> tuple[list[str], str]:
    cols = default_feature_cols()
    if extra_feature_cols:
        cols = cols + list(extra_feature_cols)
    cols = [c for c in cols if c in df.columns]
    return cols, label_col

