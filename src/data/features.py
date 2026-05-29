"""Feature engineering utilities used by the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


ROLLING_WINDOWS = [5, 10, 20, 60]

CROSS_SECTION_BASE_COLUMNS = [
    "ret_1",
    "open_gap",
    "intraday_ret",
    "high_low_range",
    "close_vwap_gap",
    "close_position",
    "log_vol",
    "log_amount",
    "turnover_rate",
    "pb",
    "ps_ttm",
    "log_total_mv",
    "log_circ_mv",
    "net_mf_amount_ratio",
    "large_net_amount_ratio",
    "buy_lg_amount_ratio",
    "buy_elg_amount_ratio",
    "rsi_6",
    "rsi_14",
    "rsi_24",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "macd_dif",
    "macd_dea",
    "macd_hist",
    "corr_ret_logvol_chg_10",
    "corr_ret_logvol_chg_20",
    "ret_x_volume_ratio_10",
    "ret_x_volume_ratio_20",
    "turnover_shock_20",
    "industry_momentum_20",
    "stock_minus_industry_mom_20",
]

ROLLING_CROSS_SECTION_PATTERNS = [
    "momentum_{window}",
    "volatility_{window}",
    "ma_gap_{window}",
    "volume_ratio_{window}",
    "turnover_mean_{window}",
    "moneyflow_ratio_{window}",
]

TS_ZSCORE_COLUMNS = ["ret_1", "momentum_5", "momentum_20", "volatility_20", "volume_ratio_20"]
ROBUST_Z_COLUMNS = ["ret_1", "momentum_20", "volatility_20", "log_total_mv"]
DIRECT_FEATURE_COLUMNS = ["pe_ttm_missing", "dv_ttm_missing", "stock_rank_in_industry"]


def existing_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return candidate columns that exist in the dataframe, preserving order."""
    return [col for col in candidates if col in df.columns]


def cross_section_source_columns(df: pd.DataFrame, windows: list[int] | None = None) -> list[str]:
    """Return raw columns that should get date-wise cross-section rank features."""
    windows = windows or ROLLING_WINDOWS
    candidates = list(CROSS_SECTION_BASE_COLUMNS)
    for window in windows:
        candidates.extend(pattern.format(window=window) for pattern in ROLLING_CROSS_SECTION_PATTERNS)
    return existing_columns(df, candidates)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add same-day features available after market close."""
    out = df.copy()
    out["ret_1"] = out["close"] / out["pre_close"] - 1
    out["open_gap"] = out["open"] / out["pre_close"] - 1
    out["intraday_ret"] = out["close"] / out["open"] - 1
    out["high_low_range"] = out["high"] / out["low"] - 1
    out["close_vwap_gap"] = out["close"] / out["vwap"] - 1
    high_low = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_position"] = (out["close"] - out["low"]) / high_low
    out["log_vol"] = np.log1p(out["vol"])
    out["log_amount"] = np.log1p(out["amount"])
    return out


def add_moneyflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert absolute money-flow fields into amount-normalized features."""
    out = df.copy()
    amount = out["amount"].replace(0, np.nan)

    if "net_mf_amount" in out.columns:
        out["net_mf_amount_ratio"] = out["net_mf_amount"] / amount

    required = {"buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount"}
    if required.issubset(out.columns):
        out["large_net_amount_ratio"] = (
            out["buy_lg_amount"]
            + out["buy_elg_amount"]
            - out["sell_lg_amount"]
            - out["sell_elg_amount"]
        ) / amount
        out["buy_lg_amount_ratio"] = out["buy_lg_amount"] / amount
        out["buy_elg_amount_ratio"] = out["buy_elg_amount"] / amount
    return out


def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple transformed fundamental features and missing masks."""
    out = df.copy()
    for col in ["total_mv", "circ_mv"]:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col])
    for col in ["pe_ttm", "dv_ttm"]:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype("int8")
    return out


def add_rolling_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add per-stock rolling features using only current and historical rows."""
    windows = windows or ROLLING_WINDOWS
    out = df.sort_values(["ts_code", "trade_date"]).copy()
    grouped = out.groupby("ts_code", group_keys=False)

    for window in windows:
        out[f"momentum_{window}"] = grouped["close"].transform(lambda s: s / s.shift(window) - 1)
        out[f"volatility_{window}"] = grouped["ret_1"].transform(
            lambda s: s.rolling(window, min_periods=window).std()
        )
        out[f"ma_gap_{window}"] = grouped["close"].transform(
            lambda s: s / s.rolling(window, min_periods=window).mean() - 1
        )
        out[f"volume_ratio_{window}"] = grouped["vol"].transform(
            lambda s: s / s.rolling(window, min_periods=window).mean()
        )
        if "turnover_rate" in out.columns:
            out[f"turnover_mean_{window}"] = grouped["turnover_rate"].transform(
                lambda s: s.rolling(window, min_periods=window).mean()
            )
        if "net_mf_amount_ratio" in out.columns:
            out[f"moneyflow_ratio_{window}"] = grouped["net_mf_amount_ratio"].transform(
                lambda s: s.rolling(window, min_periods=window).mean()
            )
    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add classic technical indicators as candidate features.

    These indicators are computed per stock using only current and historical
    OHLCV rows, so they are available after the current day's close.
    """
    out = df.sort_values(["ts_code", "trade_date"]).copy()
    grouped = out.groupby("ts_code", group_keys=False)

    def rsi(close: pd.Series, window: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    out["rsi_6"] = grouped["close"].transform(lambda s: rsi(s, 6))
    out["rsi_14"] = grouped["close"].transform(lambda s: rsi(s, 14))
    out["rsi_24"] = grouped["close"].transform(lambda s: rsi(s, 24))

    low_9 = grouped["low"].transform(lambda s: s.rolling(9, min_periods=9).min())
    high_9 = grouped["high"].transform(lambda s: s.rolling(9, min_periods=9).max())
    rsv = (out["close"] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
    out["kdj_k"] = rsv.groupby(out["ts_code"]).transform(lambda s: s.ewm(alpha=1 / 3, adjust=False).mean())
    out["kdj_d"] = out["kdj_k"].groupby(out["ts_code"]).transform(lambda s: s.ewm(alpha=1 / 3, adjust=False).mean())
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]

    ema_12 = grouped["close"].transform(lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean())
    ema_26 = grouped["close"].transform(lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean())
    out["macd_dif"] = ema_12 - ema_26
    out["macd_dea"] = out["macd_dif"].groupby(out["ts_code"]).transform(
        lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    out["macd_hist"] = 2 * (out["macd_dif"] - out["macd_dea"])
    return out


def add_volume_price_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candidate features describing price-volume coordination."""
    out = df.sort_values(["ts_code", "trade_date"]).copy()
    grouped = out.groupby("ts_code", group_keys=False)
    out["log_vol_chg"] = grouped["log_vol"].diff()

    for window in [10, 20]:
        out[f"corr_ret_logvol_chg_{window}"] = grouped.apply(
            lambda g: g["ret_1"].rolling(window, min_periods=window).corr(g["log_vol_chg"]),
            include_groups=False,
        ).reset_index(level=0, drop=True)

        volume_ratio_col = f"volume_ratio_{window}"
        if volume_ratio_col in out.columns:
            out[f"ret_x_volume_ratio_{window}"] = out["ret_1"] * out[volume_ratio_col]

    if "turnover_rate" in out.columns:
        turnover_mean_20 = grouped["turnover_rate"].transform(lambda s: s.rolling(20, min_periods=20).mean())
        out["turnover_shock_20"] = out["turnover_rate"] / turnover_mean_20.replace(0, np.nan) - 1
    return out


def add_industry_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add industry-relative features computed inside the tradable universe."""
    out = df.copy()
    if "industry" not in out.columns or "momentum_20" not in out.columns:
        return out

    industry_keys = [out["trade_date"], out["industry"].fillna("UNKNOWN")]
    industry_mom = out["momentum_20"].groupby(industry_keys).transform("mean")
    out["industry_momentum_20"] = industry_mom
    out["stock_minus_industry_mom_20"] = out["momentum_20"] - industry_mom

    rank = out["momentum_20"].groupby(industry_keys).rank(pct=True)
    out["stock_rank_in_industry"] = rank * 2 - 1
    return out


def cross_section_rank(
    df: pd.DataFrame,
    columns: list[str],
    missing_threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | str]]]:
    """Rank features within each date into [-1, 1], with high-missing dates neutralized."""
    out = df[["trade_date", "ts_code"]].copy()
    meta: dict[str, dict[str, float | str]] = {}

    for col in columns:
        values = df[col].replace([np.inf, -np.inf], np.nan)
        missing_by_date = values.isna().groupby(df["trade_date"]).transform("mean")
        ranked = values.groupby(df["trade_date"]).rank(pct=True)
        ranked = ranked * 2 - 1
        ranked = ranked.mask(missing_by_date > missing_threshold, 0).fillna(0)

        feature_col = f"{col}__cs_rank"
        out[feature_col] = ranked.astype("float32")
        meta[feature_col] = {
            "source_column": col,
            "processor": "cross_section_rank",
            "missing_rate": float(values.isna().mean()),
        }

        if values.isna().mean() > 0:
            mask_col = f"{col}__missing"
            out[mask_col] = values.isna().astype("int8")
            meta[mask_col] = {
                "source_column": col,
                "processor": "binary_mask",
                "missing_rate": float(values.isna().mean()),
            }
    return out, meta


def cross_section_robust_z(
    df: pd.DataFrame,
    columns: list[str],
    clip: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | str]]]:
    """Add date-wise robust z-score features based on median and MAD."""
    out = df[["trade_date", "ts_code"]].copy()
    meta: dict[str, dict[str, float | str]] = {}

    for col in columns:
        values = df[col].replace([np.inf, -np.inf], np.nan)
        grouped = values.groupby(df["trade_date"])
        median = grouped.transform("median")
        abs_dev = (values - median).abs()
        mad = abs_dev.groupby(df["trade_date"]).transform("median")
        denom = 1.4826 * mad.replace(0, np.nan)
        z = ((values - median) / denom).clip(-clip, clip).fillna(0)

        feature_col = f"{col}__cs_robust_z"
        out[feature_col] = z.astype("float32")
        meta[feature_col] = {
            "source_column": col,
            "processor": "cross_section_robust_z",
            "clip": float(clip),
            "missing_rate": float(values.isna().mean()),
        }
    return out, meta


def rolling_zscore(df: pd.DataFrame, columns: list[str], window: int = 60) -> tuple[pd.DataFrame, dict[str, dict[str, float | str]]]:
    """Add per-stock rolling z-score features."""
    out = df[["trade_date", "ts_code"]].copy()
    meta: dict[str, dict[str, float | str]] = {}
    grouped = df.sort_values(["ts_code", "trade_date"]).groupby("ts_code", group_keys=False)

    for col in columns:
        values = df[col].replace([np.inf, -np.inf], np.nan)
        mean = grouped[col].transform(lambda s: s.rolling(window, min_periods=max(5, window // 3)).mean())
        std = grouped[col].transform(lambda s: s.rolling(window, min_periods=max(5, window // 3)).std())
        z = ((values - mean) / std.replace(0, np.nan)).clip(-5, 5).fillna(0)

        feature_col = f"{col}__ts_z{window}"
        out[feature_col] = z.astype("float32")
        meta[feature_col] = {
            "source_column": col,
            "processor": "rolling_zscore",
            "window": window,
            "missing_rate": float(values.isna().mean()),
        }
    return out, meta

