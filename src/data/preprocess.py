"""Command line data preprocessing pipeline.

Run:
    python -m src.experiments preprocess --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.data.features import (
    DIRECT_FEATURE_COLUMNS,
    ROBUST_Z_COLUMNS,
    TS_ZSCORE_COLUMNS,
    add_industry_relative_features,
    add_basic_features,
    add_fundamental_features,
    add_moneyflow_features,
    add_rolling_features,
    add_technical_indicators,
    add_volume_price_interaction_features,
    cross_section_robust_z,
    cross_section_rank,
    cross_section_source_columns,
    existing_columns,
    rolling_zscore,
)
from src.data.feature_meta import build_feature_groups, write_feature_meta
from src.data.label import add_cross_section_label_rank, add_forward_return_labels, add_market_excess_labels
from src.data.load import downcast_numeric, list_date_files, read_csv


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_panel(raw_dir: Path, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    daily_files = list_date_files(raw_dir / "daily", start_date, end_date)
    frames = []

    for daily_path in tqdm(daily_files, desc="building panel"):
        date = daily_path.stem
        daily = read_csv(daily_path)

        metric_path = raw_dir / "metric" / f"{date}.csv"
        if metric_path.exists():
            metric = read_csv(metric_path).drop(columns=["close"], errors="ignore")
            daily = daily.merge(metric, on=["trade_date", "ts_code"], how="left")

        moneyflow_path = raw_dir / "moneyflow" / f"{date}.csv"
        if moneyflow_path.exists():
            moneyflow = read_csv(moneyflow_path)
            daily = daily.merge(moneyflow, on=["trade_date", "ts_code"], how="left")

        frames.append(downcast_numeric(daily))

    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel["ts_code"] = panel["ts_code"].astype(str)
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return downcast_numeric(panel)


def build_quality_report(panel: pd.DataFrame) -> dict:
    price_cols = [col for col in ["open", "high", "low", "close", "pre_close", "vwap"] if col in panel]
    volume_cols = [col for col in ["vol", "amount"] if col in panel]
    daily_count = panel.groupby("trade_date")["ts_code"].count()

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(panel)),
        "start_date": str(panel["trade_date"].min()),
        "end_date": str(panel["trade_date"].max()),
        "n_trade_dates": int(panel["trade_date"].nunique()),
        "n_stocks": int(panel["ts_code"].nunique()),
        "daily_stock_count": {
            "min": int(daily_count.min()),
            "median": float(daily_count.median()),
            "max": int(daily_count.max()),
        },
        "duplicate_trade_date_ts_code": int(panel.duplicated(["trade_date", "ts_code"]).sum()),
        "non_positive_price_counts": {col: int((panel[col] <= 0).sum()) for col in price_cols},
        "negative_volume_counts": {col: int((panel[col] < 0).sum()) for col in volume_cols},
        "missing_rate_top20": panel.isna().mean().sort_values(ascending=False).head(20).to_dict(),
    }


def load_st_rows(raw_dir: Path, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    files = list_date_files(raw_dir / "stock_st", start_date, end_date)
    frames = []
    for path in files:
        st = read_csv(path)
        if {"trade_date", "ts_code"}.issubset(st.columns):
            frames.append(st[["trade_date", "ts_code"]])
    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", "is_st"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    out["is_st"] = True
    return out


def load_basic_metadata(raw_dir: Path) -> pd.DataFrame:
    basic = read_csv(raw_dir / "basic.csv")
    basic["list_date"] = basic["list_date"].astype(str)
    keep_cols = [col for col in ["ts_code", "market", "industry", "list_date"] if col in basic.columns]
    return basic[keep_cols]


def build_universe(
    panel: pd.DataFrame,
    raw_dir: Path,
    min_list_days: int,
    start_date: str,
    end_date: str | None,
    config: dict,
) -> pd.DataFrame:
    basic = load_basic_metadata(raw_dir)

    universe = panel[["trade_date", "ts_code", "vol", "amount"]].copy()
    universe = universe.merge(basic, on="ts_code", how="left")

    st = load_st_rows(raw_dir, start_date, end_date)
    universe = universe.merge(st, on=["trade_date", "ts_code"], how="left")
    universe["is_st"] = universe["is_st"].fillna(False)

    trade_dt = pd.to_datetime(universe["trade_date"], format="%Y%m%d", errors="coerce")
    list_dt = pd.to_datetime(universe["list_date"], format="%Y%m%d", errors="coerce")
    listed_days = (trade_dt - list_dt).dt.days + 1
    listed_days = listed_days.where(listed_days > 0, 0).fillna(0)
    universe["listed_days_in_data"] = listed_days.astype("int32")

    liquidity_cfg = config.get("universe", {}).get("liquidity_filter", {}) or {}
    liquidity_window = int(liquidity_cfg.get("window", 20))
    liquidity_min_periods = min(5, liquidity_window)
    sorted_universe = universe.sort_values(["ts_code", "trade_date"])
    universe["amount_mean_20"] = sorted_universe.groupby("ts_code")["amount"].transform(
        lambda s: s.rolling(liquidity_window, min_periods=liquidity_min_periods).mean()
    ).reindex(universe.index)

    base_mask = (
        universe["market"].ne("北交所")
        & ~universe["is_st"]
        & universe["vol"].gt(0)
        & universe["amount"].gt(0)
        & universe["listed_days_in_data"].ge(min_list_days)
    )

    liquidity_enabled = bool(liquidity_cfg.get("enabled", True))
    bottom_pct = float(liquidity_cfg.get("bottom_pct", 0.2))
    if liquidity_enabled and bottom_pct > 0:
        amount_mean = universe["amount_mean_20"].where(base_mask)
        liquidity_rank = amount_mean.groupby(universe["trade_date"]).rank(pct=True)
        liquidity_mask = liquidity_rank.gt(bottom_pct).fillna(False)
    else:
        liquidity_mask = pd.Series(True, index=universe.index)

    universe["passes_liquidity"] = liquidity_mask.astype(bool)
    universe["in_universe"] = base_mask & universe["passes_liquidity"]
    return universe[
        [
            "trade_date",
            "ts_code",
            "in_universe",
            "is_st",
            "market",
            "industry",
            "listed_days_in_data",
            "amount_mean_20",
            "passes_liquidity",
        ]
    ]


def build_raw_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = add_basic_features(panel)
    df = add_moneyflow_features(df)
    df = add_fundamental_features(df)
    df = add_rolling_features(df)
    df = add_technical_indicators(df)
    return add_volume_price_interaction_features(df)


def build_features(panel: pd.DataFrame, universe: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    df = build_raw_features(panel)
    df = df.merge(
        universe[["trade_date", "ts_code", "in_universe", "industry"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    df = df[df["in_universe"].fillna(False)].drop(columns=["in_universe"]).copy()
    df = add_industry_relative_features(df)

    cs_columns = cross_section_source_columns(df)
    ts_columns = existing_columns(df, TS_ZSCORE_COLUMNS)
    robust_columns = existing_columns(df, ROBUST_Z_COLUMNS)

    cs_features, cs_meta = cross_section_rank(df, cs_columns)
    ts_features, ts_meta = rolling_zscore(df, ts_columns, window=int(config["sample"].get("lookback", 60)))
    robust_features, robust_meta = cross_section_robust_z(
        df,
        robust_columns,
        clip=float(config.get("features", {}).get("robust_z_clip", 3.0)),
    )

    features = cs_features.merge(ts_features, on=["trade_date", "ts_code"], how="left")
    features = features.merge(robust_features, on=["trade_date", "ts_code"], how="left")
    meta = {**cs_meta, **ts_meta, **robust_meta}
    direct_feature_frames = []
    for col in existing_columns(df, DIRECT_FEATURE_COLUMNS):
        direct = df[["trade_date", "ts_code"]].copy()
        is_industry_rank = col == "stock_rank_in_industry"
        direct[col] = (
            df[col]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0 if is_industry_rank else 1)
            .astype("float32" if is_industry_rank else "int8")
        )
        direct_feature_frames.append(direct)
        meta[col] = {
            "source_column": "momentum_20" if is_industry_rank else col,
            "processor": "industry_rank" if is_industry_rank else "binary_mask",
            "missing_rate": float(df[col].isna().mean()),
        }
    for direct in direct_feature_frames:
        features = features.merge(direct, on=["trade_date", "ts_code"], how="left")

    feature_cols = [col for col in features.columns if col not in {"trade_date", "ts_code"}]
    features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return downcast_numeric(features), meta


def filter_labels_to_universe(labels: pd.DataFrame, universe: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Keep labels whose decision, buy, and sell dates are inside the tradable universe."""
    out = labels.copy()
    date_universe = universe[["trade_date", "ts_code", "in_universe"]].copy()
    out = out.merge(date_universe, on=["trade_date", "ts_code"], how="left")
    mask = out["in_universe"].fillna(False).to_numpy(dtype=bool)
    out = out.drop(columns=["in_universe"])

    buy_universe = date_universe.rename(columns={"trade_date": "buy_date", "in_universe": "buy_in_universe"})
    out = out.merge(buy_universe, on=["buy_date", "ts_code"], how="left")
    mask = mask & out["buy_in_universe"].fillna(False).to_numpy(dtype=bool)
    out = out.drop(columns=["buy_in_universe"])

    for horizon in horizons:
        sell_col = f"sell_date_{horizon}d"
        sell_flag = f"sell_in_universe_{horizon}d"
        sell_universe = date_universe.rename(columns={"trade_date": sell_col, "in_universe": sell_flag})
        out = out.merge(sell_universe, on=[sell_col, "ts_code"], how="left")
        mask = mask & out[sell_flag].fillna(False).to_numpy(dtype=bool)
        out = out.drop(columns=[sell_flag])
    return out[mask].copy()


def write_processed_readme(processed_dir: Path, report: dict, config: dict, feature_count: int) -> None:
    text = f"""# Processed Data

本目录由 `python -m src.experiments preprocess --config configs/config.yaml` 生成。

## 数据范围

- 开始日期：`{report["start_date"]}`
- 结束日期：`{report["end_date"]}`
- 交易日数量：`{report["n_trade_dates"]}`
- 股票数量：`{report["n_stocks"]}`
- 面板行数：`{report["rows"]}`
- 特征数量：`{feature_count}`

## 文件说明

- `panel.parquet`：合并后的原始日频面板，键为 `trade_date, ts_code`。
- `data_quality.json`：数据质量检查摘要。
- `universe.parquet`：每日股票池过滤结果。
- `features.parquet`：模型输入特征。
- `labels.parquet`：`label_1d`、`label_5d`、市场超额标签及其截面 rank 标签。
- `feature_meta.json`：默认特征列、特征分组、处理方式和生成配置。
- `splits.json`：训练、验证、测试时间切分。

## 关键约束

- 特征只使用当日盘后及以前可获得的数据。
- 截面标准化只在当日 `in_universe=True` 的可交易股票池内部计算。
- 标签从 `T+1` 买入价开始计算，避免使用不可交易收益。
- 标签使用全市场交易日历对齐，不按单只股票记录跳过停牌/缺失日期。
- 训练、验证、测试按时间切分，禁止随机日期切分。
"""
    (processed_dir / "README.md").write_text(text, encoding="utf-8")


def next_yyyymmdd(date_text: str) -> str:
    date = pd.Timestamp(str(date_text))
    return (date + pd.Timedelta(days=1)).strftime("%Y%m%d")


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    start_date = str(config["sample"]["start_date"]).replace("-", "")
    end_date = args.end_date.replace("-", "") if args.end_date else None
    min_list_days = int(config.get("universe", {}).get("min_list_days", 60))

    panel = build_panel(raw_dir, start_date, end_date)
    panel.to_parquet(processed_dir / "panel.parquet", index=False)

    quality = build_quality_report(panel)
    (processed_dir / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    universe = build_universe(panel, raw_dir, min_list_days, start_date, end_date, config)
    universe.to_parquet(processed_dir / "universe.parquet", index=False)

    features, feature_meta = build_features(panel, universe, config)
    features.to_parquet(processed_dir / "features.parquet", index=False)

    horizons = [1, 5]
    trade_dates = sorted(panel["trade_date"].astype(str).unique().tolist())
    labels = add_forward_return_labels(panel, horizons=horizons, trade_dates=trade_dates)
    labels = labels.dropna(subset=["label_1d", "label_5d"])
    labels = filter_labels_to_universe(labels, universe, horizons=horizons)
    label_columns = ["label_1d", "label_5d"]
    labels = add_market_excess_labels(labels, label_columns)
    labels = add_cross_section_label_rank(labels, label_columns + ["label_1d_excess", "label_5d_excess"])
    labels.to_parquet(processed_dir / "labels.parquet", index=False)

    feature_columns = [col for col in features.columns if col not in {"trade_date", "ts_code"}]
    feature_groups = build_feature_groups(feature_columns)
    write_feature_meta(
        processed_dir / "feature_meta.json",
        feature_columns,
        feature_meta,
        {
            "config_path": args.config,
            "start_date": start_date,
            "end_date": end_date,
            "min_list_days": min_list_days,
            "liquidity_filter": config.get("universe", {}).get("liquidity_filter", {}),
            "label_horizons": horizons,
            "label_calendar": "global_trade_dates",
            "normalization_universe": "in_universe_only",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        feature_groups=feature_groups,
    )

    splits = {
        "train": [config["sample"]["start_date"], config["sample"]["train_end"]],
        "valid": [
            next_yyyymmdd(str(config["sample"]["train_end"])),
            config["sample"]["valid_end"],
        ],
        "test": [
            next_yyyymmdd(str(config["sample"]["valid_end"])),
            quality["end_date"],
        ],
    }
    (processed_dir / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    write_processed_readme(processed_dir, quality, config, len(feature_columns))


if __name__ == "__main__":
    run_cli()
