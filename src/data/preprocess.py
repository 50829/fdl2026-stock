"""Command line data preprocessing pipeline.

Run:
    python -m src.data.preprocess --config configs/config.yaml
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
    add_basic_features,
    add_fundamental_features,
    add_moneyflow_features,
    add_rolling_features,
    add_technical_indicators,
    add_volume_price_interaction_features,
    cross_section_rank,
    rolling_zscore,
    write_feature_meta,
)
from src.data.label import add_cross_section_label_rank, add_forward_return_labels
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


def build_universe(panel: pd.DataFrame, raw_dir: Path, min_list_days: int, start_date: str, end_date: str | None) -> pd.DataFrame:
    basic = read_csv(raw_dir / "basic.csv")
    basic["list_date"] = basic["list_date"].astype(str)
    basic = basic[["ts_code", "market", "list_date"]]

    universe = panel[["trade_date", "ts_code", "vol", "amount"]].copy()
    universe = universe.merge(basic, on="ts_code", how="left")

    st = load_st_rows(raw_dir, start_date, end_date)
    universe = universe.merge(st, on=["trade_date", "ts_code"], how="left")
    universe["is_st"] = universe["is_st"].fillna(False)

    listed_days = universe.sort_values(["ts_code", "trade_date"]).groupby("ts_code").cumcount() + 1
    universe["listed_days_in_data"] = listed_days.astype("int32")
    universe["in_universe"] = (
        universe["market"].ne("北交所")
        & ~universe["is_st"]
        & universe["vol"].gt(0)
        & universe["amount"].gt(0)
        & universe["listed_days_in_data"].ge(min_list_days)
    )
    return universe[["trade_date", "ts_code", "in_universe", "is_st", "market", "listed_days_in_data"]]


def build_features(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    df = add_basic_features(panel)
    df = add_moneyflow_features(df)
    df = add_fundamental_features(df)
    df = add_rolling_features(df)
    df = add_technical_indicators(df)
    df = add_volume_price_interaction_features(df)

    cs_candidates = [
        "ret_1",
        "open_gap",
        "intraday_ret",
        "high_low_range",
        "close_vwap_gap",
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
    ]
    for window in [5, 10, 20, 60]:
        cs_candidates.extend(
            [
                f"momentum_{window}",
                f"volatility_{window}",
                f"ma_gap_{window}",
                f"volume_ratio_{window}",
                f"turnover_mean_{window}",
                f"moneyflow_ratio_{window}",
            ]
        )
    cs_columns = [col for col in cs_candidates if col in df.columns]

    ts_columns = [col for col in ["ret_1", "momentum_5", "momentum_20", "volatility_20", "volume_ratio_20"] if col in df.columns]

    cs_features, cs_meta = cross_section_rank(df, cs_columns)
    ts_features, ts_meta = rolling_zscore(df, ts_columns, window=int(config["sample"].get("lookback", 60)))

    features = cs_features.merge(ts_features, on=["trade_date", "ts_code"], how="left")
    meta = {**cs_meta, **ts_meta}
    for col in ["pe_ttm_missing", "dv_ttm_missing"]:
        if col in df.columns:
            features[col] = df[col].fillna(1).astype("int8")
            meta[col] = {"source_column": col, "processor": "binary_mask", "missing_rate": 0.0}

    feature_cols = [col for col in features.columns if col not in {"trade_date", "ts_code"}]
    features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return downcast_numeric(features), meta


def build_feature_groups(feature_columns: list[str]) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Group feature columns for ablation experiments."""
    groups = {
        "core_price": [
            col for col in feature_columns
            if col.startswith(("ret_1__", "open_gap__", "intraday_ret__", "high_low_range__", "close_vwap_gap__"))
        ],
        "volume_liquidity": [
            col for col in feature_columns
            if col.startswith(("log_vol__", "log_amount__", "volume_ratio_", "turnover_rate__", "turnover_mean_"))
        ],
        "momentum_ma": [
            col for col in feature_columns
            if col.startswith(("momentum_", "ma_gap_"))
        ],
        "volatility": [
            col for col in feature_columns
            if col.startswith("volatility_")
        ],
        "moneyflow": [
            col for col in feature_columns
            if col.startswith(("net_mf_", "large_net_", "buy_lg_", "buy_elg_", "moneyflow_ratio_"))
        ],
        "fundamental_size": [
            col for col in feature_columns
            if col.startswith(("pb__", "ps_ttm__", "log_total_mv__", "log_circ_mv__", "pe_ttm", "dv_ttm"))
        ],
        "ts_zscore": [
            col for col in feature_columns
            if "__ts_z" in col
        ],
        "oscillator": [
            col for col in feature_columns
            if col.startswith(("rsi_", "kdj_"))
        ],
        "macd": [
            col for col in feature_columns
            if col.startswith("macd_")
        ],
        "volume_price_interaction": [
            col for col in feature_columns
            if col.startswith(("corr_ret_logvol_chg_", "ret_x_volume_ratio_", "turnover_shock_"))
        ],
    }
    groups = {name: cols for name, cols in groups.items() if cols}
    default_groups = [
        "core_price",
        "volume_liquidity",
        "momentum_ma",
        "volatility",
        "moneyflow",
        "fundamental_size",
        "ts_zscore",
        "oscillator",
        "volume_price_interaction",
    ]
    default_columns = []
    for group in default_groups:
        default_columns.extend(groups.get(group, []))
    default_columns = list(dict.fromkeys(default_columns))
    return groups, default_groups, default_columns


def write_processed_readme(processed_dir: Path, report: dict, config: dict, feature_count: int, all_feature_count: int | None = None) -> None:
    all_feature_count = all_feature_count or feature_count
    text = f"""# Processed Data

本目录由 `python -m src.data.preprocess --config configs/config.yaml` 生成。

## 数据范围

- 开始日期：`{report["start_date"]}`
- 结束日期：`{report["end_date"]}`
- 交易日数量：`{report["n_trade_dates"]}`
- 股票数量：`{report["n_stocks"]}`
- 面板行数：`{report["rows"]}`
- 默认特征数量：`{feature_count}`
- 特征池总数量：`{all_feature_count}`

## 文件说明

- `panel.parquet`：合并后的原始日频面板，键为 `trade_date, ts_code`。
- `data_quality.json`：数据质量检查摘要。
- `universe.parquet`：每日股票池过滤结果。
- `features.parquet`：模型输入特征。
- `labels.parquet`：`label_1d`、`label_5d` 及其截面 rank 标签。
- `feature_meta.json`：默认特征列、完整特征池、特征分组、处理方式和生成配置。
- `splits.json`：训练、验证、测试时间切分。

## 关键约束

- 特征只使用当日盘后及以前可获得的数据。
- 标签从 `T+1` 买入价开始计算，避免使用不可交易收益。
- 训练、验证、测试按时间切分，禁止随机日期切分。
"""
    (processed_dir / "README.md").write_text(text, encoding="utf-8")


def next_yyyymmdd(date_text: str) -> str:
    date = pd.Timestamp(str(date_text))
    return (date + pd.Timedelta(days=1)).strftime("%Y%m%d")


def main() -> None:
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

    universe = build_universe(panel, raw_dir, min_list_days, start_date, end_date)
    universe.to_parquet(processed_dir / "universe.parquet", index=False)

    features, feature_meta = build_features(panel, config)
    features = features.merge(universe[["trade_date", "ts_code", "in_universe"]], on=["trade_date", "ts_code"], how="left")
    features = features[features["in_universe"].fillna(False)].drop(columns=["in_universe"])
    features.to_parquet(processed_dir / "features.parquet", index=False)

    labels = add_forward_return_labels(panel, horizons=[1, 5])
    labels = labels.dropna(subset=["label_1d", "label_5d"])
    labels = add_cross_section_label_rank(labels, ["label_1d", "label_5d"])
    labels = labels.merge(universe[["trade_date", "ts_code", "in_universe"]], on=["trade_date", "ts_code"], how="left")
    labels = labels[labels["in_universe"].fillna(False)].drop(columns=["in_universe"])
    labels.to_parquet(processed_dir / "labels.parquet", index=False)

    all_feature_columns = [col for col in features.columns if col not in {"trade_date", "ts_code"}]
    feature_groups, default_feature_groups, default_feature_columns = build_feature_groups(all_feature_columns)
    write_feature_meta(
        processed_dir / "feature_meta.json",
        default_feature_columns,
        feature_meta,
        {
            "config_path": args.config,
            "start_date": start_date,
            "end_date": end_date,
            "min_list_days": min_list_days,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        all_feature_columns=all_feature_columns,
        feature_groups=feature_groups,
        default_feature_groups=default_feature_groups,
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
    write_processed_readme(processed_dir, quality, config, len(default_feature_columns), len(all_feature_columns))


if __name__ == "__main__":
    main()
