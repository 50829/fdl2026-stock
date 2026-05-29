"""Feature metadata loading, validation, and column resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ID_COLUMNS = {"trade_date", "ts_code"}
FEATURE_GROUP_PREFIXES = {
    "core_price": ("ret_1__", "open_gap__", "intraday_ret__", "high_low_range__", "close_vwap_gap__"),
    "volume_liquidity": ("log_vol__", "log_amount__", "volume_ratio_", "turnover_rate__", "turnover_mean_"),
    "momentum_ma": ("momentum_", "ma_gap_"),
    "volatility": ("volatility_",),
    "moneyflow": ("net_mf_", "large_net_", "buy_lg_", "buy_elg_", "moneyflow_ratio_"),
    "fundamental_size": ("pb__", "ps_ttm__", "log_total_mv__", "log_circ_mv__", "pe_ttm", "dv_ttm"),
    "oscillator": ("rsi_", "kdj_"),
    "macd": ("macd_",),
    "industry_relative": ("industry_momentum_20", "stock_minus_industry_mom_20", "stock_rank_in_industry"),
    "candlestick": ("close_position__",),
    "volume_price_interaction": ("corr_ret_logvol_chg_", "ret_x_volume_ratio_", "turnover_shock_"),
}


def read_feature_meta(path: str | Path) -> dict[str, Any]:
    """Read feature metadata JSON."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_parquet_feature_columns(path: str | Path) -> list[str]:
    """Read feature column names from parquet schema without loading the table."""
    return [col for col in pq.read_schema(path).names if col not in ID_COLUMNS]


def build_feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    """Build named feature groups from the full feature list."""
    groups = {
        name: [col for col in feature_columns if col.startswith(prefixes)]
        for name, prefixes in FEATURE_GROUP_PREFIXES.items()
    }
    groups["ts_zscore"] = [col for col in feature_columns if "__ts_z" in col]
    groups["robust_z"] = [col for col in feature_columns if col.endswith("__cs_robust_z")]
    return {name: cols for name, cols in groups.items() if cols}


def write_feature_meta(
    path: str | Path,
    feature_columns: list[str],
    feature_definitions: dict[str, Any],
    config: dict[str, Any],
    feature_groups: dict[str, list[str]],
) -> None:
    """Write feature metadata JSON."""
    payload = {
        "feature_columns": feature_columns,
        "feature_groups": feature_groups,
        "features": feature_definitions,
        "config": config,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_feature_meta(meta: dict[str, Any], parquet_feature_columns: list[str]) -> None:
    """Validate feature metadata against the actual feature parquet schema.

    `feature_columns` is the full ordered feature list and every metadata column
    must exist in the parquet file.
    """
    feature_columns = list(meta.get("feature_columns", []))
    feature_groups = dict(meta.get("feature_groups", {}))
    feature_defs = dict(meta.get("features", {}))
    parquet_cols = list(parquet_feature_columns)
    parquet_set = set(parquet_cols)

    if not feature_columns:
        raise ValueError("feature_meta.json missing non-empty `feature_columns`.")
    if not feature_groups:
        raise ValueError("feature_meta.json missing non-empty `feature_groups`.")

    group_cols = {col for cols in feature_groups.values() for col in cols}
    meta_cols = set(feature_columns) | group_cols | set(feature_defs)
    missing_in_parquet = sorted(meta_cols - parquet_set)
    if missing_in_parquet:
        raise ValueError(f"feature_meta columns missing from features.parquet: {missing_in_parquet}")

    missing_in_meta = sorted(parquet_set - meta_cols)
    if missing_in_meta:
        raise ValueError(f"features.parquet columns missing from feature_meta: {missing_in_meta}")

    feature_set = set(feature_columns)
    if feature_set != parquet_set:
        extra = sorted(feature_set - parquet_set)
        missing = sorted(parquet_set - feature_set)
        raise ValueError(f"feature_columns must equal features.parquet columns. extra={extra}, missing={missing}")

    if feature_columns != parquet_cols:
        raise ValueError("feature_columns must preserve features.parquet column order.")


def resolve_feature_columns(
    meta: dict[str, Any],
    parquet_feature_columns: list[str],
    mode: str = "default",
    groups: list[str] | None = None,
    columns: list[str] | None = None,
) -> list[str]:
    """Resolve model input columns while preserving parquet column order."""
    feature_groups = dict(meta.get("feature_groups", {}))
    parquet_cols = list(parquet_feature_columns)

    if mode == "default":
        selected = set(meta["feature_columns"])
    elif mode == "all":
        selected = set(parquet_cols)
    elif mode == "groups":
        selected = set()
        for group in groups or []:
            if group not in feature_groups:
                raise ValueError(f"Unknown feature group: {group}")
            selected.update(feature_groups[group])
    elif mode == "explicit":
        selected = set(columns or [])
    else:
        raise ValueError(f"Unknown feature selection mode: {mode}")

    missing = sorted(selected - set(parquet_cols))
    if missing:
        raise ValueError(f"Requested feature columns missing from features.parquet: {missing}")
    return [col for col in parquet_cols if col in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="data/processed/feature_meta.json")
    parser.add_argument("--features", default="data/processed/features.parquet")
    args = parser.parse_args()

    meta = read_feature_meta(args.meta)
    parquet_cols = read_parquet_feature_columns(args.features)
    validate_feature_meta(meta, parquet_cols)
    print(
        json.dumps(
            {
                "status": "ok",
                "feature_columns": len(meta["feature_columns"]),
                "parquet_feature_columns": len(parquet_cols),
                "feature_groups": {name: len(cols) for name, cols in meta["feature_groups"].items()},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
