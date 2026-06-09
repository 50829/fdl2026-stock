from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import StrategyBacktestConfig

DEFAULT_TRADE_CONSTRAINT_COLUMNS = ["in_universe", "is_st", "passes_liquidity", "amount_mean_20"]


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


def merge_trade_constraint_columns(
    df: pd.DataFrame,
    constraint_path: str | Path,
    *,
    min_amount_mean_20: float = 0.0,
    buyable_col: str = "is_buyable",
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(constraint_path)
    if not path.exists():
        raise FileNotFoundError(f"trade constraint file does not exist: {path}")
    constraints = pd.read_parquet(path)
    required = ["trade_date", "ts_code"]
    missing_keys = [col for col in required if col not in constraints.columns]
    if missing_keys:
        raise ValueError(f"{path} missing required columns: {missing_keys}")
    keep = required + [col for col in DEFAULT_TRADE_CONSTRAINT_COLUMNS if col in constraints.columns]
    constraints = constraints[keep].copy()
    constraints["trade_date"] = constraints["trade_date"].astype(str)
    constraints["ts_code"] = constraints["ts_code"].astype(str)
    constraints["_constraint_matched"] = True

    out = df.merge(constraints, on=["trade_date", "ts_code"], how="left")
    matched = out["_constraint_matched"].fillna(False).astype(bool)
    buyable = matched.copy()
    used_rules: list[str] = ["matched_constraint_row"]
    if "in_universe" in out.columns:
        out["in_universe"] = out["in_universe"].fillna(False).astype(bool)
        buyable &= out["in_universe"]
        used_rules.append("in_universe")
    if "is_st" in out.columns:
        out["is_st"] = out["is_st"].fillna(False).astype(bool)
        buyable &= ~out["is_st"]
        used_rules.append("not_st")
    if "passes_liquidity" in out.columns:
        out["passes_liquidity"] = out["passes_liquidity"].fillna(False).astype(bool)
        buyable &= out["passes_liquidity"]
        used_rules.append("passes_liquidity")
    if "amount_mean_20" in out.columns:
        out["amount_mean_20"] = pd.to_numeric(out["amount_mean_20"], errors="coerce").astype("float32")
        if min_amount_mean_20 > 0:
            buyable &= out["amount_mean_20"].fillna(0.0) >= float(min_amount_mean_20)
            used_rules.append(f"amount_mean_20>={float(min_amount_mean_20):.0f}")
    out[buyable_col] = buyable.astype(bool)
    out = out.drop(columns=["_constraint_matched"])
    stats = {
        "constraint_path": str(path),
        "rows": int(len(out)),
        "matched_rows": int(matched.sum()),
        "matched_rate": float(matched.mean()) if len(out) else 0.0,
        "buyable_col": buyable_col,
        "buyable_rows": int(out[buyable_col].sum()),
        "buyable_rate": float(out[buyable_col].mean()) if len(out) else 0.0,
        "columns_used": [col for col in keep if col not in required],
        "rules": used_rules,
        "min_amount_mean_20": float(min_amount_mean_20),
    }
    return out, stats


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
