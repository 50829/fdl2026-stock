from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import metrics_from_curve


def build_equal_weight_benchmark(
    df: pd.DataFrame,
    return_col: str = "label_1d",
    name: str = "benchmark_equal_weight_universe",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    equity = 1.0
    for date, day in df.groupby("trade_date", sort=True):
        returns = day[return_col].dropna()
        net_return = float(returns.mean()) if len(returns) else 0.0
        equity *= 1.0 + net_return
        rows.append(
            {
                "trade_date": str(date),
                "gross_return": net_return,
                "transaction_cost": 0.0,
                "net_return": net_return,
                "turnover": 0.0,
                "equity": equity,
                "n_holdings": int(len(returns)),
            }
        )
    curve = pd.DataFrame(rows)
    return {
        "metrics": metrics_from_curve(curve, name=name, strategy="benchmark_equal_weight"),
        "curve": curve,
        "trades": pd.DataFrame(),
        "holdings": pd.DataFrame(),
    }


def load_index_weight_data(path: str | Path, index_code: str) -> pd.DataFrame:
    path = Path(path)

    def read_csv(file_obj: Any) -> pd.DataFrame:
        return pd.read_csv(
            file_obj,
            dtype={"index_code": str, "con_code": str, "trade_date": str, "weight": float},
        )

    parts: list[pd.DataFrame] = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith(".csv") and index_code in name:
                    with zf.open(name) as f:
                        parts.append(read_csv(f))
    elif path.is_dir():
        for csv_path in sorted(path.rglob("*.csv")):
            if index_code in csv_path.name:
                parts.append(read_csv(csv_path))
    else:
        parts.append(read_csv(path))

    if not parts:
        raise ValueError(f"{path} contains no index weight CSV for {index_code}")
    weights = pd.concat(parts, ignore_index=True)
    required = {"index_code", "con_code", "trade_date", "weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"{path} missing index weight columns: {sorted(missing)}")
    weights = weights[weights["index_code"].astype(str) == str(index_code)].copy()
    if weights.empty:
        raise ValueError(f"{path} contains no rows for index_code={index_code}")
    weights["trade_date"] = weights["trade_date"].astype(str)
    weights["con_code"] = weights["con_code"].astype(str)
    weights["weight"] = weights["weight"].astype(float)
    weights = weights.dropna(subset=["trade_date", "con_code", "weight"])
    weights = weights[weights["weight"] > 0]
    return weights.sort_values(["trade_date", "weight"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def build_index_weight_benchmark(
    df: pd.DataFrame,
    weight_path: str | Path,
    index_code: str = "000300.SH",
    return_col: str = "label_1d",
    name: str | None = None,
) -> dict[str, Any]:
    name = name or f"benchmark_{index_code.replace('.', '_').lower()}_weight"
    weights = load_index_weight_data(weight_path, index_code)
    weight_dates = sorted(weights["trade_date"].unique().tolist())
    weight_by_date = {
        d: g.groupby("con_code", sort=False)["weight"].sum().astype(float)
        for d, g in weights.groupby("trade_date", sort=True)
    }
    returns_by_date = {
        str(d): g.set_index("ts_code")[return_col].astype(float)
        for d, g in df.groupby("trade_date", sort=True)
    }

    rows: list[dict[str, Any]] = []
    equity = 1.0
    weight_pos = -1
    for date in sorted(returns_by_date):
        while weight_pos + 1 < len(weight_dates) and weight_dates[weight_pos + 1] <= date:
            weight_pos += 1
        if weight_pos < 0:
            continue
        source_date = weight_dates[weight_pos]
        weight = weight_by_date[source_date]
        day_ret = returns_by_date[date]
        common = weight.index.intersection(day_ret.index)
        if len(common) == 0:
            continue
        aligned_weight = weight.reindex(common).astype(float)
        aligned_weight = aligned_weight / float(aligned_weight.sum())
        aligned_ret = day_ret.reindex(common).fillna(0.0).astype(float)
        net_return = float((aligned_weight * aligned_ret).sum())
        equity *= 1.0 + net_return
        rows.append(
            {
                "trade_date": str(date),
                "gross_return": net_return,
                "transaction_cost": 0.0,
                "net_return": net_return,
                "turnover": 0.0,
                "equity": equity,
                "n_holdings": int(len(common)),
                "source_weight_date": str(source_date),
                "index_code": str(index_code),
            }
        )

    if not rows:
        raise ValueError(f"index benchmark {index_code} has no overlapping constituents with prediction data")
    curve = pd.DataFrame(rows)
    return {
        "metrics": metrics_from_curve(curve, name=name, strategy="benchmark_index_weight"),
        "curve": curve,
        "trades": pd.DataFrame(),
        "holdings": pd.DataFrame(),
    }


def load_price_benchmark(
    path: str | Path,
    name: str,
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "trade_date" not in df.columns:
        raise ValueError(f"{path} missing trade_date column")
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date", kind="mergesort")

    if "equity" in df.columns:
        equity = df["equity"].astype(float)
        returns = equity.pct_change().fillna(equity.iloc[0] - 1.0)
    elif "return" in df.columns:
        returns = df["return"].astype(float).fillna(0.0)
        equity = (1.0 + returns).cumprod()
    elif "net_return" in df.columns:
        returns = df["net_return"].astype(float).fillna(0.0)
        equity = (1.0 + returns).cumprod()
    else:
        price_col = next((col for col in ["close", "adj_close", "price", "nav"] if col in df.columns), None)
        if price_col is None:
            raise ValueError(f"{path} needs one of equity, return, net_return, close, adj_close, price, nav")
        price = df[price_col].astype(float)
        equity = price / float(price.iloc[0])
        returns = equity.pct_change().fillna(0.0)

    curve = pd.DataFrame(
        {
            "trade_date": df["trade_date"].astype(str),
            "gross_return": returns.astype(float),
            "transaction_cost": 0.0,
            "net_return": returns.astype(float),
            "turnover": 0.0,
            "equity": equity.astype(float),
            "n_holdings": math.nan,
        }
    )
    return {
        "metrics": metrics_from_curve(curve, name=name, strategy="benchmark_index", trading_days_per_year=trading_days_per_year),
        "curve": curve,
        "trades": pd.DataFrame(),
        "holdings": pd.DataFrame(),
    }


def align_benchmark_to_dates(
    benchmark: dict[str, Any],
    dates: pd.Series | list[str],
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    date_set = {str(d) for d in dates}
    curve = benchmark["curve"].copy()
    curve["trade_date"] = curve["trade_date"].astype(str)
    curve = curve[curve["trade_date"].isin(date_set)].sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    if curve.empty:
        raise ValueError("benchmark has no overlapping trade_date values with the prediction split")
    if "net_return" in curve.columns:
        curve["equity"] = (1.0 + curve["net_return"].astype(float).fillna(0.0)).cumprod()
    metrics = benchmark["metrics"]
    return {
        "metrics": metrics_from_curve(
            curve,
            name=str(metrics.get("name", "benchmark")),
            strategy=str(metrics.get("strategy", "benchmark_index")),
            trading_days_per_year=trading_days_per_year,
        ),
        "curve": curve,
        "trades": benchmark.get("trades", pd.DataFrame()),
        "holdings": benchmark.get("holdings", pd.DataFrame()),
    }
