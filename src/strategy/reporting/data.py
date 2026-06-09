from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .html import write_html_report


def split_curve_name(name: str) -> tuple[str, str]:
    if "__" not in name:
        return "benchmark" if is_benchmark_name(name) else "", name
    model, variant = name.split("__", 1)
    return model, variant


def is_benchmark_name(name: str) -> bool:
    return str(name).startswith("benchmark") or "000300" in str(name)


def normalize_metrics(rows_by_split: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split, rows in rows_by_split.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).copy()
        if "split" not in df:
            df["split"] = split
        else:
            df["split"] = df["split"].fillna(split).astype(str)
        df["name"] = df["name"].astype(str)
        df["is_benchmark"] = df["name"].map(is_benchmark_name)
        split_names = df["name"].map(split_curve_name)
        df["model"] = [model for model, _ in split_names]
        df["variant"] = [variant for _, variant in split_names]
        if "strategy" not in df:
            df["strategy"] = df["variant"]
        df.loc[df["is_benchmark"], "model"] = "benchmark"
        df.loc[df["is_benchmark"], "variant"] = df.loc[df["is_benchmark"], "name"]
        df["strategy_family"] = df["strategy"].astype(str)
        df.loc[df["is_benchmark"], "strategy_family"] = "benchmark"
        df["display_name"] = df.apply(
            lambda row: str(row["variant"]) if bool(row["is_benchmark"]) else f"{row['model']} / {row['variant']}",
            axis=1,
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_equity_curves(curves_by_split: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split, curves in curves_by_split.items():
        for name, curve in curves.items():
            if curve.empty:
                continue
            model, variant = split_curve_name(str(name))
            is_benchmark = is_benchmark_name(str(name))
            out = curve.copy()
            out["split"] = split
            out["name"] = str(name)
            out["model"] = "benchmark" if is_benchmark else model
            out["variant"] = str(name) if is_benchmark else variant
            out["is_benchmark"] = is_benchmark
            out["trade_date"] = out["trade_date"].astype(str)
            out["equity"] = out["equity"].astype(float)
            out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
            keep = [
                "split",
                "trade_date",
                "name",
                "model",
                "variant",
                "is_benchmark",
                "equity",
                "drawdown",
                "net_return",
                "gross_return",
                "transaction_cost",
                "fee_cost",
                "slippage_cost",
                "total_cost",
                "turnover",
                "n_holdings",
                "raw_gross_exposure",
                "gross_exposure",
                "exposure_limit",
                "market_exposure_limit",
                "drawdown_exposure_limit",
                "market_stressed",
                "market_stress_return",
                "portfolio_drawdown_pre",
            ]
            frames.append(out[[col for col in keep if col in out.columns]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_standard_tables(
    out_root: str | Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    curves_by_split: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, str]:
    root = Path(out_root)
    metrics = normalize_metrics(rows_by_split)
    equity = normalize_equity_curves(curves_by_split)
    metrics_path = root / "metrics_long.csv"
    equity_path = root / "equity_long.parquet"
    metrics.to_csv(metrics_path, index=False)
    equity.to_parquet(equity_path, index=False)
    return {"metrics_long": str(metrics_path), "equity_long": str(equity_path)}


def load_existing_aggregate_outputs(out_root: str | Path, splits: list[str] | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, pd.DataFrame]]]:
    root = Path(out_root)
    chosen_splits = splits or [path.name for path in sorted(root.iterdir()) if path.is_dir() and (path / "strategy_metrics.csv").exists()]
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    curves_by_split: dict[str, dict[str, pd.DataFrame]] = {}
    for split in chosen_splits:
        metrics_path = root / split / "strategy_metrics.csv"
        if not metrics_path.exists():
            continue
        rows = pd.read_csv(metrics_path).to_dict("records")
        rows_by_split[split] = rows
        split_curves: dict[str, pd.DataFrame] = {}
        for row in rows:
            name = str(row["name"])
            if is_benchmark_name(name):
                matches = sorted(root.glob(f"*/{split}/{name}/equity_curve.csv"))
                if matches:
                    split_curves[name] = pd.read_csv(matches[0])
                continue
            model, variant = split_curve_name(name)
            curve_path = root / model / split / variant / "equity_curve.csv"
            if curve_path.exists():
                split_curves[name] = pd.read_csv(curve_path)
        curves_by_split[split] = split_curves
    return rows_by_split, curves_by_split


def write_report_artifacts(
    out_root: str | Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    curves_by_split: dict[str, dict[str, pd.DataFrame]],
    *,
    benchmark_note: str = "",
    title: str | None = None,
) -> dict[str, str]:
    paths = write_standard_tables(out_root, rows_by_split, curves_by_split)
    html_path = write_html_report(out_root, metrics_path=paths["metrics_long"], benchmark_note=benchmark_note, title=title)
    paths["report_html"] = str(html_path)
    return paths
