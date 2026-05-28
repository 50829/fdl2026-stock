"""Utilities for loading raw A-share CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_date_files(directory: Path, start_date: str | None = None, end_date: str | None = None) -> list[Path]:
    """Return YYYYMMDD csv files within an optional inclusive date range."""
    files = []
    for path in sorted(directory.glob("*.csv")):
        date = path.stem
        if not date.isdigit():
            continue
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        files.append(path)
    return files


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and normalize date/code columns used by the project."""
    df = pd.read_csv(path)
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str)
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str)
    return df


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory use while keeping identifiers untouched."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        if col not in {"trade_date"}:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df
