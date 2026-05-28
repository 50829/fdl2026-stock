from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def normalize_date_str(date_str: str) -> str:
    s = str(date_str).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s.replace("-", "")
    if len(s) == 8 and s.isdigit():
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 8:
        return digits
    raise ValueError(f"Unsupported date format: {date_str}")


def split_by_date(df: pd.DataFrame, train_end: str, val_end: str, date_col: str = "trade_date"):
    train_end = normalize_date_str(train_end)
    val_end = normalize_date_str(val_end)
    tr = df[df[date_col] <= train_end].copy()
    va = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    return tr, va


def _list_date_files(daily_dir: Path, start_date: str, end_date: str) -> list[Path]:
    start_date = normalize_date_str(start_date)
    end_date = normalize_date_str(end_date)
    files = []
    for p in daily_dir.glob("*.csv"):
        stem = p.stem
        if len(stem) == 8 and stem.isdigit() and start_date <= stem <= end_date:
            files.append(p)
    files.sort(key=lambda x: x.stem)
    return files


def load_daily_dir(
    daily_dir: str,
    start_date: str,
    end_date: str,
    usecols: Optional[list[str]] = None,
    limit_files: Optional[int] = None,
    use_tqdm: bool = False,
    tqdm_desc: str = "load_daily",
) -> pd.DataFrame:
    daily_dir_p = Path(daily_dir)
    files = _list_date_files(daily_dir_p, start_date, end_date)
    if limit_files is not None:
        files = files[: int(limit_files)]
    iterable = files
    if use_tqdm:
        try:
            from tqdm import tqdm  # type: ignore

            iterable = tqdm(files, desc=tqdm_desc, total=len(files))
        except Exception:
            iterable = files

    dfs = []
    for p in iterable:
        df = pd.read_csv(p, usecols=usecols, dtype={"ts_code": str, "trade_date": str})
        df["trade_date"] = df["trade_date"].map(normalize_date_str)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No daily files found in {daily_dir} for [{start_date}, {end_date}]")
    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["ts_code", "trade_date"], kind="mergesort")
    return out


def load_panel_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"ts_code": str, "trade_date": str})
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].map(normalize_date_str)
    df = df.sort_values(["ts_code", "trade_date"], kind="mergesort")
    return df


def load_basic_info(basic_csv: str) -> pd.DataFrame:
    df = pd.read_csv(basic_csv, dtype={"ts_code": str})
    return df


def load_st_codes_for_range(stock_st_dir: str, start_date: str, end_date: str) -> dict[str, set[str]]:
    st_dir = Path(stock_st_dir)
    start_date = normalize_date_str(start_date)
    end_date = normalize_date_str(end_date)
    out: dict[str, set[str]] = {}
    for p in st_dir.glob("*.csv"):
        stem = p.stem
        if len(stem) != 8 or not stem.isdigit():
            continue
        if not (start_date <= stem <= end_date):
            continue
        df = pd.read_csv(p, dtype={"ts_code": str, "trade_date": str})
        if "trade_date" in df.columns:
            d = normalize_date_str(df["trade_date"].iloc[0])
        else:
            d = stem
        out[d] = set(df["ts_code"].astype(str).tolist())
    return out


def filter_stock_pool(
    df: pd.DataFrame,
    basic_csv: Optional[str] = None,
    stock_st_dir: Optional[str] = None,
    exclude_bj: bool = True,
    exclude_st: bool = True,
) -> pd.DataFrame:
    out = df
    if basic_csv is not None and exclude_bj:
        basic = load_basic_info(basic_csv)
        bj = set(basic.loc[basic.get("market") == "北交所", "ts_code"].astype(str).tolist())
        if bj:
            out = out[~out["ts_code"].isin(bj)].copy()
    if stock_st_dir is not None and exclude_st:
        start_date = out["trade_date"].min()
        end_date = out["trade_date"].max()
        st_map = load_st_codes_for_range(stock_st_dir, start_date, end_date)
        if st_map:
            kept = []
            for d, g in out.groupby("trade_date", sort=False):
                st_set = st_map.get(d)
                if not st_set:
                    kept.append(g)
                else:
                    kept.append(g[~g["ts_code"].isin(st_set)])
            out = pd.concat(kept, ignore_index=True)
    out = out.sort_values(["ts_code", "trade_date"], kind="mergesort")
    return out


def merge_on_keys(base: pd.DataFrame, extra: pd.DataFrame, keys: Iterable[str] = ("ts_code", "trade_date")) -> pd.DataFrame:
    keys = list(keys)
    extra_cols = [c for c in extra.columns if c not in keys]
    if not extra_cols:
        return base
    return base.merge(extra[keys + extra_cols], on=keys, how="left")
