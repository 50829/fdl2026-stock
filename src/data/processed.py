from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np


@dataclass(frozen=True)
class ProcessedSplit:
    name: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class ProcessedConfig:
    processed_dir: str
    features_path: str = "features.parquet"
    labels_path: str = "labels.parquet"
    universe_path: str = "universe.parquet"
    feature_meta_path: str = "feature_meta.json"
    splits_path: str = "splits.json"
    universe_flag_col: str = "in_universe"
    key_cols: tuple[str, str] = ("trade_date", "ts_code")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_columns(cfg: ProcessedConfig) -> list[str]:
    meta_path = Path(cfg.processed_dir) / cfg.feature_meta_path
    meta = _read_json(meta_path)
    cols = meta.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError(f"Invalid feature_columns in {meta_path}")
    return [str(c) for c in cols]


def build_processed_splits(cfg: ProcessedConfig, fallback: dict | None = None) -> dict[str, ProcessedSplit]:
    splits_path = Path(cfg.processed_dir) / cfg.splits_path
    if splits_path.exists():
        splits = _read_json(splits_path)
        out = {}
        for k in ("train", "valid", "test"):
            if k in splits and isinstance(splits[k], list) and len(splits[k]) == 2:
                out[k] = ProcessedSplit(name=k, start_date=str(splits[k][0]), end_date=str(splits[k][1]))
        if out:
            return out

    if fallback is None:
        raise FileNotFoundError(f"Missing {splits_path} and no fallback splits provided")
    sample = fallback.get("sample", {})
    start = str(sample.get("start_date"))
    train_end = str(sample.get("train_end"))
    valid_end = str(sample.get("valid_end"))
    if not (start and train_end and valid_end):
        raise ValueError("Fallback splits require sample.start_date/train_end/valid_end")
    return {
        "train": ProcessedSplit(name="train", start_date=start, end_date=train_end),
        "valid": ProcessedSplit(name="valid", start_date=_next_day(train_end), end_date=valid_end),
    }


def _next_day(date_str: str) -> str:
    s = str(date_str)
    if len(s) != 8:
        return s
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    import datetime as _dt

    dt = _dt.date(y, m, d) + _dt.timedelta(days=1)
    return dt.strftime("%Y%m%d")


def _get_tqdm(enabled: bool):
    if not enabled:
        return None
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def iter_processed_batches(
    cfg: ProcessedConfig,
    split: ProcessedSplit,
    feature_cols: list[str],
    label_col: str,
    batch_size: int,
    filter_in_universe: bool,
    return_keys: bool,
    use_tqdm: bool = False,
    stage_desc: str = "scan",
) -> Iterator[dict[str, object]]:
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
    except Exception as e:
        raise ImportError("pyarrow is required to read data/processed/*.parquet") from e

    proc = Path(cfg.processed_dir)
    f_path = proc / cfg.features_path
    l_path = proc / cfg.labels_path
    u_path = proc / cfg.universe_path

    key_trade, key_code = cfg.key_cols
    f_cols = [key_trade, key_code] + list(feature_cols)
    l_cols = [key_trade, key_code, label_col]

    date_filter = (ds.field(key_trade) >= split.start_date) & (ds.field(key_trade) <= split.end_date)
    fds = ds.dataset(str(f_path), format="parquet")
    lds = ds.dataset(str(l_path), format="parquet")
    fscan = fds.scanner(columns=f_cols, filter=date_filter, batch_size=int(batch_size))
    lscan = lds.scanner(columns=l_cols, filter=date_filter, batch_size=int(batch_size))
    fr = fscan.to_reader()
    lr = lscan.to_reader()

    ur = None
    u_flag = None
    if filter_in_universe and u_path.exists():
        uds = ds.dataset(str(u_path), format="parquet")
        u_flag = cfg.universe_flag_col
        u_cols = [key_trade, key_code, u_flag]
        uscan = uds.scanner(columns=u_cols, filter=date_filter, batch_size=int(batch_size))
        ur = uscan.to_reader()

    tqdm_mod = _get_tqdm(use_tqdm)

    def _col_as_np(batch, name: str) -> np.ndarray:
        arr = batch.column(batch.schema.get_field_index(name))
        out = arr.to_numpy(zero_copy_only=False)
        return out

    it = zip(fr, lr) if ur is None else zip(fr, lr, ur)
    if tqdm_mod is not None:
        it = tqdm_mod(it, desc=f"{stage_desc}:{split.name}")

    for packs in it:
        if ur is None:
            fb, lb = packs
            ub = None
        else:
            fb, lb, ub = packs

        f_trade = _col_as_np(fb, key_trade)
        l_trade = _col_as_np(lb, key_trade)
        f_code = _col_as_np(fb, key_code)
        l_code = _col_as_np(lb, key_code)
        if len(f_trade) != len(l_trade) or len(f_code) != len(l_code) or not np.array_equal(f_trade, l_trade) or not np.array_equal(f_code, l_code):
            raise ValueError("features.parquet and labels.parquet are not aligned on (trade_date, ts_code)")

        mask = None
        if ub is not None and u_flag is not None:
            u_trade = _col_as_np(ub, key_trade)
            u_code = _col_as_np(ub, key_code)
            if len(u_trade) != len(f_trade) or not np.array_equal(u_trade, f_trade) or not np.array_equal(u_code, f_code):
                raise ValueError("universe.parquet is not aligned with features/labels")
            flag = _col_as_np(ub, u_flag)
            if flag.dtype != np.bool_:
                flag = flag.astype(bool)
            mask = flag

        cols_np = []
        for c in feature_cols:
            x = _col_as_np(fb, c)
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            cols_np.append(x)
        X = np.stack(cols_np, axis=1)

        y = _col_as_np(lb, label_col)
        if y.dtype != np.float32:
            y = y.astype(np.float32)

        if mask is not None:
            X = X[mask]
            y = y[mask]
            f_trade = f_trade[mask]
            f_code = f_code[mask]

        out: dict[str, object] = {"X": X, "y": y}
        if return_keys:
            out["trade_date"] = f_trade.astype(str)
            out["ts_code"] = f_code.astype(str)
        yield out

