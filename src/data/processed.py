from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from collections import deque

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

    fds = ds.dataset(str(f_path), format="parquet")
    lds = ds.dataset(str(l_path), format="parquet")
    uds = ds.dataset(str(u_path), format="parquet") if (filter_in_universe and u_path.exists()) else None
    u_flag = cfg.universe_flag_col if uds is not None else None

    date_range_filter = (ds.field(key_trade) >= split.start_date) & (ds.field(key_trade) <= split.end_date)

    tqdm_mod = _get_tqdm(use_tqdm)

    def _iter_unique_trade_dates() -> list[str]:
        dates: set[str] = set()
        scan = fds.scanner(columns=[key_trade], filter=date_range_filter, batch_size=1 << 20)
        for b in scan.to_reader():
            arr = b.column(0)
            dates.update([str(x) for x in arr.to_pylist()])
        return sorted(dates)

    trade_dates = _iter_unique_trade_dates()
    it_dates = trade_dates
    if tqdm_mod is not None:
        it_dates = tqdm_mod(trade_dates, desc=f"{stage_desc}:{split.name}:dates", total=len(trade_dates))

    for d in it_dates:
        day_filter = (ds.field(key_trade) == d) & date_range_filter
        ftab = fds.to_table(columns=f_cols, filter=day_filter)
        ltab = lds.to_table(columns=l_cols, filter=day_filter)
        if ftab.num_rows == 0 or ltab.num_rows == 0:
            continue

        fdf = ftab.to_pandas()
        ldf = ltab.to_pandas()
        m = fdf.merge(ldf[[key_trade, key_code, label_col]], on=[key_trade, key_code], how="inner")
        if m.empty:
            continue

        if uds is not None and u_flag is not None:
            utab = uds.to_table(columns=[key_trade, key_code, u_flag], filter=day_filter)
            if utab.num_rows:
                udf = utab.to_pandas()
                m = m.merge(udf, on=[key_trade, key_code], how="left")
                m = m[m[u_flag].fillna(False)].drop(columns=[u_flag])
                if m.empty:
                    continue

        X_all = m[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y_all = m[label_col].to_numpy(dtype=np.float32, copy=False)

        if return_keys:
            trade_all = m[key_trade].astype(str).to_numpy(copy=False)
            code_all = m[key_code].astype(str).to_numpy(copy=False)

        n = int(len(m))
        for i in range(0, n, int(batch_size)):
            j = min(n, i + int(batch_size))
            out: dict[str, object] = {"X": X_all[i:j], "y": y_all[i:j]}
            if return_keys:
                out["trade_date"] = trade_all[i:j]
                out["ts_code"] = code_all[i:j]
            yield out


def iter_processed_sequence_batches(
    cfg: ProcessedConfig,
    split: ProcessedSplit,
    feature_cols: list[str],
    label_col: str,
    seq_len: int,
    batch_size: int,
    filter_in_universe: bool,
    return_keys: bool,
    use_tqdm: bool = False,
    stage_desc: str = "seq_scan",
) -> Iterator[dict[str, object]]:
    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise ImportError("pyarrow is required to read data/processed/*.parquet") from e

    seq_len = int(seq_len)
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")

    proc = Path(cfg.processed_dir)
    f_path = proc / cfg.features_path
    l_path = proc / cfg.labels_path
    u_path = proc / cfg.universe_path

    key_trade, key_code = cfg.key_cols
    f_cols = [key_trade, key_code] + list(feature_cols)
    l_cols = [key_trade, key_code, label_col]
    u_flag = cfg.universe_flag_col

    fds = ds.dataset(str(f_path), format="parquet")
    lds = ds.dataset(str(l_path), format="parquet")
    uds = ds.dataset(str(u_path), format="parquet") if (filter_in_universe and u_path.exists()) else None

    date_range_filter = (ds.field(key_trade) >= split.start_date) & (ds.field(key_trade) <= split.end_date)

    tqdm_mod = _get_tqdm(use_tqdm)

    def _iter_unique_trade_dates() -> list[str]:
        dates: set[str] = set()
        scan = fds.scanner(columns=[key_trade], filter=date_range_filter, batch_size=1 << 20)
        for b in scan.to_reader():
            dates.update([str(x) for x in b.column(0).to_pylist()])
        return sorted(dates)

    trade_dates = _iter_unique_trade_dates()
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}

    it_dates = trade_dates
    if tqdm_mod is not None:
        it_dates = tqdm_mod(trade_dates, desc=f"{stage_desc}:{split.name}:dates", total=len(trade_dates))

    buffers: dict[str, deque[np.ndarray]] = {}
    last_idx: dict[str, int] = {}

    X_buf: list[np.ndarray] = []
    y_buf: list[np.ndarray] = []
    d_buf: list[np.ndarray] = []
    c_buf: list[np.ndarray] = []

    def _flush(force: bool = False):
        nonlocal X_buf, y_buf, d_buf, c_buf
        while len(X_buf) >= int(batch_size) or (force and len(X_buf) > 0):
            n = int(batch_size) if len(X_buf) >= int(batch_size) else len(X_buf)
            X = np.stack(X_buf[:n], axis=0).astype(np.float32, copy=False)
            y = np.asarray(y_buf[:n], dtype=np.float32)
            out: dict[str, object] = {"X": X, "y": y}
            if return_keys:
                out["trade_date"] = np.asarray(d_buf[:n]).astype(str)
                out["ts_code"] = np.asarray(c_buf[:n]).astype(str)
            yield out
            X_buf = X_buf[n:]
            y_buf = y_buf[n:]
            d_buf = d_buf[n:]
            c_buf = c_buf[n:]

    for d in it_dates:
        day_filter = (ds.field(key_trade) == d) & date_range_filter
        ftab = fds.to_table(columns=f_cols, filter=day_filter)
        ltab = lds.to_table(columns=l_cols, filter=day_filter)
        if ftab.num_rows == 0 or ltab.num_rows == 0:
            continue

        fdf = ftab.to_pandas()
        ldf = ltab.to_pandas()
        m = fdf.merge(ldf[[key_trade, key_code, label_col]], on=[key_trade, key_code], how="inner")
        if m.empty:
            continue

        if uds is not None:
            utab = uds.to_table(columns=[key_trade, key_code, u_flag], filter=day_filter)
            if utab.num_rows:
                udf = utab.to_pandas()
                m = m.merge(udf, on=[key_trade, key_code], how="left")
                m = m[m[u_flag].fillna(False)].drop(columns=[u_flag])
                if m.empty:
                    continue

        m = m.sort_values(key_code, kind="mergesort")
        idx = int(date_to_idx[str(d)])
        X_day = m[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y_day = m[label_col].to_numpy(dtype=np.float32, copy=False)
        codes = m[key_code].astype(str).to_numpy(copy=False)

        for row_i, code in enumerate(codes):
            prev = last_idx.get(code)
            if prev is None or idx - prev != 1:
                buffers[code] = deque(maxlen=seq_len)
            buf = buffers[code]
            buf.append(X_day[row_i])
            last_idx[code] = idx
            if len(buf) == seq_len:
                X_buf.append(np.stack(buf, axis=0))
                y_buf.append(np.asarray(y_day[row_i], dtype=np.float32))
                if return_keys:
                    d_buf.append(np.asarray(d))
                    c_buf.append(np.asarray(code))

            for out in _flush(force=False):
                yield out

    for out in _flush(force=True):
        yield out


def iter_processed_sequence_feature_batches(
    cfg: ProcessedConfig,
    start_date: str,
    end_date: str,
    feature_cols: list[str],
    seq_len: int,
    batch_size: int,
    filter_in_universe: bool,
    return_keys: bool,
    use_tqdm: bool = False,
    stage_desc: str = "seq_pred",
    emit_start_date: str | None = None,
) -> Iterator[dict[str, object]]:
    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise ImportError("pyarrow is required to read data/processed/*.parquet") from e

    seq_len = int(seq_len)
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")

    emit_start_date = str(emit_start_date) if emit_start_date is not None else str(start_date)

    proc = Path(cfg.processed_dir)
    f_path = proc / cfg.features_path
    u_path = proc / cfg.universe_path

    key_trade, key_code = cfg.key_cols
    f_cols = [key_trade, key_code] + list(feature_cols)
    u_flag = cfg.universe_flag_col

    fds = ds.dataset(str(f_path), format="parquet")
    uds = ds.dataset(str(u_path), format="parquet") if (filter_in_universe and u_path.exists()) else None

    date_filter = (ds.field(key_trade) >= str(start_date)) & (ds.field(key_trade) <= str(end_date))

    tqdm_mod = _get_tqdm(use_tqdm)

    def _iter_unique_trade_dates() -> list[str]:
        dates: set[str] = set()
        scan = fds.scanner(columns=[key_trade], filter=date_filter, batch_size=1 << 20)
        for b in scan.to_reader():
            dates.update([str(x) for x in b.column(0).to_pylist()])
        return sorted(dates)

    trade_dates = _iter_unique_trade_dates()
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}
    emit_idx = date_to_idx.get(emit_start_date)
    if emit_idx is None:
        emit_idx = 0

    it_dates = trade_dates
    if tqdm_mod is not None:
        it_dates = tqdm_mod(trade_dates, desc=f"{stage_desc}:dates", total=len(trade_dates))

    buffers: dict[str, deque[np.ndarray]] = {}
    last_idx: dict[str, int] = {}

    X_buf: list[np.ndarray] = []
    d_buf: list[np.ndarray] = []
    c_buf: list[np.ndarray] = []

    def _flush(force: bool = False):
        nonlocal X_buf, d_buf, c_buf
        while len(X_buf) >= int(batch_size) or (force and len(X_buf) > 0):
            n = int(batch_size) if len(X_buf) >= int(batch_size) else len(X_buf)
            X = np.stack(X_buf[:n], axis=0).astype(np.float32, copy=False)
            out: dict[str, object] = {"X": X}
            if return_keys:
                out["trade_date"] = np.asarray(d_buf[:n]).astype(str)
                out["ts_code"] = np.asarray(c_buf[:n]).astype(str)
            yield out
            X_buf = X_buf[n:]
            d_buf = d_buf[n:]
            c_buf = c_buf[n:]

    for d in it_dates:
        idx = int(date_to_idx[str(d)])
        day_filter = (ds.field(key_trade) == d) & date_filter
        ftab = fds.to_table(columns=f_cols, filter=day_filter)
        if ftab.num_rows == 0:
            continue
        m = ftab.to_pandas()

        if uds is not None:
            utab = uds.to_table(columns=[key_trade, key_code, u_flag], filter=day_filter)
            if utab.num_rows:
                udf = utab.to_pandas()
                m = m.merge(udf, on=[key_trade, key_code], how="left")
                m = m[m[u_flag].fillna(False)].drop(columns=[u_flag])
                if m.empty:
                    continue

        m = m.sort_values(key_code, kind="mergesort")
        X_day = m[feature_cols].to_numpy(dtype=np.float32, copy=False)
        codes = m[key_code].astype(str).to_numpy(copy=False)

        for row_i, code in enumerate(codes):
            prev = last_idx.get(code)
            if prev is None or idx - prev != 1:
                buffers[code] = deque(maxlen=seq_len)
            buf = buffers[code]
            buf.append(X_day[row_i])
            last_idx[code] = idx
            if idx >= emit_idx and len(buf) == seq_len:
                X_buf.append(np.stack(buf, axis=0))
                if return_keys:
                    d_buf.append(np.asarray(d))
                    c_buf.append(np.asarray(code))
            for out in _flush(force=False):
                yield out

    for out in _flush(force=True):
        yield out
