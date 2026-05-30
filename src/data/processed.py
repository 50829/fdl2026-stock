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


def _date_range_filter(ds_mod, key_trade: str, start_date: str, end_date: str):
    return (ds_mod.field(key_trade) >= str(start_date)) & (ds_mod.field(key_trade) <= str(end_date))


def _load_cached_labeled_frame(
    cfg: ProcessedConfig,
    split: ProcessedSplit,
    feature_cols: list[str],
    label_col: str,
    filter_in_universe: bool,
):
    try:
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
    date_filter = _date_range_filter(ds, key_trade, split.start_date, split.end_date)

    fdf = ds.dataset(str(f_path), format="parquet").to_table(columns=f_cols, filter=date_filter).to_pandas()
    ldf = ds.dataset(str(l_path), format="parquet").to_table(columns=l_cols, filter=date_filter).to_pandas()
    m = fdf.merge(ldf[[key_trade, key_code, label_col]], on=[key_trade, key_code], how="inner")

    if filter_in_universe and u_path.exists() and not m.empty:
        u_flag = cfg.universe_flag_col
        udf = (
            ds.dataset(str(u_path), format="parquet")
            .to_table(columns=[key_trade, key_code, u_flag], filter=date_filter)
            .to_pandas()
        )
        m = m.merge(udf, on=[key_trade, key_code], how="left")
        m = m[m[u_flag].fillna(False)].drop(columns=[u_flag])

    if m.empty:
        return m
    return m.sort_values([key_trade, key_code], kind="mergesort").reset_index(drop=True)


def _load_cached_feature_frame(
    cfg: ProcessedConfig,
    start_date: str,
    end_date: str,
    feature_cols: list[str],
    filter_in_universe: bool,
):
    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise ImportError("pyarrow is required to read data/processed/*.parquet") from e

    proc = Path(cfg.processed_dir)
    f_path = proc / cfg.features_path
    u_path = proc / cfg.universe_path

    key_trade, key_code = cfg.key_cols
    f_cols = [key_trade, key_code] + list(feature_cols)
    date_filter = _date_range_filter(ds, key_trade, start_date, end_date)

    m = ds.dataset(str(f_path), format="parquet").to_table(columns=f_cols, filter=date_filter).to_pandas()
    if filter_in_universe and u_path.exists() and not m.empty:
        u_flag = cfg.universe_flag_col
        udf = (
            ds.dataset(str(u_path), format="parquet")
            .to_table(columns=[key_trade, key_code, u_flag], filter=date_filter)
            .to_pandas()
        )
        m = m.merge(udf, on=[key_trade, key_code], how="left")
        m = m[m[u_flag].fillna(False)].drop(columns=[u_flag])

    if m.empty:
        return m
    return m.sort_values([key_trade, key_code], kind="mergesort").reset_index(drop=True)


def _iter_cached_batches(
    cfg: ProcessedConfig,
    split: ProcessedSplit,
    feature_cols: list[str],
    label_col: str,
    batch_size: int,
    filter_in_universe: bool,
    return_keys: bool,
) -> Iterator[dict[str, object]]:
    key_trade, key_code = cfg.key_cols
    m = _load_cached_labeled_frame(cfg, split, feature_cols, label_col, filter_in_universe)
    if m.empty:
        return

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


def _iter_cached_sequence_batches(
    cfg: ProcessedConfig,
    split: ProcessedSplit,
    feature_cols: list[str],
    label_col: str,
    seq_len: int,
    batch_size: int,
    filter_in_universe: bool,
    return_keys: bool,
) -> Iterator[dict[str, object]]:
    key_trade, key_code = cfg.key_cols
    m = _load_cached_labeled_frame(cfg, split, feature_cols, label_col, filter_in_universe)
    if m.empty:
        return

    trade_dates = sorted(m[key_trade].astype(str).unique().tolist())
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}

    X_buf: list[np.ndarray] = []
    y_buf: list[np.ndarray] = []
    d_buf: list[np.ndarray] = []
    c_buf: list[np.ndarray] = []
    buf_n = 0

    def _flush(force: bool = False):
        nonlocal X_buf, y_buf, d_buf, c_buf, buf_n
        while buf_n >= int(batch_size) or (force and buf_n > 0):
            need = int(batch_size) if buf_n >= int(batch_size) else buf_n
            out_x: list[np.ndarray] = []
            out_y: list[np.ndarray] = []
            out_d: list[np.ndarray] = []
            out_c: list[np.ndarray] = []
            take_left = need
            while take_left > 0:
                take = min(take_left, int(len(y_buf[0])))
                out_x.append(X_buf[0][:take])
                out_y.append(y_buf[0][:take])
                if return_keys:
                    out_d.append(d_buf[0][:take])
                    out_c.append(c_buf[0][:take])
                if take == int(len(y_buf[0])):
                    X_buf.pop(0)
                    y_buf.pop(0)
                    if return_keys:
                        d_buf.pop(0)
                        c_buf.pop(0)
                else:
                    X_buf[0] = X_buf[0][take:]
                    y_buf[0] = y_buf[0][take:]
                    if return_keys:
                        d_buf[0] = d_buf[0][take:]
                        c_buf[0] = c_buf[0][take:]
                take_left -= take
            buf_n -= need
            X = np.concatenate(out_x, axis=0).astype(np.float32, copy=False)
            y = np.concatenate(out_y, axis=0).astype(np.float32, copy=False)
            out: dict[str, object] = {"X": X, "y": y}
            if return_keys:
                out["trade_date"] = np.concatenate(out_d, axis=0).astype(str)
                out["ts_code"] = np.concatenate(out_c, axis=0).astype(str)
            yield out

    def _run_bounds(idx: np.ndarray):
        breaks = np.flatnonzero(np.diff(idx) != 1) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(idx)]))
        return zip(starts, ends)

    for code, stock in m.groupby(key_code, sort=True):
        stock = stock.sort_values(key_trade, kind="mergesort")
        idx = stock[key_trade].astype(str).map(date_to_idx).to_numpy(dtype=np.int32, copy=False)
        X_stock = stock[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y_stock = stock[label_col].to_numpy(dtype=np.float32, copy=False)
        d_stock = stock[key_trade].astype(str).to_numpy(copy=False)

        for start, end in _run_bounds(idx):
            if end - start < seq_len:
                continue
            X_run = X_stock[start:end]
            windows = np.lib.stride_tricks.sliding_window_view(X_run, seq_len, axis=0).transpose(0, 2, 1)
            y_win = y_stock[start + seq_len - 1 : end]
            X_buf.append(windows)
            y_buf.append(y_win)
            if return_keys:
                n_win = int(len(y_win))
                d_buf.append(d_stock[start + seq_len - 1 : end])
                c_buf.append(np.full(n_win, str(code), dtype=object))
            buf_n += int(len(y_win))
            for out in _flush(force=False):
                yield out

    for out in _flush(force=True):
        yield out


def _iter_cached_sequence_feature_batches(
    cfg: ProcessedConfig,
    start_date: str,
    end_date: str,
    feature_cols: list[str],
    seq_len: int,
    batch_size: int,
    filter_in_universe: bool,
    return_keys: bool,
    emit_start_date: str | None,
) -> Iterator[dict[str, object]]:
    key_trade, key_code = cfg.key_cols
    emit_start_date = str(emit_start_date) if emit_start_date is not None else str(start_date)
    m = _load_cached_feature_frame(cfg, start_date, end_date, feature_cols, False)
    if m.empty:
        return

    eligible_keys: set[tuple[str, str]] | None = None
    if filter_in_universe:
        try:
            import pyarrow.dataset as ds
        except Exception as e:
            raise ImportError("pyarrow is required to read data/processed/*.parquet") from e
        proc = Path(cfg.processed_dir)
        u_path = proc / cfg.universe_path
        if u_path.exists():
            u_flag = cfg.universe_flag_col
            emit_filter = _date_range_filter(ds, key_trade, emit_start_date, end_date)
            universe = (
                ds.dataset(str(u_path), format="parquet")
                .to_table(columns=[key_trade, key_code, u_flag], filter=emit_filter)
                .to_pandas()
            )
            if not universe.empty:
                universe[key_trade] = universe[key_trade].astype(str)
                universe[key_code] = universe[key_code].astype(str)
                universe = universe[universe[u_flag].fillna(False)]
                eligible_keys = set(zip(universe[key_trade], universe[key_code]))

    trade_dates = sorted(m[key_trade].astype(str).unique().tolist())
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}
    emit_idx = date_to_idx.get(emit_start_date, 0)

    X_buf: list[np.ndarray] = []
    d_buf: list[np.ndarray] = []
    c_buf: list[np.ndarray] = []
    buf_n = 0

    def _flush(force: bool = False):
        nonlocal X_buf, d_buf, c_buf, buf_n
        while buf_n >= int(batch_size) or (force and buf_n > 0):
            need = int(batch_size) if buf_n >= int(batch_size) else buf_n
            out_x: list[np.ndarray] = []
            out_d: list[np.ndarray] = []
            out_c: list[np.ndarray] = []
            take_left = need
            while take_left > 0:
                take = min(take_left, int(len(X_buf[0])))
                out_x.append(X_buf[0][:take])
                if return_keys:
                    out_d.append(d_buf[0][:take])
                    out_c.append(c_buf[0][:take])
                if take == int(len(X_buf[0])):
                    X_buf.pop(0)
                    if return_keys:
                        d_buf.pop(0)
                        c_buf.pop(0)
                else:
                    X_buf[0] = X_buf[0][take:]
                    if return_keys:
                        d_buf[0] = d_buf[0][take:]
                        c_buf[0] = c_buf[0][take:]
                take_left -= take
            buf_n -= need
            X = np.concatenate(out_x, axis=0).astype(np.float32, copy=False)
            out: dict[str, object] = {"X": X}
            if return_keys:
                out["trade_date"] = np.concatenate(out_d, axis=0).astype(str)
                out["ts_code"] = np.concatenate(out_c, axis=0).astype(str)
            yield out

    def _run_bounds(idx: np.ndarray):
        breaks = np.flatnonzero(np.diff(idx) != 1) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(idx)]))
        return zip(starts, ends)

    for code, stock in m.groupby(key_code, sort=True):
        stock = stock.sort_values(key_trade, kind="mergesort")
        idx = stock[key_trade].astype(str).map(date_to_idx).to_numpy(dtype=np.int32, copy=False)
        X_stock = stock[feature_cols].to_numpy(dtype=np.float32, copy=False)
        d_stock = stock[key_trade].astype(str).to_numpy(copy=False)

        for start, end in _run_bounds(idx):
            if end - start < seq_len:
                continue
            X_run = X_stock[start:end]
            windows = np.lib.stride_tricks.sliding_window_view(X_run, seq_len, axis=0).transpose(0, 2, 1)
            end_idx = idx[start + seq_len - 1 : end]
            keep = end_idx >= int(emit_idx)
            if eligible_keys is not None and np.any(keep):
                d_win = d_stock[start + seq_len - 1 : end]
                key_keep = np.fromiter(
                    ((str(d), str(code)) in eligible_keys for d in d_win),
                    dtype=bool,
                    count=len(d_win),
                )
                keep = keep & key_keep
            if not np.any(keep):
                continue
            windows = windows[keep]
            X_buf.append(windows)
            if return_keys:
                n_win = int(len(windows))
                d_buf.append(d_stock[start + seq_len - 1 : end][keep])
                c_buf.append(np.full(n_win, str(code), dtype=object))
            buf_n += int(len(windows))
            for out in _flush(force=False):
                yield out

    for out in _flush(force=True):
        yield out


def _iter_cached_sequence_labeled_feature_batches(
    cfg: ProcessedConfig,
    start_date: str,
    end_date: str,
    emit_start_date: str,
    feature_cols: list[str],
    label_col: str,
    seq_len: int,
    batch_size: int,
    filter_in_universe: bool,
    return_keys: bool,
) -> Iterator[dict[str, object]]:
    try:
        import pandas as pd
        import pyarrow.dataset as ds
    except Exception as e:
        raise ImportError("pandas and pyarrow are required to read data/processed/*.parquet") from e

    key_trade, key_code = cfg.key_cols
    proc = Path(cfg.processed_dir)
    f_path = proc / cfg.features_path
    l_path = proc / cfg.labels_path
    u_path = proc / cfg.universe_path
    u_flag = cfg.universe_flag_col

    start_date = str(start_date)
    end_date = str(end_date)
    emit_start_date = str(emit_start_date)
    f_cols = [key_trade, key_code] + list(feature_cols)
    feature_filter = _date_range_filter(ds, key_trade, start_date, end_date)
    emit_filter = _date_range_filter(ds, key_trade, emit_start_date, end_date)

    features = ds.dataset(str(f_path), format="parquet").to_table(columns=f_cols, filter=feature_filter).to_pandas()
    labels = (
        ds.dataset(str(l_path), format="parquet")
        .to_table(columns=[key_trade, key_code, label_col], filter=emit_filter)
        .to_pandas()
    )
    if features.empty or labels.empty:
        return

    features[key_trade] = features[key_trade].astype(str)
    features[key_code] = features[key_code].astype(str)
    labels[key_trade] = labels[key_trade].astype(str)
    labels[key_code] = labels[key_code].astype(str)
    labels = labels.dropna(subset=[label_col])

    if filter_in_universe and u_path.exists() and not labels.empty:
        universe = (
            ds.dataset(str(u_path), format="parquet")
            .to_table(columns=[key_trade, key_code, u_flag], filter=emit_filter)
            .to_pandas()
        )
        universe[key_trade] = universe[key_trade].astype(str)
        universe[key_code] = universe[key_code].astype(str)
        labels = labels.merge(universe, on=[key_trade, key_code], how="left")
        labels = labels[labels[u_flag].fillna(False)].drop(columns=[u_flag])

    if labels.empty:
        return

    label_map = labels.set_index([key_trade, key_code])[label_col]
    trade_dates = sorted(features[key_trade].unique().tolist())
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}
    emit_idx = date_to_idx.get(emit_start_date, 0)

    X_buf: list[np.ndarray] = []
    y_buf: list[np.ndarray] = []
    d_buf: list[np.ndarray] = []
    c_buf: list[np.ndarray] = []
    buf_n = 0

    def _flush(force: bool = False):
        nonlocal X_buf, y_buf, d_buf, c_buf, buf_n
        while buf_n >= int(batch_size) or (force and buf_n > 0):
            need = int(batch_size) if buf_n >= int(batch_size) else buf_n
            out_x: list[np.ndarray] = []
            out_y: list[np.ndarray] = []
            out_d: list[np.ndarray] = []
            out_c: list[np.ndarray] = []
            take_left = need
            while take_left > 0:
                take = min(take_left, int(len(y_buf[0])))
                out_x.append(X_buf[0][:take])
                out_y.append(y_buf[0][:take])
                if return_keys:
                    out_d.append(d_buf[0][:take])
                    out_c.append(c_buf[0][:take])
                if take == int(len(y_buf[0])):
                    X_buf.pop(0)
                    y_buf.pop(0)
                    if return_keys:
                        d_buf.pop(0)
                        c_buf.pop(0)
                else:
                    X_buf[0] = X_buf[0][take:]
                    y_buf[0] = y_buf[0][take:]
                    if return_keys:
                        d_buf[0] = d_buf[0][take:]
                        c_buf[0] = c_buf[0][take:]
                take_left -= take
            buf_n -= need
            out: dict[str, object] = {
                "X": np.concatenate(out_x, axis=0).astype(np.float32, copy=False),
                "y": np.concatenate(out_y, axis=0).astype(np.float32, copy=False),
            }
            if return_keys:
                out["trade_date"] = np.concatenate(out_d, axis=0).astype(str)
                out["ts_code"] = np.concatenate(out_c, axis=0).astype(str)
            yield out

    def _run_bounds(idx: np.ndarray):
        breaks = np.flatnonzero(np.diff(idx) != 1) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(idx)]))
        return zip(starts, ends)

    features = features.sort_values([key_code, key_trade], kind="mergesort")
    for code, stock in features.groupby(key_code, sort=True):
        stock = stock.sort_values(key_trade, kind="mergesort")
        idx = stock[key_trade].map(date_to_idx).to_numpy(dtype=np.int32, copy=False)
        X_stock = stock[feature_cols].to_numpy(dtype=np.float32, copy=False)
        d_stock = stock[key_trade].astype(str).to_numpy(copy=False)

        for start, end in _run_bounds(idx):
            if end - start < seq_len:
                continue
            X_run = X_stock[start:end]
            windows = np.lib.stride_tricks.sliding_window_view(X_run, seq_len, axis=0).transpose(0, 2, 1)
            d_win = d_stock[start + seq_len - 1 : end]
            end_idx = idx[start + seq_len - 1 : end]
            keep = end_idx >= int(emit_idx)
            if not np.any(keep):
                continue

            d_keep = d_win[keep]
            label_index = pd.MultiIndex.from_arrays(
                [d_keep.astype(str), np.full(len(d_keep), str(code), dtype=object)],
                names=[key_trade, key_code],
            )
            y_keep = label_map.reindex(label_index).to_numpy(dtype=np.float32, na_value=np.nan)
            label_keep = np.isfinite(y_keep)
            if not np.any(label_keep):
                continue

            X_part = windows[keep][label_keep]
            y_part = y_keep[label_keep]
            X_buf.append(X_part)
            y_buf.append(y_part)
            if return_keys:
                n_part = int(len(y_part))
                d_buf.append(d_keep[label_keep])
                c_buf.append(np.full(n_part, str(code), dtype=object))
            buf_n += int(len(y_part))
            for out in _flush(force=False):
                yield out

    for out in _flush(force=True):
        yield out


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
    cache_in_memory: bool = False,
) -> Iterator[dict[str, object]]:
    if cache_in_memory:
        yield from _iter_cached_batches(cfg, split, feature_cols, label_col, batch_size, filter_in_universe, return_keys)
        return

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
    cache_in_memory: bool = False,
) -> Iterator[dict[str, object]]:
    if cache_in_memory:
        yield from _iter_cached_sequence_batches(
            cfg,
            split,
            feature_cols,
            label_col,
            seq_len,
            batch_size,
            filter_in_universe,
            return_keys,
        )
        return

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
    cache_in_memory: bool = False,
) -> Iterator[dict[str, object]]:
    if cache_in_memory:
        yield from _iter_cached_sequence_feature_batches(
            cfg,
            start_date,
            end_date,
            feature_cols,
            seq_len,
            batch_size,
            filter_in_universe,
            return_keys,
            emit_start_date,
        )
        return

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


def iter_processed_sequence_labeled_feature_batches(
    cfg: ProcessedConfig,
    start_date: str,
    end_date: str,
    emit_start_date: str,
    feature_cols: list[str],
    label_col: str,
    seq_len: int,
    batch_size: int,
    filter_in_universe: bool,
    return_keys: bool,
    use_tqdm: bool = False,
    stage_desc: str = "seq_labeled",
    cache_in_memory: bool = False,
) -> Iterator[dict[str, object]]:
    del use_tqdm, stage_desc, cache_in_memory
    yield from _iter_cached_sequence_labeled_feature_batches(
        cfg,
        start_date,
        end_date,
        emit_start_date,
        feature_cols,
        label_col,
        seq_len,
        batch_size,
        filter_in_universe,
        return_keys,
    )
