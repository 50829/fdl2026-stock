"""Daily prediction entry point."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data import (
    ProcessedConfig,
    ProcessedSplit,
    build_processed_splits,
    iter_processed_batches,
    iter_processed_sequence_feature_batches,
)
from src.models import build_model


def _get_tqdm(enabled: bool):
    if not enabled:
        return None
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _pwrite(tqdm_mod, msg: str):
    if tqdm_mod is not None:
        tqdm_mod.write(msg)
    else:
        print(msg)


def _default_pred_path(start_date: str, end_date: str) -> str:
    if start_date == end_date:
        return f"outputs/predictions/pred_{start_date}.parquet"
    return f"outputs/predictions/pred_{start_date}_{end_date}.parquet"


def predict(cfg: dict):
    pred_cfg = cfg.get("predict", {})
    use_tqdm = bool(pred_cfg.get("use_tqdm", True))
    tqdm_mod = _get_tqdm(use_tqdm)

    data_cfg = cfg.get("data", {})
    processed_dir = str(data_cfg.get("processed_dir", "data/processed"))
    pcfg = ProcessedConfig(processed_dir=processed_dir)

    ckpt_path = Path(str(pred_cfg.get("ckpt", cfg.get("eval", {}).get("ckpt", "outputs/models/ckpt.pt"))))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    feature_cols = list(ckpt["feature_cols"])
    label_col = str(ckpt.get("label_col", "label_1d"))

    model = build_model(cfg, in_dim=int(len(feature_cols)))
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(str(pred_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model.to(device)
    model.eval()

    splits = build_processed_splits(pcfg, fallback=cfg)
    split_name = str(pred_cfg.get("split", "test"))
    if split_name in splits:
        s = splits[split_name]
        start_date = str(pred_cfg.get("start_date", s.start_date))
        end_date = str(pred_cfg.get("end_date", s.end_date))
    else:
        start_date = str(pred_cfg.get("start_date"))
        end_date = str(pred_cfg.get("end_date"))
        if not start_date or not end_date:
            raise ValueError("predict requires start_date/end_date or a valid split")

    filter_in_universe = bool(pred_cfg.get("filter_in_universe", True))
    cache_data = bool(pred_cfg.get("cache_data", cfg.get("train", {}).get("cache_data", False)))
    batch_size = int(pred_cfg.get("batch_size", 4096))
    out_path = str(pred_cfg.get("output_path", _default_pred_path(start_date, end_date)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        raise ImportError("pyarrow is required to write parquet predictions") from e

    schema = pa.schema(
        [
            pa.field("trade_date", pa.string()),
            pa.field("ts_code", pa.string()),
            pa.field("pred", pa.float32()),
        ]
    )
    writer = pq.ParquetWriter(out_path, schema=schema)

    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "mlp")).strip().lower()
    is_sequence_model = model_name in {"lstm", "transformer", "tf", "alstm", "tcn", "temporal_conv", "temporal_convolution"}
    seq_len = int(model_cfg.get("seq_len", cfg.get("sample", {}).get("lookback", 60)))
    warmup_start = str(pred_cfg.get("warmup_start_date", start_date))

    if is_sequence_model:
        it = iter_processed_sequence_feature_batches(
            pcfg,
            start_date=warmup_start,
            end_date=end_date,
            feature_cols=feature_cols,
            seq_len=seq_len,
            batch_size=batch_size,
            filter_in_universe=filter_in_universe,
            return_keys=True,
            use_tqdm=use_tqdm,
            stage_desc="predict_seq",
            emit_start_date=start_date,
            cache_in_memory=cache_data,
        )
    else:
        split = ProcessedSplit(name="predict", start_date=start_date, end_date=end_date)
        it = iter_processed_batches(
            pcfg,
            split,
            feature_cols=feature_cols,
            label_col=label_col,
            batch_size=batch_size,
            filter_in_universe=filter_in_universe,
            return_keys=True,
            use_tqdm=use_tqdm,
            stage_desc="predict_tab",
            cache_in_memory=cache_data,
        )

    n_rows = 0
    with torch.no_grad():
        for batch in it:
            xb = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
            pb = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
            trade_date = np.asarray(batch["trade_date"]).astype(str)
            ts_code = np.asarray(batch["ts_code"]).astype(str)
            tab = pa.table({"trade_date": trade_date, "ts_code": ts_code, "pred": pb}, schema=schema)
            writer.write_table(tab)
            n_rows += int(len(pb))

    writer.close()
    _pwrite(tqdm_mod, json.dumps({"saved_pred": out_path, "rows": n_rows}, ensure_ascii=False))
