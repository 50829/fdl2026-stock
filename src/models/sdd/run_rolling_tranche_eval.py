from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import ProcessedConfig
from src.models.sdd.run_e0_e1 import BacktestConfig, backtest_rolling_tranche, backtest_topk, ic_metrics, write_json


DEFAULT_PRED_PATHS = {
    "valid": "outputs/sdd_ablation_full/layer1/valid/valid_pred.parquet",
    "test": "outputs/sdd_final_test_eval/layer1/test/test_pred.parquet",
}


def attach_label_columns(pcfg: ProcessedConfig, pred_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    import pyarrow.dataset as ds

    key_trade, key_code = pcfg.key_cols
    need_cols = [col for col in columns if col not in pred_df.columns]
    if not need_cols or pred_df.empty:
        return pred_df

    start_date = str(pred_df[key_trade].min())
    end_date = str(pred_df[key_trade].max())
    label_path = Path(pcfg.processed_dir) / pcfg.labels_path
    date_filter = (ds.field(key_trade) >= start_date) & (ds.field(key_trade) <= end_date)
    labels = (
        ds.dataset(str(label_path), format="parquet")
        .to_table(columns=[key_trade, key_code] + need_cols, filter=date_filter)
        .to_pandas()
    )
    labels[key_trade] = labels[key_trade].astype(str)
    labels[key_code] = labels[key_code].astype(str)
    return pred_df.merge(labels, on=[key_trade, key_code], how="left")


def evaluate_pred_file(
    pred_path: Path,
    processed_dir: str,
    out_dir: Path,
    split: str,
    label_col: str,
    five_day_return_col: str,
    daily_return_col: str,
    n_hold: int,
    k_rotate: int,
    tranche_size: int,
    hold_days: int,
    transaction_cost_bps: float,
) -> dict:
    pcfg = ProcessedConfig(processed_dir=processed_dir)
    pred_df = pd.read_parquet(pred_path)
    pred_df["trade_date"] = pred_df["trade_date"].astype(str)
    pred_df["ts_code"] = pred_df["ts_code"].astype(str)
    pred_df = attach_label_columns(pcfg, pred_df, [label_col, five_day_return_col, daily_return_col])
    pred_df = pred_df.dropna(subset=[label_col])

    topk_cfg = BacktestConfig(
        mode="topk",
        n_hold=n_hold,
        k_rotate=k_rotate,
        step_days=hold_days,
        transaction_cost_bps=transaction_cost_bps,
    )
    rolling_cfg = BacktestConfig(
        mode="rolling_tranche",
        tranche_size=tranche_size,
        hold_days=hold_days,
        daily_return_col=daily_return_col,
        transaction_cost_bps=transaction_cost_bps,
    )

    metrics = {
        "split": split,
        "pred_path": str(pred_path),
        "samples": int(len(pred_df)),
        "label_col": label_col,
        "five_day_return_col": five_day_return_col,
        "daily_return_col": daily_return_col,
    }
    metrics.update(ic_metrics(pred_df, label_col=label_col))
    metrics["topk_5d_nonoverlap"] = backtest_topk(pred_df, return_col=five_day_return_col, cfg=topk_cfg)
    metrics["rolling_tranche_daily"] = backtest_rolling_tranche(pred_df, cfg=rolling_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{split}_rolling_tranche_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--valid-pred", default=DEFAULT_PRED_PATHS["valid"])
    parser.add_argument("--test-pred", default=DEFAULT_PRED_PATHS["test"])
    parser.add_argument("--out-root", default="outputs/sdd_rolling_tranche_eval")
    parser.add_argument("--label-col", default="label_5d__cs_rank")
    parser.add_argument("--five-day-return-col", default="label_5d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--n-hold", type=int, default=20)
    parser.add_argument("--k-rotate", type=int, default=5)
    parser.add_argument("--tranche-size", type=int, default=4)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    summaries = {}
    for split, pred in {"valid": args.valid_pred, "test": args.test_pred}.items():
        pred_path = Path(pred)
        if not pred_path.exists():
            print(json.dumps({"split": split, "missing": str(pred_path)}, ensure_ascii=False), flush=True)
            continue
        metrics = evaluate_pred_file(
            pred_path=pred_path,
            processed_dir=args.processed_dir,
            out_dir=out_root / split,
            split=split,
            label_col=args.label_col,
            five_day_return_col=args.five_day_return_col,
            daily_return_col=args.daily_return_col,
            n_hold=args.n_hold,
            k_rotate=args.k_rotate,
            tranche_size=args.tranche_size,
            hold_days=args.hold_days,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        summaries[split] = metrics
        print(json.dumps({"split": split, "metrics": metrics}, ensure_ascii=False), flush=True)
    write_json(out_root / "summary.json", {"splits": summaries})


if __name__ == "__main__":
    main()
