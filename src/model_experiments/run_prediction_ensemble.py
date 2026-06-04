from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import BacktestConfig, evaluate_prediction_scores, load_prediction_frame
from src.utils import make_run_dir, write_json


def load_pred(path: str | Path, name: str) -> pd.DataFrame:
    return load_prediction_frame(path, pred_name=name)


def rank_feature(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("trade_date")[col].rank(method="average", pct=True)


def weight_grid(names: list[str], step: float) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    combos = []
    for parts in itertools.product(range(units + 1), repeat=len(names)):
        if sum(parts) != units:
            continue
        combos.append({name: part / units for name, part in zip(names, parts)})
    return combos


def evaluate(df: pd.DataFrame, label_col: str, raw_return_col: str, daily_return_col: str) -> dict:
    return evaluate_prediction_scores(
        df,
        label_col=label_col,
        raw_return_col=raw_return_col,
        daily_return_col=daily_return_col,
        topk_cfg=BacktestConfig(mode="topk", n_hold=20, k_rotate=5, step_days=5, transaction_cost_bps=5.0),
        rolling_cfg=BacktestConfig(
            mode="rolling_tranche",
            tranche_size=4,
            hold_days=5,
            daily_return_col=daily_return_col,
            transaction_cost_bps=5.0,
        ),
    )


def merge_predictions(paths: dict[str, str], label_col: str, raw_return_col: str, daily_return_col: str) -> pd.DataFrame:
    names = list(paths)
    merged = load_pred(paths[names[0]], names[0])
    keep_cols = ["trade_date", "ts_code", f"pred_{names[0]}"]
    for col in [label_col, raw_return_col, daily_return_col]:
        if col in merged.columns and col not in keep_cols:
            keep_cols.append(col)
    merged = merged[keep_cols]

    for name in names[1:]:
        df = load_pred(paths[name], name)[["trade_date", "ts_code", f"pred_{name}"]]
        merged = merged.merge(df, on=["trade_date", "ts_code"], how="inner")

    missing_labels = [c for c in [label_col, raw_return_col, daily_return_col] if c not in merged.columns]
    if missing_labels:
        raise ValueError(f"Base prediction file must contain labels: {missing_labels}")
    return merged.dropna(subset=[label_col]).reset_index(drop=True)


def run_split(
    split: str,
    paths: dict[str, str],
    out_dir: Path,
    label_col: str,
    raw_return_col: str,
    daily_return_col: str,
    step: float,
) -> pd.DataFrame:
    names = list(paths)
    df = merge_predictions(paths, label_col, raw_return_col, daily_return_col)
    for name in names:
        df[f"rank_{name}"] = rank_feature(df, f"pred_{name}")

    rows = []
    for weights in weight_grid(names, step):
        score = np.zeros(len(df), dtype=np.float32)
        for name, weight in weights.items():
            if weight:
                score += float(weight) * df[f"rank_{name}"].to_numpy(dtype=np.float32, copy=False)
        pred_df = df[["trade_date", "ts_code", label_col, raw_return_col, daily_return_col]].copy()
        pred_df["pred"] = score
        metrics = evaluate(pred_df, label_col, raw_return_col, daily_return_col)
        row = {"split": split, **{f"w_{name}": weight for name, weight in weights.items()}, **metrics}
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(["icir", "ic_mean"], ascending=False, kind="mergesort").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / f"{split}_ensemble_grid.csv", index=False)
    best = result.iloc[0].to_dict() if not result.empty else {}
    write_json(out_dir / f"{split}_best_metrics.json", best)
    print(json.dumps({"split": split, "best": best}, ensure_ascii=False))
    return result


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="ensemble_full")
    parser.add_argument("--no-timestamp", action="store_true", help="Write to <out-root>/<run-name> instead of timestamping the run directory.")
    parser.add_argument("--label-col", default="label_5d__cs_rank")
    parser.add_argument("--raw-return-col", default="label_5d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--valid-gru", default="outputs/models/20260530_103415__sequence_ablation_full/layer1/valid/valid_pred.parquet")
    parser.add_argument("--test-gru", default="outputs/models/20260530_103903__final_test_eval/layer1/test/test_pred.parquet")
    parser.add_argument("--valid-lightgbm", default="outputs/models/20260530_200734__gbdt_full/lightgbm/valid/valid_pred.parquet")
    parser.add_argument("--test-lightgbm", default="outputs/models/20260530_200734__gbdt_full/lightgbm/test/test_pred.parquet")
    parser.add_argument("--valid-xgboost", default="outputs/models/20260530_200734__gbdt_full/xgboost/valid/valid_pred.parquet")
    parser.add_argument("--test-xgboost", default="outputs/models/20260530_200734__gbdt_full/xgboost/test/test_pred.parquet")
    args = parser.parse_args()

    out_root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    valid = run_split(
        "valid",
        {"lightgbm": args.valid_lightgbm, "xgboost": args.valid_xgboost, "gru": args.valid_gru},
        out_root,
        args.label_col,
        args.raw_return_col,
        args.daily_return_col,
        args.grid_step,
    )
    test = run_split(
        "test",
        {"lightgbm": args.test_lightgbm, "xgboost": args.test_xgboost, "gru": args.test_gru},
        out_root,
        args.label_col,
        args.raw_return_col,
        args.daily_return_col,
        args.grid_step,
    )
    summary = {
        "valid_best_by_icir": valid.iloc[0].to_dict() if not valid.empty else {},
        "test_best_by_icir": test.iloc[0].to_dict() if not test.empty else {},
    }
    write_json(out_root / "summary.json", summary)


if __name__ == "__main__":
    run_cli()
