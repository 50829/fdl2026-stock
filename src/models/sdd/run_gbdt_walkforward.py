from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ProcessedConfig, ProcessedSplit, load_feature_columns
from src.models.sdd.run_e0_e1 import write_json
from src.models.sdd.run_gbdt import (
    evaluate_predictions,
    feature_importance,
    load_tabular_frame,
    predict_model,
    save_model,
    train_lightgbm,
    train_xgboost,
)


def resolve_features(pcfg: ProcessedConfig, feature_list: str | None) -> list[str]:
    all_cols = load_feature_columns(pcfg)
    if not feature_list:
        return all_cols
    requested = [line.strip() for line in Path(feature_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = sorted(set(requested) - set(all_cols))
    if missing:
        raise ValueError(f"Feature list contains columns missing from processed data: {missing}")
    selected = set(requested)
    return [col for col in all_cols if col in selected]


def year_split(name: str, start_year: int, end_year: int) -> ProcessedSplit:
    return ProcessedSplit(name=name, start_date=f"{start_year}0101", end_date=f"{end_year}1231")


def build_folds(valid_years: list[int], schemes: list[str], min_year: int) -> list[dict[str, object]]:
    folds = []
    for valid_year in valid_years:
        for scheme in schemes:
            if scheme == "expanding":
                train_start = min_year
            elif scheme.startswith("rolling"):
                window = int(scheme.replace("rolling", ""))
                train_start = max(min_year, valid_year - window)
            else:
                raise ValueError(f"Unknown scheme: {scheme}")
            train_end = valid_year - 1
            if train_start > train_end:
                continue
            folds.append(
                {
                    "scheme": scheme,
                    "valid_year": valid_year,
                    "train": year_split("train", train_start, train_end),
                    "valid": year_split("valid", valid_year, valid_year),
                }
            )
    return folds


def run_fold(args: argparse.Namespace, pcfg: ProcessedConfig, feature_cols: list[str], fold: dict[str, object]) -> dict:
    label_cols = [args.target, args.raw_return_col, args.daily_return_col]
    train_split = fold["train"]
    valid_split = fold["valid"]
    assert isinstance(train_split, ProcessedSplit)
    assert isinstance(valid_split, ProcessedSplit)

    fold_name = f"{fold['scheme']}_valid{fold['valid_year']}"
    out_dir = Path(args.out_root) / args.model / fold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "stage": "load",
                "fold": fold_name,
                "train": [train_split.start_date, train_split.end_date],
                "valid": [valid_split.start_date, valid_split.end_date],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    train_df = load_tabular_frame(pcfg, train_split, feature_cols, label_cols, args.filter_in_universe)
    train_df = train_df.dropna(subset=[args.target]).reset_index(drop=True)
    valid_df = load_tabular_frame(pcfg, valid_split, feature_cols, label_cols, args.filter_in_universe)
    valid_df = valid_df.dropna(subset=[args.target]).reset_index(drop=True)

    t0 = time.perf_counter()
    if args.model == "lightgbm":
        model, params = train_lightgbm(train_df, valid_df, feature_cols, args.target, args)
    elif args.model == "xgboost":
        model, params = train_xgboost(train_df, valid_df, feature_cols, args.target, args)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    train_sec = time.perf_counter() - t0

    if args.save_models:
        save_model(model, args.model, out_dir / "model")
        feature_importance(model, args.model, feature_cols).to_csv(out_dir / "feature_importance.csv", index=False)

    pred = predict_model(model, args.model, valid_df, feature_cols)
    key_trade, key_code = pcfg.key_cols
    pred_df = valid_df[[key_trade, key_code, args.target, args.raw_return_col, args.daily_return_col]].copy()
    pred_df["pred"] = pred
    if args.save_predictions:
        pred_df.to_parquet(out_dir / "valid_pred.parquet", index=False)
    metrics = evaluate_predictions(pred_df, args.target, args)
    summary = {
        "fold": fold_name,
        "model": args.model,
        "scheme": fold["scheme"],
        "valid_year": fold["valid_year"],
        "train_start": train_split.start_date,
        "train_end": train_split.end_date,
        "valid_start": valid_split.start_date,
        "valid_end": valid_split.end_date,
        "feature_count": len(feature_cols),
        "feature_list": args.feature_list,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_sec": train_sec,
        "params": params,
        "metrics": metrics,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-root", default="outputs/sdd_gbdt_walkforward")
    parser.add_argument("--feature-list", default=None)
    parser.add_argument("--target", default="label_5d__cs_rank")
    parser.add_argument("--raw-return-col", default="label_5d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--valid-years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument("--schemes", nargs="+", default=["expanding", "rolling3", "rolling5"])
    parser.add_argument("--min-year", type=int, default=2016)
    parser.add_argument("--filter-in-universe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-data-in-leaf", type=int, default=1000)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-min-child-weight", type=float, default=100.0)
    parser.add_argument("--n-hold", type=int, default=20)
    parser.add_argument("--k-rotate", type=int, default=5)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--tranche-size", type=int, default=4)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    pcfg = ProcessedConfig(processed_dir=args.processed_dir)
    feature_cols = resolve_features(pcfg, args.feature_list)
    folds = build_folds(args.valid_years, args.schemes, args.min_year)
    summaries = [run_fold(args, pcfg, feature_cols, fold) for fold in folds]

    out_root = Path(args.out_root) / args.model
    rows = []
    for item in summaries:
        row = {k: item[k] for k in ["fold", "model", "scheme", "valid_year", "train_start", "train_end", "feature_count", "train_rows", "valid_rows", "train_sec"]}
        row.update(item["metrics"])
        rows.append(row)
    result = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_root / "walkforward_summary.csv", index=False)
    write_json(out_root / "walkforward_summary.json", {"experiments": summaries})
    grouped = (
        result.groupby("scheme", as_index=False)[["ic_mean", "icir", "mse", "bt_total_return", "rolling_bt_total_return"]]
        .mean(numeric_only=True)
        .sort_values("icir", ascending=False, kind="mergesort")
    )
    grouped.to_csv(out_root / "walkforward_by_scheme.csv", index=False)
    print(json.dumps({"by_scheme": grouped.to_dict(orient="records")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
