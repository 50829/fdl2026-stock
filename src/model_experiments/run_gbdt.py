from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data import ProcessedConfig, ProcessedSplit, build_processed_splits, load_feature_columns
from src.evaluation import BacktestConfig, evaluate_prediction_scores
from src.utils import make_run_dir, write_json


def load_tabular_frame(
    pcfg: ProcessedConfig,
    split: ProcessedSplit,
    feature_cols: list[str],
    label_cols: Iterable[str],
    filter_in_universe: bool = True,
) -> pd.DataFrame:
    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise ImportError("pyarrow is required to read data/processed/*.parquet") from e

    proc = Path(pcfg.processed_dir)
    key_trade, key_code = pcfg.key_cols
    label_cols = list(dict.fromkeys(str(c) for c in label_cols))
    date_filter = (ds.field(key_trade) >= split.start_date) & (ds.field(key_trade) <= split.end_date)

    f_cols = [key_trade, key_code] + list(feature_cols)
    l_cols = [key_trade, key_code] + label_cols
    fdf = ds.dataset(str(proc / pcfg.features_path), format="parquet").to_table(columns=f_cols, filter=date_filter).to_pandas()
    ldf = ds.dataset(str(proc / pcfg.labels_path), format="parquet").to_table(columns=l_cols, filter=date_filter).to_pandas()
    df = fdf.merge(ldf, on=[key_trade, key_code], how="inner")

    if filter_in_universe and not df.empty and (proc / pcfg.universe_path).exists():
        flag_col = pcfg.universe_flag_col
        udf = (
            ds.dataset(str(proc / pcfg.universe_path), format="parquet")
            .to_table(columns=[key_trade, key_code, flag_col], filter=date_filter)
            .to_pandas()
        )
        df = df.merge(udf, on=[key_trade, key_code], how="left")
        df = df[df[flag_col].fillna(False)].drop(columns=[flag_col])

    if df.empty:
        return df
    df[key_trade] = df[key_trade].astype(str)
    df[key_code] = df[key_code].astype(str)
    return df.sort_values([key_trade, key_code], kind="mergesort").reset_index(drop=True)


def train_lightgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    args: argparse.Namespace,
):
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "max_bin": args.max_bin,
        "seed": args.seed,
        "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
        "verbosity": -1,
        "num_threads": args.num_threads,
    }
    params = {k: v for k, v in params.items() if v is not None}

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_df[label_col].to_numpy(dtype=np.float32, copy=False)
    X_valid = valid_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_valid = valid_df[label_col].to_numpy(dtype=np.float32, copy=False)
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set, feature_name=feature_cols, free_raw_data=False)

    callbacks = [lgb.log_evaluation(period=args.log_period)]
    if args.early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(args.early_stopping_rounds, first_metric_only=True))

    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return model, params


def train_xgboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    args: argparse.Namespace,
):
    try:
        import xgboost as xgb
    except Exception as e:
        raise RuntimeError(
            "xgboost is not installed in this environment. "
            "Install a CPU-only xgboost wheel before running --model xgboost."
        ) from e

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": args.learning_rate,
        "max_depth": args.xgb_max_depth,
        "min_child_weight": args.xgb_min_child_weight,
        "subsample": args.bagging_fraction,
        "colsample_bytree": args.feature_fraction,
        "lambda": args.lambda_l2,
        "alpha": args.lambda_l1,
        "seed": args.seed,
        "nthread": args.num_threads,
    }
    dtrain = xgb.DMatrix(
        train_df[feature_cols].to_numpy(dtype=np.float32, copy=False),
        label=train_df[label_col].to_numpy(dtype=np.float32, copy=False),
        feature_names=feature_cols,
    )
    dvalid = xgb.DMatrix(
        valid_df[feature_cols].to_numpy(dtype=np.float32, copy=False),
        label=valid_df[label_col].to_numpy(dtype=np.float32, copy=False),
        feature_names=feature_cols,
    )
    callbacks = []
    if args.early_stopping_rounds > 0:
        callbacks.append(xgb.callback.EarlyStopping(rounds=args.early_stopping_rounds, save_best=True))
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=args.log_period,
        callbacks=callbacks,
    )
    return model, params


def predict_model(model, model_name: str, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    if model_name == "xgboost":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(X, feature_names=feature_cols)).astype(np.float32, copy=False)
    best_iter = getattr(model, "best_iteration", None)
    kwargs = {"num_iteration": best_iter} if best_iter else {}
    return model.predict(X, **kwargs).astype(np.float32, copy=False)


def feature_importance(model, model_name: str, feature_cols: list[str]) -> pd.DataFrame:
    if model_name == "xgboost":
        scores = model.get_score(importance_type="gain")
        return (
            pd.DataFrame({"feature": feature_cols, "importance_gain": [float(scores.get(c, 0.0)) for c in feature_cols]})
            .sort_values("importance_gain", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )

    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    return (
        pd.DataFrame({"feature": feature_cols, "importance_gain": gain, "importance_split": split})
        .sort_values("importance_gain", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def evaluate_predictions(pred_df: pd.DataFrame, label_col: str, args: argparse.Namespace) -> dict:
    topk_cfg = BacktestConfig(
        mode="topk",
        n_hold=args.n_hold,
        k_rotate=args.k_rotate,
        step_days=args.step_days,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    rolling_cfg = BacktestConfig(
        mode="rolling_tranche",
        tranche_size=args.tranche_size,
        hold_days=args.hold_days,
        daily_return_col=args.daily_return_col,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    return evaluate_prediction_scores(
        pred_df,
        label_col=label_col,
        raw_return_col=args.raw_return_col,
        daily_return_col=args.daily_return_col,
        topk_cfg=topk_cfg,
        rolling_cfg=rolling_cfg,
    )


def save_model(model, model_name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "xgboost":
        model.save_model(str(path.with_suffix(".json")))
    else:
        model.save_model(str(path.with_suffix(".txt")))


def run(args: argparse.Namespace) -> dict:
    pcfg = ProcessedConfig(processed_dir=args.processed_dir)
    splits = build_processed_splits(pcfg)
    all_feature_cols = load_feature_columns(pcfg)
    if args.feature_list:
        feature_path = Path(args.feature_list)
        feature_cols = [line.strip() for line in feature_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        missing = sorted(set(feature_cols) - set(all_feature_cols))
        if missing:
            raise ValueError(f"Feature list contains columns missing from processed data: {missing}")
        feature_cols = [col for col in all_feature_cols if col in set(feature_cols)]
    else:
        feature_cols = all_feature_cols
    label_cols = [args.target, args.raw_return_col, args.daily_return_col]

    out_dir = Path(args.out_root) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"stage": "load", "split": "train", "processed_dir": args.processed_dir}, ensure_ascii=False))
    train_df = load_tabular_frame(pcfg, splits["train"], feature_cols, label_cols, args.filter_in_universe)
    train_df = train_df.dropna(subset=[args.target]).reset_index(drop=True)
    if args.max_train_rows and len(train_df) > args.max_train_rows:
        train_df = train_df.sample(n=args.max_train_rows, random_state=args.seed).sort_values(pcfg.key_cols[0]).reset_index(drop=True)
    print(json.dumps({"stage": "load", "split": "valid"}, ensure_ascii=False))
    valid_df = load_tabular_frame(pcfg, splits["valid"], feature_cols, label_cols, args.filter_in_universe)
    valid_df = valid_df.dropna(subset=[args.target]).reset_index(drop=True)

    t0 = time.perf_counter()
    if args.model == "lightgbm":
        model, params = train_lightgbm(train_df, valid_df, feature_cols, args.target, args)
    elif args.model == "xgboost":
        model, params = train_xgboost(train_df, valid_df, feature_cols, args.target, args)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    train_sec = time.perf_counter() - t0

    save_model(model, args.model, out_dir / "model")
    feature_importance(model, args.model, feature_cols).to_csv(out_dir / "feature_importance.csv", index=False)

    summary: dict[str, object] = {
        "model": args.model,
        "processed_dir": args.processed_dir,
        "target": args.target,
        "feature_count": len(feature_cols),
        "feature_list": str(args.feature_list) if args.feature_list else None,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_sec": train_sec,
        "params": params,
        "splits": {k: [v.start_date, v.end_date] for k, v in splits.items()},
    }

    for split_name in ["valid", "test"]:
        if split_name == "valid":
            eval_df = valid_df
        else:
            print(json.dumps({"stage": "load", "split": "test"}, ensure_ascii=False))
            eval_df = load_tabular_frame(pcfg, splits[split_name], feature_cols, label_cols, args.filter_in_universe)
            eval_df = eval_df.dropna(subset=[args.target]).reset_index(drop=True)
        pred = predict_model(model, args.model, eval_df, feature_cols)
        pred_df = eval_df[[pcfg.key_cols[0], pcfg.key_cols[1], args.target, args.raw_return_col, args.daily_return_col]].copy()
        pred_df = pred_df.assign(pred=pred)
        split_out = out_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(split_out / f"{split_name}_pred.parquet", index=False)
        metrics = evaluate_predictions(pred_df, args.target, args)
        metrics["split"] = split_name
        write_json(split_out / f"{split_name}_metrics.json", metrics)
        summary[split_name] = metrics
        print(json.dumps({"model": args.model, "split": split_name, "metrics": metrics}, ensure_ascii=False))

    write_json(out_dir / "summary.json", summary)
    return summary


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--processed-dir", default="data/processed_pilot")
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="gbdt_pilot")
    parser.add_argument("--no-timestamp", action="store_true", help="Write to <out-root>/<run-name> instead of timestamping the run directory.")
    parser.add_argument("--target", default="label_5d__cs_rank")
    parser.add_argument("--raw-return-col", default="label_5d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--feature-list", default=None, help="Optional newline-delimited feature list.")
    parser.add_argument("--filter-in-universe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--log-period", type=int, default=50)
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
    args.out_root = str(make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp))
    run(args)


if __name__ == "__main__":
    run_cli()
