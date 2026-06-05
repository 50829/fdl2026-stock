from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation import BacktestConfig, evaluate_prediction_scores, load_prediction_frame
from src.evaluation.prediction_io import FINAL_PRED_COLUMNS, save_prediction_frame
from src.models.fusion import ResidualRankFusionModel, load_residual_rank_fusion, merge_lgb_xgb_predictions
from src.utils import (
    DEFAULT_ARTIFACT_REGISTRY,
    DEFAULT_EXPERIMENT_REGISTRY,
    apply_experiment_defaults,
    load_registry,
    make_run_dir,
    parser_defaults,
    resolve_bundle,
    resolve_experiment,
    write_json,
    write_run_metadata,
)


DEFAULT_ARTIFACT_BUNDLE = "final_handoff_inputs"


def evaluate_output(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_prediction_scores(
        df,
        label_col=args.target,
        raw_return_col=args.raw_return_col,
        daily_return_col=args.daily_return_col,
        topk_cfg=BacktestConfig(
            mode="topk",
            n_hold=args.n_hold,
            k_rotate=args.k_rotate,
            step_days=args.step_days,
            transaction_cost_bps=args.transaction_cost_bps,
        ),
        rolling_cfg=BacktestConfig(
            mode="rolling_tranche",
            tranche_size=args.tranche_size,
            hold_days=args.hold_days,
            daily_return_col=args.daily_return_col,
            transaction_cost_bps=args.transaction_cost_bps,
        ),
    )


def build_split(
    split: str,
    lgb_path: str | Path,
    xgb_path: str | Path,
    model: ResidualRankFusionModel,
    args: argparse.Namespace,
) -> dict[str, Any]:
    lgb = load_prediction_frame(lgb_path, pred_name="lgb")
    xgb = load_prediction_frame(xgb_path, pred_name="xgb")
    labels = [args.target, args.raw_return_col, args.daily_return_col]
    merged = merge_lgb_xgb_predictions(lgb, xgb, label_cols=labels).dropna(subset=[args.target]).reset_index(drop=True)

    missing_inputs = sorted(set(model.input_columns) - set(merged.columns))
    if missing_inputs:
        raise ValueError(f"Fusion model input columns missing from merged predictions: {missing_inputs}")

    scored = model.predict_frame(merged, batch_size=args.batch_size, device=args.device)
    output_cols = [c for c in FINAL_PRED_COLUMNS + labels if c in scored.columns]
    out_df = scored[output_cols].copy()

    out_dir = Path(args.out_root) / split
    out_path = out_dir / f"{split}_pred.parquet"
    save_prediction_frame(out_df, out_path)
    metrics = evaluate_output(out_df, args)
    write_json(out_dir / f"{split}_metrics.json", metrics)
    print(json.dumps({"split": split, "rows": int(len(out_df)), "output": str(out_path), "metrics": metrics}, ensure_ascii=False), flush=True)
    return {
        "split": split,
        "lgb_path": str(lgb_path),
        "xgb_path": str(xgb_path),
        "output": str(out_path),
        "rows": int(len(out_df)),
        "metrics": metrics,
    }


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-registry", default=DEFAULT_EXPERIMENT_REGISTRY)
    parser.add_argument("--experiment", default=None, help="Load defaults from configs/registry/experiments.yaml.")
    parser.add_argument("--artifact-registry", default=DEFAULT_ARTIFACT_REGISTRY)
    parser.add_argument("--artifact-bundle", default=DEFAULT_ARTIFACT_BUNDLE)
    parser.add_argument("--model-path", default=None, help="Override fusion model artifact path.")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="final_model_handoff")
    parser.add_argument("--no-timestamp", action="store_true", help="Write to <out-root>/<run-name> instead of timestamping the run directory.")
    parser.add_argument("--valid-lgb", default=None)
    parser.add_argument("--valid-xgb", default=None)
    parser.add_argument("--test-lgb", default=None)
    parser.add_argument("--test-xgb", default=None)
    parser.add_argument("--splits", nargs="+", choices=["valid", "test"], default=["valid", "test"])
    parser.add_argument("--target", default="label_5d__cs_rank")
    parser.add_argument("--raw-return-col", default="label_5d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-hold", type=int, default=20)
    parser.add_argument("--k-rotate", type=int, default=5)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--tranche-size", type=int, default=4)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    defaults = parser_defaults(parser)
    args = parser.parse_args()
    if args.experiment:
        try:
            experiment_cfg = resolve_experiment(load_registry(args.experiment_registry), args.experiment, source=args.experiment_registry)
            apply_experiment_defaults(args, experiment_cfg, defaults)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        artifact_registry = load_registry(args.artifact_registry)
        bundle = resolve_bundle(artifact_registry, args.artifact_bundle, source=args.artifact_registry)
    except ValueError as exc:
        parser.error(str(exc))
    args.model_path = args.model_path or bundle["fusion_model"]
    args.valid_lgb = args.valid_lgb or bundle["valid_lgb"]
    args.valid_xgb = args.valid_xgb or bundle["valid_xgb"]
    args.test_lgb = args.test_lgb or bundle["test_lgb"]
    args.test_xgb = args.test_xgb or bundle["test_xgb"]
    args.out_root = str(make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp))
    write_run_metadata(
        args.out_root,
        command="final-handoff",
        args=args,
        inputs={
            "fusion_model": args.model_path,
            "valid_lgb": args.valid_lgb,
            "valid_xgb": args.valid_xgb,
            "test_lgb": args.test_lgb,
            "test_xgb": args.test_xgb,
        },
        registry_paths=[args.experiment_registry, args.artifact_registry],
    )

    split_paths = {
        "valid": (args.valid_lgb, args.valid_xgb),
        "test": (args.test_lgb, args.test_xgb),
    }
    summary: dict[str, Any] = {
        "model_path": str(args.model_path),
        "alpha": float(args.alpha),
        "out_root": str(args.out_root),
        "splits": {},
    }
    model = load_residual_rank_fusion(args.model_path, alpha=args.alpha)
    for split in args.splits:
        lgb_path, xgb_path = split_paths[split]
        summary["splits"][split] = build_split(split, lgb_path, xgb_path, model, args)
    write_json(Path(args.out_root) / "summary.json", summary)


if __name__ == "__main__":
    run_cli()
