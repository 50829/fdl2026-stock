from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation import prediction_metrics
from src.pipelines.run_strategy_backtest import load_model_registry, resolve_prediction_path
from src.utils import make_run_dir, write_json, write_run_metadata


DEFAULT_MODEL_REGISTRY = "configs/registry/models_report_label1d.yaml"
DEFAULT_FUSION_MODELS = [
    "mlp_label1d",
    "gru_label1d",
    "tcn_label1d",
    "lgb_label1d",
    "xgb_label1d",
]
GBDT_MODELS = ["lgb_label1d", "xgb_label1d"]


def _load_prediction(path: str | Path, model: str, score_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"trade_date", "ts_code", score_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"`{path}` for `{model}` missing columns: {missing}")
    out = df.copy()
    out = out.rename(columns={score_col: f"pred__{model}"})
    return out


def _merge_split(registry: dict[str, Any], models: list[str], split: str, score_col: str) -> pd.DataFrame:
    first = models[0]
    merged = _load_prediction(resolve_prediction_path(registry, first, split), first, score_col)
    pred_cols = [f"pred__{first}"]
    for model in models[1:]:
        df = _load_prediction(resolve_prediction_path(registry, model, split), model, score_col)
        pred_col = f"pred__{model}"
        pred_cols.append(pred_col)
        merged = merged.merge(df[["trade_date", "ts_code", pred_col]], on=["trade_date", "ts_code"], how="inner")
    return merged.dropna(subset=pred_cols).reset_index(drop=True)


def _daily_rank(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("trade_date", sort=False)[col].rank(method="average", pct=True).astype(np.float32)


def _rank_ic_by_day(df: pd.DataFrame, score_col: str, label_col: str) -> pd.Series:
    def corr_one_day(g: pd.DataFrame) -> float:
        if g[score_col].nunique(dropna=True) <= 1 or g[label_col].nunique(dropna=True) <= 1:
            return np.nan
        return float(g[score_col].corr(g[label_col], method="pearson"))

    return df.groupby("trade_date", sort=False).apply(corr_one_day, include_groups=False).dropna()


def _diagnose_valid_ic(df: pd.DataFrame, models: list[str], label_col: str) -> pd.DataFrame:
    rows = []
    for model in models:
        rank_col = f"rank__{model}"
        ic = _rank_ic_by_day(df, rank_col, label_col)
        mean = float(ic.mean()) if len(ic) else 0.0
        std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
        rows.append(
            {
                "model": model,
                "n_days": int(len(ic)),
                "ic_mean": mean,
                "ic_std": std,
                "icir": float(mean / std) if std > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["icir", "ic_mean"], ascending=False, kind="mergesort")


def _positive_weights(metric_df: pd.DataFrame, metric: str, models: list[str]) -> dict[str, float]:
    values = metric_df.set_index("model").reindex(models)[metric].fillna(0.0).clip(lower=0.0).astype(float)
    if float(values.sum()) <= 0.0:
        return {model: 1.0 / len(models) for model in models}
    weights = values / float(values.sum())
    return {model: float(weights.loc[model]) for model in models}


def _weighted_rank_score(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    score = np.zeros(len(df), dtype=np.float32)
    for model, weight in weights.items():
        if weight:
            score += float(weight) * df[f"rank__{model}"].to_numpy(dtype=np.float32, copy=False)
    return score


def _prediction_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    cols = ["trade_date", "ts_code"]
    for col in [label_col, "label_1d", "label_5d", "label_1d__cs_rank", "label_5d__cs_rank"]:
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols


def _write_registry(path: Path, base_registry: dict[str, Any], fusion_entries: dict[str, dict[str, Any]]) -> None:
    payload = {
        "models": {**base_registry.get("models", {}), **fusion_entries},
        "feature_sets": base_registry.get("feature_sets", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_prediction_metrics(pred_df: pd.DataFrame, split: str, label_col: str, out_dir: Path) -> dict[str, object]:
    metrics = prediction_metrics(pred_df, label_col=label_col, raw_return_col="label_1d" if "label_1d" in pred_df.columns else None)
    metrics.update(
        {
            "split": split,
            "start_date": str(pred_df["trade_date"].min()) if len(pred_df) else None,
            "end_date": str(pred_df["trade_date"].max()) if len(pred_df) else None,
            "n_dates": int(pred_df["trade_date"].nunique()) if len(pred_df) else 0,
            "metric_type": "prediction_quality",
        }
    )
    write_json(out_dir / f"{split}_metrics.json", metrics)
    return metrics


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Generate report-grade label1d prediction fusions.")
    parser.add_argument("--model-registry", default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--models", nargs="+", default=DEFAULT_FUSION_MODELS)
    parser.add_argument("--splits", nargs="+", choices=["valid", "test"], default=["valid", "test"])
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="report_label1d_fusion")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--score-col", default="pred")
    parser.add_argument("--label-col", default="label_1d__cs_rank")
    args = parser.parse_args()

    registry = load_model_registry(args.model_registry)
    models = [str(model) for model in args.models]
    unknown = sorted(set(models) - set(registry["models"]))
    if unknown:
        parser.error("unknown --models value(s): " + ", ".join(unknown))
    if "valid" not in args.splits:
        parser.error("valid split is required to estimate IC-based fusion weights")

    out_root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    write_run_metadata(
        out_root,
        command="report-fusion",
        args=args,
        inputs={"model_registry": args.model_registry, "models": models},
        registry_paths=[args.model_registry],
    )

    valid_all = _merge_split(registry, models, "valid", args.score_col)
    if args.label_col not in valid_all.columns:
        raise ValueError(f"valid split missing label column `{args.label_col}`")
    for model in models:
        valid_all[f"rank__{model}"] = _daily_rank(valid_all, f"pred__{model}")

    valid_ic = _diagnose_valid_ic(valid_all, models, args.label_col)
    valid_ic.to_csv(out_root / "valid_model_rank_ic.csv", index=False)

    equal_all = {model: 1.0 / len(models) for model in models}
    gbdt_present = [model for model in GBDT_MODELS if model in models]
    variants: dict[str, dict[str, float]] = {
        "fusion_rank_equal_all": equal_all,
        "fusion_rank_valid_ic_weighted_all": _positive_weights(valid_ic, "ic_mean", models),
        "fusion_rank_valid_icir_weighted_all": _positive_weights(valid_ic, "icir", models),
    }
    if gbdt_present:
        variants["fusion_rank_equal_gbdt"] = {model: 1.0 / len(gbdt_present) for model in gbdt_present}

    fusion_entries: dict[str, dict[str, Any]] = {}
    weight_rows = []
    for variant, weights in variants.items():
        fusion_entries[variant] = {
            "description": f"报告主实验：label1d 预测层融合 `{variant}`。",
            "predictions": {},
        }
        for model in models:
            weight_rows.append({"variant": variant, "model": model, "weight": float(weights.get(model, 0.0))})
        active_models = [model for model in models if weights.get(model, 0.0) > 0.0]
        for split in args.splits:
            df = _merge_split(registry, active_models, split, args.score_col)
            if args.label_col not in df.columns:
                raise ValueError(f"split `{split}` missing label column `{args.label_col}`")
            for model in active_models:
                df[f"rank__{model}"] = _daily_rank(df, f"pred__{model}")
            pred_df = df[_prediction_columns(df, args.label_col)].copy()
            pred_df["pred"] = _weighted_rank_score(df, weights)
            split_dir = out_root / variant / split
            split_dir.mkdir(parents=True, exist_ok=True)
            pred_path = split_dir / f"{split}_pred.parquet"
            pred_df.to_parquet(pred_path, index=False)
            metrics = _write_prediction_metrics(pred_df, split, args.label_col, split_dir)
            fusion_entries[variant]["predictions"][split] = str(pred_path)
            fusion_entries[variant].setdefault("metrics", {})[split] = str(split_dir / f"{split}_metrics.json")
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "split": split,
                        "rows": int(len(pred_df)),
                        "path": str(pred_path),
                        "metrics": metrics,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    pd.DataFrame(weight_rows).to_csv(out_root / "fusion_weights.csv", index=False)
    registry_path = out_root / "models_report_label1d_with_fusion.yaml"
    _write_registry(registry_path, registry, fusion_entries)
    summary = {
        "out_root": str(out_root),
        "registry": str(registry_path),
        "base_models": models,
        "variants": list(variants),
        "valid_ic_csv": str(out_root / "valid_model_rank_ic.csv"),
        "weights_csv": str(out_root / "fusion_weights.csv"),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved_summary": str(out_root / "summary.json"), "registry": str(registry_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
