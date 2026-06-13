from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import BacktestConfig, evaluate_prediction_scores
from src.utils import make_run_dir, write_json, write_run_metadata


LABEL_COL = "label_1d__cs_rank"
RETURN_COL = "label_1d"
DEFAULT_VALID_PRED = "outputs/models/20260612_151735__nsntk_inspired_label1d/main_model_stability/valid/valid_pred.parquet"
DEFAULT_TEST_PRED = "outputs/models/20260612_151735__nsntk_inspired_label1d/main_model_stability/test/test_pred.parquet"


def _metrics(df: pd.DataFrame) -> dict[str, object]:
    return evaluate_prediction_scores(
        df,
        label_col=LABEL_COL,
        raw_return_col=RETURN_COL,
        daily_return_col=RETURN_COL,
        topk_cfg=BacktestConfig(mode="topk", n_hold=10, k_rotate=2, step_days=1, transaction_cost_bps=5.0),
        rolling_cfg=BacktestConfig(
            mode="rolling_tranche",
            tranche_size=2,
            hold_days=5,
            daily_return_col=RETURN_COL,
            transaction_cost_bps=5.0,
        ),
    )


def _load_universe(path: Path) -> pd.DataFrame:
    cols = ["trade_date", "ts_code", "industry"]
    universe = pd.read_parquet(path, columns=cols)
    universe["trade_date"] = universe["trade_date"].astype(str)
    universe["industry"] = universe["industry"].fillna("UNKNOWN").astype(str)
    return universe.drop_duplicates(["trade_date", "ts_code"])


def _add_industry_neighbor_rank(pred: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    df = pred.merge(universe, on=["trade_date", "ts_code"], how="left")
    df["industry"] = df["industry"].fillna("UNKNOWN")
    df["base_rank"] = df.groupby("trade_date", sort=False)["pred"].rank(method="average", pct=True).astype(np.float32)
    keys = [df["trade_date"], df["industry"]]
    group_sum = df["base_rank"].groupby(keys).transform("sum")
    group_count = df["base_rank"].groupby(keys).transform("count")
    neighbor = (group_sum - df["base_rank"]) / (group_count - 1).replace(0, np.nan)
    df["industry_neighbor_rank"] = neighbor.fillna(df["base_rank"]).astype(np.float32)
    df["industry_count"] = group_count.astype(np.int32)
    return df


def _propagate(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = df[["trade_date", "ts_code", LABEL_COL, RETURN_COL]].copy()
    score = (1.0 - alpha) * df["base_rank"].to_numpy(dtype=np.float32) + alpha * df["industry_neighbor_rank"].to_numpy(dtype=np.float32)
    out["pred"] = score.astype(np.float32)
    return out.dropna(subset=[LABEL_COL, RETURN_COL, "pred"]).reset_index(drop=True)


def _evaluate_grid(split: str, base: pd.DataFrame, alphas: list[float], out_dir: Path) -> tuple[pd.DataFrame, dict[float, Path]]:
    rows: list[dict[str, Any]] = []
    pred_paths: dict[float, Path] = {}
    for alpha in alphas:
        pred = _propagate(base, float(alpha))
        alpha_tag = str(float(alpha)).replace("-", "neg").replace(".", "_")
        alpha_dir = out_dir / split / f"alpha_{alpha_tag}"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        pred_path = alpha_dir / f"{split}_pred.parquet"
        pred.to_parquet(pred_path, index=False)
        metrics = _metrics(pred)
        write_json(alpha_dir / f"{split}_metrics.json", metrics)
        pred_paths[float(alpha)] = pred_path
        rows.append(
            {
                "split": split,
                "alpha_graph": float(alpha),
                "alpha_self": float(1.0 - alpha),
                "pred_path": str(pred_path),
                **metrics,
            }
        )
    grid = pd.DataFrame(rows).sort_values(["bt_sharpe", "icir", "ic_mean"], ascending=False, kind="mergesort").reset_index(drop=True)
    grid.to_csv(out_dir / f"{split}_alpha_grid.csv", index=False)
    return grid, pred_paths


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run industry-graph propagation baseline for stock ranking scores.")
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="gcn_industry_graph_propagation_label1d")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--valid-pred", default=DEFAULT_VALID_PRED)
    parser.add_argument("--test-pred", default=DEFAULT_TEST_PRED)
    parser.add_argument("--universe", default="data/processed/universe.parquet")
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[-0.30, -0.20, -0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
    args = parser.parse_args()

    out_dir = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, command="gcn-propagation", args=args)

    universe = _load_universe(Path(args.universe))
    valid_base = _add_industry_neighbor_rank(pd.read_parquet(args.valid_pred), universe)
    test_base = _add_industry_neighbor_rank(pd.read_parquet(args.test_pred), universe)
    valid_grid, valid_paths = _evaluate_grid("valid", valid_base, [float(x) for x in args.alpha_grid], out_dir)
    test_grid, test_paths = _evaluate_grid("test", test_base, [float(x) for x in args.alpha_grid], out_dir)

    best_alpha = float(valid_grid.iloc[0]["alpha_graph"])
    selected_valid = valid_grid[valid_grid["alpha_graph"].eq(best_alpha)].iloc[0].to_dict()
    selected_test = test_grid[test_grid["alpha_graph"].eq(best_alpha)].iloc[0].to_dict()
    selected = pd.DataFrame(
        [
            {"selection": "valid_best_by_topk10_drop2_sharpe", **selected_valid},
            {"selection": "valid_selected_test_once", **selected_test},
        ]
    )
    selected.to_csv(out_dir / "selected_by_valid_sharpe.csv", index=False)

    selected_dir = out_dir / f"selected_alpha_{str(best_alpha).replace('-', 'neg').replace('.', '_')}"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for split, paths in [("valid", valid_paths), ("test", test_paths)]:
        src = paths[best_alpha]
        dst = selected_dir / f"{split}_pred.parquet"
        pd.read_parquet(src).to_parquet(dst, index=False)

    write_json(
        out_dir / "summary.json",
        {
            "valid_alpha_grid": str(out_dir / "valid_alpha_grid.csv"),
            "test_alpha_grid": str(out_dir / "test_alpha_grid.csv"),
            "selected": str(out_dir / "selected_by_valid_sharpe.csv"),
            "selected_alpha_graph": best_alpha,
            "interpretation": "industry graph propagation: pred = (1-alpha)*self_rank + alpha*industry_neighbor_mean_rank",
        },
    )
    print(json.dumps({"saved": str(out_dir), "selected_alpha_graph": best_alpha}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
