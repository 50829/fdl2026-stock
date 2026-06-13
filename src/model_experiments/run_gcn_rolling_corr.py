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


def _period_start(date: pd.Timestamp, freq: str) -> pd.Timestamp:
    freq = freq.upper()
    if freq == "M":
        return date.to_period("M").start_time
    if freq == "Q":
        return date.to_period("Q").start_time
    raise ValueError(f"unsupported rebalance frequency: {freq}")


def _load_returns(path: Path, return_feature: str) -> pd.DataFrame:
    cols = ["trade_date", "ts_code", return_feature]
    df = pd.read_parquet(path, columns=cols).dropna(subset=[return_feature]).copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    df["ts_code"] = df["ts_code"].astype(str)
    df[return_feature] = df[return_feature].astype(np.float32)
    return df.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def _load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    df["ts_code"] = df["ts_code"].astype(str)
    return df.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def _build_neighbors(
    returns: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    window_days: int,
    return_feature: str,
    top_k: int,
    min_obs: int,
    min_corr: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    hist_dates = returns.loc[returns["trade_date"] < as_of, "trade_date"].drop_duplicates().sort_values().tail(window_days)
    if hist_dates.empty:
        return {}
    hist = returns[returns["trade_date"].isin(hist_dates)]
    pivot = hist.pivot_table(index="trade_date", columns="ts_code", values=return_feature, aggfunc="mean")
    pivot = pivot.dropna(axis=1, thresh=min_obs)
    if pivot.shape[1] < 3:
        return {}

    corr = pivot.corr(min_periods=min_obs).astype(np.float32)
    corr_values = corr.to_numpy(dtype=np.float32, copy=True)
    np.fill_diagonal(corr_values, np.nan)
    corr = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)
    corr = corr.where(corr >= float(min_corr))
    neighbors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, row in corr.iterrows():
        top = row.dropna().sort_values(ascending=False).head(top_k)
        if top.empty:
            continue
        weights = top.to_numpy(dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-12)
        neighbors[str(code)] = (top.index.astype(str).to_numpy(), weights)
    return neighbors


def _add_corr_neighbor_rank(
    pred: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    window_days: int,
    return_feature: str,
    top_k: int,
    min_obs: int,
    min_corr: float,
    rebalance: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pred.copy()
    df["period_start"] = df["trade_date"].map(lambda x: _period_start(pd.Timestamp(x), rebalance))
    df["base_rank"] = df.groupby("trade_date", sort=False)["pred"].rank(method="average", pct=True).astype(np.float32)

    period_frames: list[pd.DataFrame] = []
    graph_rows: list[dict[str, Any]] = []
    periods = sorted(df["period_start"].drop_duplicates().tolist())
    for idx, period in enumerate(periods, start=1):
        print(f"[rolling-corr-gcn] build graph {idx}/{len(periods)} period={pd.Timestamp(period).date()}", flush=True)
        neighbors = _build_neighbors(
            returns,
            as_of=pd.Timestamp(period),
            window_days=window_days,
            return_feature=return_feature,
            top_k=top_k,
            min_obs=min_obs,
            min_corr=min_corr,
        )
        part = df[df["period_start"].eq(period)].copy()
        graph_rows.append(
            {
                "period_start": str(pd.Timestamp(period).date()),
                "n_dates": int(part["trade_date"].nunique()),
                "n_rows": int(len(part)),
                "n_nodes_with_edges": int(len(neighbors)),
                "avg_degree": float(np.mean([len(v[0]) for v in neighbors.values()])) if neighbors else 0.0,
            }
        )

        neighbor_scores: list[pd.Series] = []
        for _, day in part.groupby("trade_date", sort=True):
            rank = day.set_index("ts_code")["base_rank"]
            values: list[float] = []
            for code, own_rank in zip(day["ts_code"].to_numpy(), day["base_rank"].to_numpy(dtype=np.float32), strict=False):
                if code not in neighbors:
                    values.append(float(own_rank))
                    continue
                codes, weights = neighbors[code]
                available = rank.reindex(codes).dropna()
                if available.empty:
                    values.append(float(own_rank))
                    continue
                weight_series = pd.Series(weights, index=codes).reindex(available.index).astype(np.float32)
                weight_sum = float(weight_series.sum())
                if weight_sum <= 0:
                    values.append(float(own_rank))
                else:
                    values.append(float(np.dot(available.to_numpy(dtype=np.float32), weight_series.to_numpy(dtype=np.float32)) / weight_sum))
            tmp = day.copy()
            tmp["corr_neighbor_rank"] = np.asarray(values, dtype=np.float32)
            neighbor_scores.append(tmp)
        if neighbor_scores:
            period_frames.append(pd.concat(neighbor_scores, ignore_index=True))

    out = pd.concat(period_frames, ignore_index=True) if period_frames else df.assign(corr_neighbor_rank=df["base_rank"])
    out = out.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    return out, pd.DataFrame(graph_rows)


def _propagate(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = df[["trade_date", "ts_code", LABEL_COL, RETURN_COL]].copy()
    score = (1.0 - alpha) * df["base_rank"].to_numpy(dtype=np.float32) + alpha * df["corr_neighbor_rank"].to_numpy(dtype=np.float32)
    out["pred"] = score.astype(np.float32)
    out["trade_date"] = out["trade_date"].dt.strftime("%Y%m%d")
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
        rows.append({"split": split, "alpha_graph": float(alpha), "alpha_self": float(1.0 - alpha), "pred_path": str(pred_path), **metrics})
    grid = pd.DataFrame(rows).sort_values(["bt_sharpe", "icir", "ic_mean"], ascending=False, kind="mergesort").reset_index(drop=True)
    grid.to_csv(out_dir / f"{split}_alpha_grid.csv", index=False)
    return grid, pred_paths


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run rolling return-correlation graph propagation for stock ranking scores.")
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="gcn_rolling_corr_label1d")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--valid-pred", default=DEFAULT_VALID_PRED)
    parser.add_argument("--test-pred", default=DEFAULT_TEST_PRED)
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--return-feature", default="ret_1__cs_rank")
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--top-k-neighbors", type=int, default=10)
    parser.add_argument("--min-obs", type=int, default=40)
    parser.add_argument("--min-corr", type=float, default=0.20)
    parser.add_argument("--rebalance", choices=["M", "Q"], default="M")
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[-0.30, -0.20, -0.10, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
    args = parser.parse_args()

    out_dir = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, command="gcn-rolling-corr", args=args)

    returns = _load_returns(Path(args.features), args.return_feature)
    valid_base, valid_graph = _add_corr_neighbor_rank(
        _load_pred(Path(args.valid_pred)),
        returns,
        window_days=int(args.window_days),
        return_feature=str(args.return_feature),
        top_k=int(args.top_k_neighbors),
        min_obs=int(args.min_obs),
        min_corr=float(args.min_corr),
        rebalance=str(args.rebalance),
    )
    test_base, test_graph = _add_corr_neighbor_rank(
        _load_pred(Path(args.test_pred)),
        returns,
        window_days=int(args.window_days),
        return_feature=str(args.return_feature),
        top_k=int(args.top_k_neighbors),
        min_obs=int(args.min_obs),
        min_corr=float(args.min_corr),
        rebalance=str(args.rebalance),
    )
    pd.concat([valid_graph.assign(split="valid"), test_graph.assign(split="test")], ignore_index=True).to_csv(out_dir / "graph_period_summary.csv", index=False)

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
        pd.read_parquet(paths[best_alpha]).to_parquet(selected_dir / f"{split}_pred.parquet", index=False)

    write_json(
        out_dir / "summary.json",
        {
            "valid_alpha_grid": str(out_dir / "valid_alpha_grid.csv"),
            "test_alpha_grid": str(out_dir / "test_alpha_grid.csv"),
            "selected": str(out_dir / "selected_by_valid_sharpe.csv"),
            "graph_period_summary": str(out_dir / "graph_period_summary.csv"),
            "selected_alpha_graph": best_alpha,
            "interpretation": "rolling correlation graph propagation: pred = (1-alpha)*self_rank + alpha*rolling_corr_neighbor_rank",
        },
    )
    print(json.dumps({"saved": str(out_dir), "selected_alpha_graph": best_alpha}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
