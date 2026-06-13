from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import make_run_dir, write_json, write_run_metadata


LABEL1D_ROOT = Path("outputs/models/20260612_151735__nsntk_inspired_label1d")
EMA_ROOT = Path("outputs/models/20260612_172058__nsntk_ema_grid_label1d")
SEQ_LEN_ROOT = Path("outputs/models/20260613_014931__seq_len_fusion_label1d")
INDUSTRY_GCN_ROOT = Path("outputs/models/20260613_094806__gcn_industry_graph_propagation_label1d")
ROLLING_CORR_GCN_ROOT = Path("outputs/models/20260613_100914__gcn_rolling_corr_label1d")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(name: str, path: Path, group: str, label: str = "label1d") -> dict[str, Any]:
    data = _read_json(path)
    return {
        "模型/实验": name,
        "类别": group,
        "标签": label,
        "Test IC": data.get("ic_mean"),
        "Test ICIR": data.get("icir"),
        "TopK10 Drop2 总收益": data.get("bt_total_return"),
        "TopK10 Drop2 夏普": data.get("bt_sharpe"),
        "最大回撤": data.get("bt_max_drawdown"),
        "平均换手": data.get("bt_avg_turnover"),
        "来源": str(path),
    }


def _selected_test_row(path: Path, name: str, group: str, label: str = "label1d") -> dict[str, Any]:
    df = pd.read_csv(path)
    row = df[df["selection"].eq("valid_selected_test_once")].iloc[0]
    out = {
        "模型/实验": name,
        "类别": group,
        "标签": label,
        "Valid 选择参数": "",
        "Test IC": row.get("ic_mean"),
        "Test ICIR": row.get("icir"),
        "TopK10 Drop2 总收益": row.get("bt_total_return"),
        "TopK10 Drop2 夏普": row.get("bt_sharpe"),
        "最大回撤": row.get("bt_max_drawdown"),
        "平均换手": row.get("bt_avg_turnover"),
        "来源": str(path),
    }
    for col in ["alpha_deep", "alpha_graph"]:
        if col in row:
            out["Valid 选择参数"] = f"{col}={row[col]}"
    return out


def _fusion_selected_rows(path: Path, label: str = "label1d") -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    rows = df[df["selection"].eq("valid_selected_test_once")].copy()
    out: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        candidate = str(row.get("candidate", "unknown"))
        out.append(
            {
                "模型/实验": f"树模型 + {candidate}",
                "类别": "树+深度融合",
                "标签": label,
                "Valid 选择参数": f"alpha_deep={row.get('alpha_deep')}",
                "Test IC": row.get("ic_mean"),
                "Test ICIR": row.get("icir"),
                "TopK10 Drop2 总收益": row.get("bt_total_return"),
                "TopK10 Drop2 夏普": row.get("bt_sharpe"),
                "最大回撤": row.get("bt_max_drawdown"),
                "平均换手": row.get("bt_avg_turnover"),
                "来源": str(path),
            }
        )
    return out


def _best_seq_len_rows() -> pd.DataFrame:
    df = pd.read_csv(SEQ_LEN_ROOT / "seq_len_candidate_summary.csv")
    seed = pd.read_csv(SEQ_LEN_ROOT / "seq_len_seed_summary.csv")
    test = df[df["split"].eq("test")].copy()
    present = set(zip(test["model"].astype(str), test["seq_len"].astype(int)))
    seed_test = seed[seed["split"].eq("test")].copy()
    missing_parts = []
    for (model, seq_len), group in seed_test.groupby(["model", "seq_len"], sort=True):
        if (str(model), int(seq_len)) in present:
            continue
        # Single-seed TCN rows are still valid seq_len ablations. If a future run has
        # multiple missing seeds, report their mean to keep one line per seq_len.
        row = group.mean(numeric_only=True).to_dict()
        row.update(
            {
                "candidate": f"{model}_seq{int(seq_len)}_seed_mean",
                "model": model,
                "seq_len": int(seq_len),
                "deep_coverage": group["deep_coverage"].mean() if "deep_coverage" in group else float("nan"),
            }
        )
        missing_parts.append(row)
    if missing_parts:
        test = pd.concat([test, pd.DataFrame(missing_parts)], ignore_index=True)
    cols = [
        "candidate",
        "model",
        "seq_len",
        "deep_coverage",
        "ic_mean",
        "icir",
        "bt_total_return",
        "bt_sharpe",
        "bt_max_drawdown",
        "bt_avg_turnover",
    ]
    out = test[cols].rename(
        columns={
            "candidate": "模型/实验",
            "model": "模型族",
            "seq_len": "序列长度",
            "deep_coverage": "覆盖率",
            "ic_mean": "Test IC",
            "icir": "Test ICIR",
            "bt_total_return": "TopK10 Drop2 总收益",
            "bt_sharpe": "TopK10 Drop2 夏普",
            "bt_max_drawdown": "最大回撤",
            "bt_avg_turnover": "平均换手",
        }
    )
    return out.sort_values(["模型族", "序列长度"], kind="mergesort").reset_index(drop=True)


def _deep_ema_rows() -> pd.DataFrame:
    compact = pd.read_csv(EMA_ROOT / "all_experiment_metrics_compact.csv")
    test = compact[compact["split"].eq("test")].copy()
    test = test[test["section"].isin(["deep_ema_seed", "seed_ensemble"])]
    keep = test[["section", "model", "variant", "seed", "seed_count", "ic_mean", "icir", "bt_total_return", "bt_sharpe", "bt_max_drawdown", "bt_avg_turnover"]].copy()
    return keep.rename(
        columns={
            "section": "实验组",
            "model": "模型",
            "variant": "权重版本",
            "seed": "seed",
            "seed_count": "seed数量",
            "ic_mean": "Test IC",
            "icir": "Test ICIR",
            "bt_total_return": "TopK10 Drop2 总收益",
            "bt_sharpe": "TopK10 Drop2 夏普",
            "bt_max_drawdown": "最大回撤",
            "bt_avg_turnover": "平均换手",
        }
    )


def _nsntk_rows() -> pd.DataFrame:
    compact = pd.read_csv(LABEL1D_ROOT / "all_experiment_metrics_compact.csv")
    test = compact[compact["split"].eq("test")].copy()
    keep = test[
        [
            "section",
            "model",
            "variant",
            "seed_count",
            "alpha",
            "ic_mean",
            "icir",
            "bt_total_return",
            "bt_sharpe",
            "bt_max_drawdown",
            "bt_avg_turnover",
        ]
    ].copy()
    return keep.rename(
        columns={
            "section": "实验组",
            "model": "模型",
            "variant": "设置",
            "seed_count": "seed数量",
            "alpha": "alpha",
            "ic_mean": "Test IC",
            "icir": "Test ICIR",
            "bt_total_return": "TopK10 Drop2 总收益",
            "bt_sharpe": "TopK10 Drop2 夏普",
            "bt_max_drawdown": "最大回撤",
            "bt_avg_turnover": "平均换手",
        }
    )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Build final Chinese report tables from completed experiment outputs.")
    parser.add_argument("--out-root", default="outputs/report_tables")
    parser.add_argument("--run-name", default="final_report_tables")
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    out_dir = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, command="final-report-tables", args=args)

    main_rows = [
        _metric_row("LightGBM uniform", LABEL1D_ROOT / "time_decay/uniform/lightgbm/test/test_metrics.json", "树模型"),
        _metric_row("XGBoost uniform", LABEL1D_ROOT / "time_decay/uniform/xgboost/test/test_metrics.json", "树模型"),
        _metric_row("LGB+XGB rank_mean 主模型", LABEL1D_ROOT / "main_model_stability/test/test_metrics.json", "树模型融合"),
        _metric_row("label5d LightGBM", Path("outputs/models/20260611_210718__report_label5d_lgb_top40_rerun/lightgbm/test/test_metrics.json"), "树模型", "label5d"),
        _metric_row("label5d XGBoost", Path("outputs/models/20260611_210936__report_label5d_xgb_top40_rerun/xgboost/test/test_metrics.json"), "树模型", "label5d"),
        _metric_row("label5d Ridge stacking", Path("outputs/models/20260611_212000__report_label5d_residual_rank_rerun/stacking_ridge/test/test_metrics.json"), "融合", "label5d"),
        _selected_test_row(INDUSTRY_GCN_ROOT / "selected_by_valid_sharpe.csv", "静态行业图传播", "图传播"),
        _selected_test_row(ROLLING_CORR_GCN_ROOT / "selected_by_valid_sharpe.csv", "滚动相关图传播", "图传播"),
    ]
    main_rows.extend(_fusion_selected_rows(SEQ_LEN_ROOT / "fusion_selected_by_valid_sharpe.csv"))
    main = pd.DataFrame(main_rows)
    main.to_csv(out_dir / "最终主结果表.csv", index=False)

    seq = _best_seq_len_rows()
    seq.to_csv(out_dir / "序列长度消融表.csv", index=False)

    deep = _deep_ema_rows()
    deep.to_csv(out_dir / "深度模型EMA与多种子表.csv", index=False)

    nsntk = _nsntk_rows()
    nsntk.to_csv(out_dir / "时间衰减_分数平滑_稳定性表.csv", index=False)

    gcn_rows = []
    for name, root in [("静态行业图传播", INDUSTRY_GCN_ROOT), ("滚动相关图传播", ROLLING_CORR_GCN_ROOT)]:
        selected = pd.read_csv(root / "selected_by_valid_sharpe.csv")
        valid = selected[selected["selection"].eq("valid_best_by_topk10_drop2_sharpe")].iloc[0]
        test = selected[selected["selection"].eq("valid_selected_test_once")].iloc[0]
        gcn_rows.append(
            {
                "图实验": name,
                "valid选择alpha": valid["alpha_graph"],
                "valid夏普": valid["bt_sharpe"],
                "test IC": test["ic_mean"],
                "test ICIR": test["icir"],
                "test夏普": test["bt_sharpe"],
                "test最大回撤": test["bt_max_drawdown"],
                "test平均换手": test["bt_avg_turnover"],
                "来源": str(root),
            }
        )
    pd.DataFrame(gcn_rows).to_csv(out_dir / "GCN图实验表.csv", index=False)

    write_json(
        out_dir / "summary.json",
        {
            "main": str(out_dir / "最终主结果表.csv"),
            "seq_len": str(out_dir / "序列长度消融表.csv"),
            "deep_ema": str(out_dir / "深度模型EMA与多种子表.csv"),
            "nsntk": str(out_dir / "时间衰减_分数平滑_稳定性表.csv"),
            "gcn": str(out_dir / "GCN图实验表.csv"),
        },
    )
    print(json.dumps({"saved": str(out_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
