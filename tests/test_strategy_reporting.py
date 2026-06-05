from __future__ import annotations

import pandas as pd

from src.strategy.reporting import load_existing_aggregate_outputs, write_report_artifacts


def _curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103", "20240104"],
            "gross_return": [0.0, 0.02, -0.01],
            "net_return": [0.0, 0.019, -0.011],
            "turnover": [0.0, 0.4, 0.2],
            "equity": [1.0, 1.019, 1.007791],
            "n_holdings": [20, 20, 20],
        }
    )


def test_write_report_artifacts_creates_standard_tables_and_html(tmp_path) -> None:
    rows_by_split = {
        "valid": [
            {
                "name": "label1d_lgb__topk20_drop3",
                "model": "label1d_lgb",
                "split": "valid",
                "strategy": "topk_drop",
                "total_return": 0.3,
                "annual_return": 0.4,
                "sharpe": 1.8,
                "max_drawdown": -0.08,
                "avg_turnover": 0.5,
                "avg_n_holdings": 20,
            },
            {
                "name": "benchmark_equal_weight_universe",
                "model": "label1d_lgb",
                "split": "valid",
                "strategy": "benchmark_equal_weight_universe",
                "total_return": 0.1,
                "annual_return": 0.1,
                "sharpe": 0.7,
                "max_drawdown": -0.05,
                "avg_turnover": 0.0,
                "avg_n_holdings": 4000,
            },
        ]
    }
    curves_by_split = {
        "valid": {
            "label1d_lgb__topk20_drop3": _curve(),
            "benchmark_equal_weight_universe": _curve(),
        }
    }

    paths = write_report_artifacts(tmp_path, rows_by_split, curves_by_split, benchmark_note="demo")

    metrics = pd.read_csv(paths["metrics_long"])
    equity = pd.read_parquet(paths["equity_long"])
    html = (tmp_path / "report.html").read_text(encoding="utf-8")

    assert {"model", "variant", "is_benchmark", "display_name"}.issubset(metrics.columns)
    assert metrics.loc[metrics["name"] == "label1d_lgb__topk20_drop3", "variant"].iloc[0] == "topk20_drop3"
    assert metrics.loc[metrics["name"] == "benchmark_equal_weight_universe", "model"].iloc[0] == "benchmark"
    assert {"drawdown", "equity", "variant"}.issubset(equity.columns)
    assert "策略回测报告" in html
    assert "label1d_lgb / topk20_drop3" in html


def test_load_existing_aggregate_outputs_rebuilds_curves_from_run_tree(tmp_path) -> None:
    run = tmp_path / "run"
    (run / "valid").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "name": "label1d_lgb__topk20_drop3",
                "model": "label1d_lgb",
                "split": "valid",
                "strategy": "topk_drop",
                "total_return": 0.3,
                "annual_return": 0.4,
                "sharpe": 1.8,
                "max_drawdown": -0.08,
                "avg_turnover": 0.5,
                "avg_n_holdings": 20,
            }
        ]
    ).to_csv(run / "valid" / "strategy_metrics.csv", index=False)
    curve_dir = run / "label1d_lgb" / "valid" / "topk20_drop3"
    curve_dir.mkdir(parents=True)
    _curve().to_csv(curve_dir / "equity_curve.csv", index=False)

    rows_by_split, curves_by_split = load_existing_aggregate_outputs(run)

    assert rows_by_split["valid"][0]["name"] == "label1d_lgb__topk20_drop3"
    assert list(curves_by_split["valid"]) == ["label1d_lgb__topk20_drop3"]
