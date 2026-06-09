from __future__ import annotations

import pandas as pd

from src.pipelines.run_strategy_risk_sweep import add_risk_return_score, pareto_frontier, resolve_risk_profiles, select_valid_risk_return


def test_resolve_risk_profiles_returns_named_overrides() -> None:
    profiles = resolve_risk_profiles(["none", "combined_mild"])

    assert [profile["name"] for profile in profiles] == ["none", "combined_mild"]
    assert profiles[0]["overrides"] == {}
    assert profiles[1]["overrides"]["apply_market_stress_deleveraging"] is True
    assert profiles[1]["overrides"]["apply_drawdown_control"] is True


def test_add_risk_return_score_penalizes_drawdown_violation() -> None:
    metrics = pd.DataFrame(
        [
            {
                "split": "valid",
                "model": "m1",
                "variant": "a",
                "risk_profile": "none",
                "sharpe": 2.0,
                "annual_return": 1.0,
                "total_return": 1.0,
                "max_drawdown": -0.20,
                "avg_turnover": 0.4,
                "avg_gross_exposure": 0.8,
            },
            {
                "split": "valid",
                "model": "m1",
                "variant": "b",
                "risk_profile": "none",
                "sharpe": 2.0,
                "annual_return": 1.0,
                "total_return": 1.0,
                "max_drawdown": -0.35,
                "avg_turnover": 0.4,
                "avg_gross_exposure": 0.8,
            },
        ]
    )

    scored = add_risk_return_score(metrics, max_drawdown_limit=-0.25)

    assert bool(scored.loc[0, "passes_drawdown_limit"])
    assert not bool(scored.loc[1, "passes_drawdown_limit"])
    assert scored.loc[0, "risk_return_score"] > scored.loc[1, "risk_return_score"]


def test_select_valid_risk_return_attaches_matching_test_rows() -> None:
    metrics = pd.DataFrame(
        [
            {
                "split": "valid",
                "model": "m1",
                "variant": "a",
                "risk_profile": "none",
                "risk_profile_label": "无风控",
                "strategy": "s",
                "sharpe": 2.0,
                "annual_return": 1.0,
                "final_equity": 2.0,
                "total_return": 1.0,
                "max_drawdown": -0.20,
                "avg_turnover": 0.4,
                "avg_gross_exposure": 0.8,
                "market_stress_days": 0,
                "drawdown_control_days": 0,
            },
            {
                "split": "test",
                "model": "m1",
                "variant": "a",
                "risk_profile": "none",
                "risk_profile_label": "无风控",
                "strategy": "s",
                "sharpe": 1.5,
                "annual_return": 0.8,
                "final_equity": 1.8,
                "total_return": 0.8,
                "max_drawdown": -0.18,
                "avg_turnover": 0.4,
                "avg_gross_exposure": 0.8,
                "market_stress_days": 0,
                "drawdown_control_days": 0,
            },
        ]
    )

    selected = select_valid_risk_return(metrics)

    assert selected.loc[0, "model"] == "m1"
    assert selected.loc[0, "risk_profile"] == "none"
    assert selected.loc[0, "sharpe_valid"] == 2.0
    assert selected.loc[0, "sharpe_test"] == 1.5


def test_pareto_frontier_removes_dominated_rows() -> None:
    metrics = pd.DataFrame(
        [
            {"split": "valid", "model": "m1", "variant": "a", "risk_profile": "p", "sharpe": 2.0, "annual_return": 1.0, "total_return": 1.0, "max_drawdown": -0.20, "avg_turnover": 0.4},
            {"split": "valid", "model": "m1", "variant": "b", "risk_profile": "p", "sharpe": 1.5, "annual_return": 0.6, "total_return": 0.6, "max_drawdown": -0.25, "avg_turnover": 0.4},
            {"split": "valid", "model": "m1", "variant": "c", "risk_profile": "p", "sharpe": 1.2, "annual_return": 1.2, "total_return": 1.2, "max_drawdown": -0.35, "avg_turnover": 0.4},
        ]
    )

    frontier = pareto_frontier(metrics)

    assert set(frontier["variant"]) == {"a", "c"}
