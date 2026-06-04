from __future__ import annotations

from pathlib import Path

from src.pipelines.normalize_outputs import discover_moves, normalize_live_dir_name, normalize_run_dir_name
from src.pipelines.run_strategy_backtest import load_model_registry, resolve_feature_set, resolve_prediction_path
from src.utils import format_run_dir_name, make_run_dir, slugify_run_name


def test_run_dir_helpers_use_timestamp_prefix_and_sanitized_names(tmp_path) -> None:
    assert slugify_run_name(" final handoff / alpha=1.5 ") == "final_handoff_alpha_1.5"
    assert format_run_dir_name("final handoff", timestamp="20260604_115959") == "20260604_115959__final_handoff"
    assert make_run_dir(tmp_path, "final handoff", timestamped=False) == tmp_path / "final_handoff"


def test_strategy_backtest_model_registry_resolves_predictions_and_features(tmp_path) -> None:
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(
        """
models:
  demo:
    predictions:
      valid: outputs/models/demo/valid.parquet
      test: outputs/models/demo/test.parquet
feature_sets:
  risk_default:
    path: data/processed/features.parquet
    columns:
      - log_total_mv__cs_rank
      - turnover_rate__cs_rank
""",
        encoding="utf-8",
    )

    registry = load_model_registry(registry_path)

    assert resolve_prediction_path(registry, "demo", "test") == "outputs/models/demo/test.parquet"
    assert resolve_feature_set(registry, "risk_default") == (
        "data/processed/features.parquet",
        ["log_total_mv__cs_rank", "turnover_rate__cs_rank"],
    )


def test_normalize_live_dir_name_converts_legacy_patterns() -> None:
    assert normalize_live_dir_name("rolling_p10_h5_20260604_from_20260603") == "20260604__rolling_p10_h5__from_20260603"
    assert (
        normalize_live_dir_name("_check_rolling_p10_h5_20260603_from_20260602_boolfix")
        == "20260603__check_rolling_p10_h5__from_20260602__boolfix"
    )
    assert normalize_live_dir_name("20260604__final__from_20260603") is None


def test_normalize_run_dir_name_puts_timestamp_first(tmp_path) -> None:
    old = tmp_path / "unified_final_20260601_001207"
    old.mkdir()
    current = tmp_path / "20260601_001207__unified_final"
    current.mkdir()

    assert normalize_run_dir_name(old) == "20260601_001207__unified_final"
    assert normalize_run_dir_name(current) is None


def test_discover_moves_routes_legacy_outputs_to_canonical_roots(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    (outputs / "gbdt_full").mkdir(parents=True)
    (outputs / "strategy" / "unified_final_20260601_001207").mkdir(parents=True)
    (outputs / "live" / "rolling_p10_h5_20260604_from_20260603").mkdir(parents=True)

    moves = discover_moves(outputs)
    pairs = {(Path(move.src).name, Path(move.dst).as_posix()) for move in moves}

    assert ("gbdt_full", (outputs / "models" / "gbdt_full").as_posix()) in pairs
    assert (
        "rolling_p10_h5_20260604_from_20260603",
        (outputs / "live" / "20260604__rolling_p10_h5__from_20260603").as_posix(),
    ) in pairs
    assert (
        "unified_final_20260601_001207",
        (outputs / "strategy" / "20260601_001207__unified_final").as_posix(),
    ) in pairs
