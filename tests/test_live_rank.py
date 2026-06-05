from __future__ import annotations

import pandas as pd

from src.pipelines.live_rank import (
    default_trade_date,
    load_watchlist,
    read_loose_raw,
    resolve_live_artifacts,
)


def test_default_trade_date_is_next_calendar_day() -> None:
    assert default_trade_date("20260603") == "20260604"


def test_load_watchlist_accepts_name_alias(tmp_path) -> None:
    path = tmp_path / "watchlist.csv"
    pd.DataFrame({"name": ["A", "B", "B2"], "ts_code": ["000001.SZ", "000002.SZ", "000002.SZ"]}).to_csv(path, index=False)

    out = load_watchlist(path)

    assert out.to_dict(orient="records") == [
        {"stock_name": "A", "ts_code": "000001.SZ"},
        {"stock_name": "B", "ts_code": "000002.SZ"},
    ]


def test_read_loose_raw_returns_empty_parts_when_optional_files_are_missing(tmp_path) -> None:
    daily, metric, moneyflow, st, open_frames = read_loose_raw(tmp_path, "20260603", "20260604")

    assert daily == []
    assert metric == []
    assert moneyflow == []
    assert st == []
    assert open_frames == []


def test_read_loose_raw_scans_historical_csvs_and_trade_open(tmp_path) -> None:
    pd.DataFrame({"trade_date": ["20260603"], "ts_code": ["000001.SZ"], "vol": [1], "amount": [2]}).to_csv(
        tmp_path / "daily 20260603.csv", index=False
    )
    pd.DataFrame({"trade_date": ["20260604"], "ts_code": ["000001.SZ"], "vol": [3], "amount": [4]}).to_csv(
        tmp_path / "daily 20260604.csv", index=False
    )
    pd.DataFrame({"trade_date": ["20260605"], "ts_code": ["000001.SZ"], "vol": [5], "amount": [6]}).to_csv(
        tmp_path / "daily 20260605.csv", index=False
    )
    pd.DataFrame({"trade_date": ["20260605"], "ts_code": ["000001.SZ"], "open": [10.0], "pre_close": [9.9]}).to_csv(
        tmp_path / "daily open 20260605.csv", index=False
    )

    daily, metric, moneyflow, st, open_frames = read_loose_raw(tmp_path, "20260604", "20260605")

    assert [frame["trade_date"].iloc[0] for frame in daily] == ["20260603", "20260604"]
    assert metric == []
    assert moneyflow == []
    assert st == []
    assert len(open_frames) == 1
    assert open_frames[0]["trade_date"].iloc[0] == "20260605"


def test_resolve_live_artifacts_reads_registry(tmp_path) -> None:
    registry = tmp_path / "models.yaml"
    registry.write_text(
        """
artifacts:
  final_live:
    lgb_model: lgb.txt
    xgb_model: xgb.json
    fusion_model: fusion.pt
""",
        encoding="utf-8",
    )

    assert resolve_live_artifacts(registry, "final_live") == {
        "lgb_model": "lgb.txt",
        "xgb_model": "xgb.json",
        "fusion_model": "fusion.pt",
    }
