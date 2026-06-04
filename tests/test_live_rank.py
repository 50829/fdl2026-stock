from __future__ import annotations

import pandas as pd

from src.pipelines.live_rank import default_trade_date, first_existing, load_watchlist, read_loose_raw


def test_default_trade_date_is_next_calendar_day() -> None:
    assert default_trade_date("20260603") == "20260604"


def test_first_existing_returns_first_existing_path(tmp_path) -> None:
    missing = tmp_path / "missing.csv"
    existing = tmp_path / "existing.csv"
    later = tmp_path / "later.csv"
    existing.write_text("x\n", encoding="utf-8")
    later.write_text("x\n", encoding="utf-8")

    assert first_existing([missing, existing, later]) == existing


def test_load_watchlist_accepts_name_alias(tmp_path) -> None:
    path = tmp_path / "watchlist.csv"
    pd.DataFrame({"name": ["A", "B", "B2"], "ts_code": ["000001.SZ", "000002.SZ", "000002.SZ"]}).to_csv(path, index=False)

    out = load_watchlist(path)

    assert out.to_dict(orient="records") == [
        {"stock_name": "A", "ts_code": "000001.SZ"},
        {"stock_name": "B", "ts_code": "000002.SZ"},
    ]


def test_read_loose_raw_returns_empty_parts_when_optional_files_are_missing(tmp_path) -> None:
    daily, metric, moneyflow, st, open_df = read_loose_raw(tmp_path, "20260603", "20260604")

    assert daily == []
    assert metric == []
    assert moneyflow == []
    assert st == []
    assert open_df is None
