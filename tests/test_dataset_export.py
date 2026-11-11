"""Tests for core/dataset_export.py — multi-format dataset serialization."""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from core import dataset_export
from core.dataset_export import FORMATS, export_bytes, export_filename, get_format


@pytest.fixture()
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "close": [185.5, 402.25, 141.0],
            "volume": [1_000, 2_000, 3_000],
        }
    )


def test_format_registry_is_consistent():
    keys = [f.key for f in FORMATS]
    assert len(keys) == len(set(keys))
    for f in FORMATS:
        assert get_format(f.key) is f
        assert f.extension and f.mime and f.label


def test_unknown_format_raises(df):
    with pytest.raises(ValueError, match="Unknown export format"):
        export_bytes(df, "xml")
    with pytest.raises(ValueError):
        get_format("nope")


def test_csv_and_tsv_round_trip(df):
    out = pd.read_csv(io.BytesIO(export_bytes(df, "csv")), parse_dates=["date"])
    pd.testing.assert_frame_equal(out, df)
    out_t = pd.read_csv(io.BytesIO(export_bytes(df, "tsv")), sep="\t", parse_dates=["date"])
    pd.testing.assert_frame_equal(out_t, df)


def test_parquet_round_trip(df):
    out = pd.read_parquet(io.BytesIO(export_bytes(df, "parquet")))
    pd.testing.assert_frame_equal(out, df)


def test_xlsx_round_trip(df):
    out = pd.read_excel(io.BytesIO(export_bytes(df, "xlsx")))
    assert list(out.columns) == list(df.columns)
    assert len(out) == len(df)
    assert out["ticker"].tolist() == df["ticker"].tolist()


def test_xlsx_row_limit_guard(df, monkeypatch):
    monkeypatch.setattr(dataset_export, "EXCEL_MAX_ROWS", 2)
    with pytest.raises(ValueError, match="Excel supports at most"):
        export_bytes(df, "xlsx")


def test_json_records(df):
    records = json.loads(export_bytes(df, "json"))
    assert len(records) == 3
    assert records[0]["ticker"] == "AAPL"
    assert records[0]["date"].startswith("2026-01-01")


def test_jsonl_one_record_per_line(df):
    lines = export_bytes(df, "jsonl").decode("utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[2])["ticker"] == "GOOG"


def test_export_filename_slugs_project_name():
    assert export_filename("My Project! (v2)", "working", "csv") == "My_Project_v2_working.csv"
    assert export_filename("///", "raw", "parquet") == "dataset_raw.parquet"
    assert export_filename("panel", "working", "xlsx") == "panel_working.xlsx"
