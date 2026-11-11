"""Tests for dataset profile detection used by interactive dashboards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.data_loader import load_file
from core.dataset_profiles import detect_dataset_profile, profile_label


def test_detect_restaurant_daily_profile():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "sample" / "restaurant_daily_sample.csv"
    if not path.exists():
        return
    df = load_file(path)
    assert detect_dataset_profile(df) == "restaurant_daily"
    assert "Restaurant" in profile_label("restaurant_daily")


def test_detect_employee_labor_profile():
    root = Path(__file__).resolve().parents[1]
    df = load_file(root / "data" / "sample" / "restaurant_labor_sample.csv")
    assert detect_dataset_profile(df) == "employee_labor"


def test_detect_generic_profile():
    root = Path(__file__).resolve().parents[1]
    df = load_file(root / "data" / "sample" / "boston_sample.csv")
    assert detect_dataset_profile(df) == "generic"


def test_detect_canonical_records_from_acquire_bridge():
    from core.acquire_bridge import load_records_df

    df = load_records_df(limit=5)
    if df.empty:
        return
    assert detect_dataset_profile(df) == "canonical_records"