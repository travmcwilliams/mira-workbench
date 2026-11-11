"""Tests for unified activity log."""

from __future__ import annotations

from core.activity_log import load_acquire_activity, unified_activity_df


def test_load_acquire_activity_returns_list():
    rows = load_acquire_activity(limit=5)
    assert isinstance(rows, list)


def test_unified_activity_df_shape():
    df = unified_activity_df(limit=10)
    assert hasattr(df, "empty")
    if not df.empty:
        assert "origin" in df.columns
        assert "stage" in df.columns