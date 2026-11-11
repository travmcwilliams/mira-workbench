"""Tests for MIRA acquire bridge."""

from __future__ import annotations

import pytest

pytest.importorskip("dotenv")
pytest.importorskip("duckdb")

from core.acquire_bridge import get_acquire_status, load_records_df, load_registry_df


def test_acquire_status():
    status = get_acquire_status()
    assert "record_count" in status
    assert status["record_count"] >= 0


def test_registry_loads():
    df = load_registry_df()
    assert "source_id" in df.columns
    assert len(df) >= 1


def test_records_query():
    df = load_records_df(limit=10)
    assert isinstance(df.columns.tolist(), list)