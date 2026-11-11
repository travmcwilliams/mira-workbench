"""Tests for server performance analytics."""

from __future__ import annotations

from pathlib import Path

from core.data_loader import load_file
from core.server_analytics import (
    prepare_server_cohort,
    server_leaderboard,
    top_vs_bottom_summary,
)


def _sample_df():
    root = Path(__file__).resolve().parents[1]
    return load_file(root / "data" / "sample" / "restaurant_labor_sample.csv")


def test_prepare_server_cohort_filters_role():
    servers, meta = prepare_server_cohort(_sample_df(), min_hours=0)
    assert meta["server_count"] > 0
    assert "tips_per_hour" in servers.columns
    assert "tips_per_hour_pct_above_median" in servers.columns
    assert "tips_per_hour_percentile" in servers.columns


def test_leaderboard_ranks_by_metric():
    servers, _ = prepare_server_cohort(_sample_df(), min_hours=20)
    board = server_leaderboard(servers, metric="tips_per_hour")
    assert not board.empty
    assert board.iloc[0]["tips_per_hour"] >= board.iloc[-1]["tips_per_hour"]


def test_top_vs_bottom_summary():
    servers, _ = prepare_server_cohort(_sample_df(), min_hours=20)
    summary = top_vs_bottom_summary(servers, metric="tips_per_hour", n=2)
    assert summary["cohort_median"] > 0
    assert len(summary["top_names"]) == 2
    assert summary["spread_top_vs_bottom"] >= 0