"""Tests for schema consolidation (core/webschema.py) — no network."""

from __future__ import annotations

import pandas as pd

from core import webschema as ws


def test_normalize_col():
    assert ws.normalize_col("  Total Revenue ($) ") == "total_revenue"
    assert ws.normalize_col("Region/Area") == "region_area"
    assert ws.normalize_col("") == "col"


def test_exact_normalized_columns_merge():
    a = pd.DataFrame({"Revenue": [1, 2], "Region": ["x", "y"]})
    b = pd.DataFrame({"revenue ": [3, 4], "region": ["z", "w"]})
    combined, report = ws.consolidate([a, b])
    # "Revenue"/"revenue " -> one column; likewise region
    assert "revenue" in combined.columns
    assert "region" in combined.columns
    assert len(combined) == 4
    rev = next(c for c in report.columns if c.name == "revenue")
    assert rev.sources == 2  # both frames contributed


def test_union_all_keeps_disjoint_columns():
    a = pd.DataFrame({"price": [10, 20]})
    b = pd.DataFrame({"weight": [1.0, 2.0]})
    combined, report = ws.consolidate([a, b], strategy="union_all")
    assert set(["price", "weight"]).issubset(set(combined.columns))
    assert report.n_sources == 2
    price = next(c for c in report.columns if c.name == "price")
    assert price.sources == 1  # only one frame had it


def test_coherent_default_drops_incoherent_sources():
    # Two schema-sharing frames + one unrelated -> coherent keeps the pair.
    a = pd.DataFrame({"state": ["A", "B"], "pop": [1, 2]})
    b = pd.DataFrame({"state": ["C", "D"], "pop": [3, 4]})
    junk = pd.DataFrame({"unrelated": [9], "other": [8]})
    combined, report = ws.consolidate([a, b, junk])
    assert report.strategy == "coherent"
    assert report.n_sources == 2            # a + b
    assert report.dropped_incoherent == 1   # junk excluded
    assert "unrelated" not in combined.columns
    assert len(report.clusters) >= 2        # alternatives reported for the UI


def test_manual_keep_indices_overrides_strategy():
    a = pd.DataFrame({"state": ["A"], "pop": [1]})
    junk = pd.DataFrame({"unrelated": [9]})
    combined, report = ws.consolidate([a, junk], keep_indices=[0, 1])
    assert report.strategy == "manual"
    assert report.n_sources == 2
    assert "unrelated" in combined.columns


def test_keep_columns_prunes_schema():
    a = pd.DataFrame({"state": ["A", "B"], "pop": [1, 2], "junk": [7, 8]})
    combined, _ = ws.consolidate([a], keep_columns=["state", "pop"])
    assert list(combined.columns) == ["state", "pop"]


def test_fuzzy_merge_near_duplicates():
    a = pd.DataFrame({"revenue": [1, 2]})
    b = pd.DataFrame({"revenues": [3, 4]})  # ratio ~93 -> merge under union_all
    combined, _ = ws.consolidate([a, b], strategy="union_all")
    assert "revenue" in combined.columns
    assert "revenues" not in combined.columns


def test_fuzzy_does_not_overmerge_substrings():
    # plain Levenshtein ratio('revenue','total_revenue') ~74 < 90 -> stay separate
    a = pd.DataFrame({"revenue": [1, 2]})
    b = pd.DataFrame({"total_revenue": [3, 4]})
    combined, _ = ws.consolidate([a, b], strategy="union_all")
    assert "revenue" in combined.columns
    assert "total_revenue" in combined.columns


def test_row_dedupe_across_coherent_frames():
    # Single shared column -> Jaccard 1.0 -> coherent -> cross-frame dedupe.
    a = pd.DataFrame({"x": [1, 1, 2]})
    b = pd.DataFrame({"x": [1, 2]})
    combined, report = ws.consolidate([a, b])
    assert report.n_sources == 2
    assert sorted(combined["x"].tolist()) == [1, 2]


def test_cluster_frames_groups_by_shared_schema():
    a = pd.DataFrame({"state": [1], "pop": [2]})
    b = pd.DataFrame({"state": [3], "pop": [4]})
    c = pd.DataFrame({"foo": [5], "bar": [6]})
    groups = ws.cluster_frames([a, b, c])
    # a+b together, c alone
    assert sorted(len(g) for g in groups) == [1, 2]


def test_empty_frames_report():
    combined, report = ws.consolidate([pd.DataFrame(), None])  # type: ignore[list-item]
    assert combined.empty
    assert report.n_rows == 0
    assert report.n_sources == 0
    assert report.dropped_empty == 2


def test_target_suggested():
    a = pd.DataFrame({"feature_a": [1, 2, 3, 4], "price": [10, 20, 30, 40]})
    _, report = ws.consolidate([a], task_type="regression")
    assert report.target_column is not None
    assert report.schema  # non-empty schema
