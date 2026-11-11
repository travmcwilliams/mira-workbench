"""
tests/test_dataset_analysis.py

Phase 9: dataset-analysis facts (preprocessing + analysis) and their flow into
context briefs and the sprint planner. Offline/deterministic.
Run with: pytest tests/test_dataset_analysis.py -q
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_projects_dir(monkeypatch, tmp_path):
    (tmp_path / "projects").mkdir()
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _messy_df():
    """A frame exercising missing, constant, id, leakage, and a numeric target."""
    n = 40
    price = pd.Series(np.arange(n, dtype=float) + 10.0)
    return pd.DataFrame({
        "id": range(n),                         # id-like
        "const": ["x"] * n,                     # constant
        "feat": np.linspace(0, 1, n),           # ordinary numeric
        "cat": (["a", "b", "c"] * n)[:n],       # categorical
        "mostly_null": [None] * (n - 2) + [1, 2],  # high-missing
        "leak": price * 2.0,                    # near-perfect corr with target
        "price": price,                         # regression target
    })


def _classif_df():
    n = 50
    y = ["yes"] * 45 + ["no"] * 5               # imbalanced
    return pd.DataFrame({
        "f1": np.arange(n, dtype=float),
        "label": y,
    })


def _project_with(df, *, target, task_type="regression"):
    from core.project import Project
    p = Project.create(name="DA Test", task_type=task_type)
    p.set_raw_data(df)
    from core.data_loader import detect_schema
    p.meta["schema"] = detect_schema(df)
    p.meta["target_column"] = target
    p.save_meta()
    return p


# --- Core analysis -----------------------------------------------------------

def test_analyze_detects_quality_flags_and_recommendations(temp_projects_dir):
    from core import dataset_analysis as da

    a = da.analyze_dataframe(_messy_df(), target="price", task_type="regression")
    assert a["available"] and a["rows"] == 40

    issues = {(f["column"], f["issue"]) for f in a["quality_flags"]}
    assert ("const", "constant") in issues
    assert ("mostly_null", "high_missing") in issues
    assert ("id", "id_like") in issues

    drops = {d["column"] for d in a["recommendations"]["drop"]}
    assert {"const", "id"} <= drops
    assert "mostly_null" in a["recommendations"]["impute"]  # numeric -> median/mean


def test_leakage_signal_flags_correlated_feature(temp_projects_dir):
    from core import dataset_analysis as da

    a = da.analyze_dataframe(_messy_df(), target="price", task_type="regression")
    leak_cols = {s["column"] for s in a["leakage"]}
    assert "leak" in leak_cols


def test_classification_target_imbalance(temp_projects_dir):
    from core import dataset_analysis as da

    a = da.analyze_dataframe(_classif_df(), target="label", task_type="classification")
    tgt = a["target"]
    assert tgt["kind"] == "classification"
    assert tgt["n_classes"] == 2
    assert tgt["imbalance_ratio"] == 9.0  # 45 / 5


def test_compact_summary_is_bounded_and_serializable(temp_projects_dir):
    import json

    from core import dataset_analysis as da

    a = da.analyze_dataframe(_messy_df(), target="price")
    c = da.compact_summary(a)
    json.dumps(c)  # must not raise on numpy types
    assert c["available"] and "recommend" in c
    assert isinstance(c["issues"], list)


# --- Project cache -----------------------------------------------------------

def test_analyze_project_caches_by_fingerprint(temp_projects_dir):
    from core import dataset_analysis as da

    p = _project_with(_messy_df(), target="price")
    a1 = da.analyze_project(p)
    assert a1["available"]
    assert (p.root / "artifacts" / "dataset_analysis.json").exists()
    assert p.meta["dataset_analysis"]["fingerprint"] == a1["fingerprint"]

    # Second call hits the cache (same fingerprint, same object content).
    a2 = da.analyze_project(p)
    assert a2["fingerprint"] == a1["fingerprint"]


def test_no_working_data_degrades(temp_projects_dir):
    from core import dataset_analysis as da
    from core.project import Project

    p = Project.create(name="empty")
    a = da.analyze_project(p)
    assert a["available"] is False


# --- Flow into context + planner ---------------------------------------------

def test_context_brief_includes_dataset_analysis(temp_projects_dir):
    from core import board, context

    p = _project_with(_messy_df(), target="price")
    board.add_task(p, "Clean the data", stage="clean")
    task = board.list_tasks(p)[0]

    ctx = context.assemble_context(p, task)
    assert "data.analysis" in ctx["resolved"]
    assert ctx["resolved"]["data.analysis"]["available"] is True

    brief = context.render_brief(ctx)
    assert "## Dataset analysis" in brief
    assert "possible leakage" in brief  # leak column surfaced


def test_planner_context_includes_dataset(temp_projects_dir):
    from core import sprint

    p = _project_with(_messy_df(), target="price")
    ctx = sprint.build_context(p)
    assert ctx["dataset"] is not None
    assert ctx["dataset"]["available"] is True


# --- CLI ---------------------------------------------------------------------

def test_cli_data_analyze(temp_projects_dir, capsys):
    from mira_cli import main

    p = _project_with(_messy_df(), target="price")
    rc = main(["data", "analyze", "--project-id", p.id])
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Dataset analysis" in out
    assert "recommend drop" in out
