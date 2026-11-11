"""Tests for pre-analysis helpers and cleaning settings."""

from __future__ import annotations

import pandas as pd

from core.preanalysis import (
    column_summary,
    recommend_missing_strategy,
    missing_matrix_summary,
)
from core.stage_settings import get_stage_settings, save_stage_settings
from core.project import Project
from stages.clean import run as clean_run


def test_column_summary_includes_missing():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
    summary = column_summary(df)
    assert "missing_pct" in summary.columns
    assert summary.loc[summary["column"] == "a", "missing"].iloc[0] == 1


def test_recommend_mean_for_numeric_with_missing():
    assert recommend_missing_strategy("numeric", 5.0) == "median"


def test_clean_mean_imputation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = Project.create(name="pre", task_type="regression")
    df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [10.0, 20.0, 30.0]})
    project.set_raw_data(df)
    project.meta["target_column"] = "y"
    project.save_meta()

    settings = {
        "drop_columns": [],
        "missing_strategy": {"x": "mean"},
        "outlier_method": "none",
        "drop_duplicates": False,
        "type_overrides": {},
        "constant_value": "Unknown",
    }
    result = clean_run(project, df, settings)
    assert result.status == "success"
    assert result.output_df is not None
    assert result.output_df["x"].isna().sum() == 0
    assert abs(result.output_df["x"].iloc[1] - 2.0) < 0.01


def test_stage_settings_persist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = Project.create(name="settings", task_type="regression")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    project.set_raw_data(df)

    save_stage_settings(project, "clean", {"missing_strategy": {"a": "mean"}, "drop_columns": ["b"]})
    loaded = get_stage_settings(project, "clean")
    assert loaded["missing_strategy"]["a"] == "mean"
    assert "b" in loaded["drop_columns"]


def test_missing_matrix_summary():
    df = pd.DataFrame({"a": [1, None], "b": [1, 1]})
    stats = missing_matrix_summary(df)
    assert stats["columns_with_missing"] == 1
    assert stats["missing_cells"] == 1