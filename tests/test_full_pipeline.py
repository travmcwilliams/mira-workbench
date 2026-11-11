"""Integration tests for headless pipeline with stage settings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.data_loader import detect_schema
from core.pipeline import STAGE_ORDER, Pipeline
from core.project import Project
from core.stage_settings import get_stage_settings, save_stage_settings


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "feature_a": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        "feature_b": ["x", "y", "x", "z", "y", "x", "z", "y", "x", "z",
                      "y", "x", "z", "y", "x", "z", "y", "x", "z", "y"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_pipeline_with_saved_clean_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = Project.create(name="pipe-test", task_type="classification")
    df = _sample_df()
    project.set_raw_data(df)
    project.meta["schema"] = detect_schema(df)
    project.meta["target_column"] = "target"
    project.save_meta()

    save_stage_settings(project, "clean", {
        "drop_columns": [],
        "missing_strategy": {"feature_a": "mean"},
        "outlier_method": "none",
        "drop_duplicates": False,
        "type_overrides": {},
        "constant_value": "Unknown",
    })

    save_stage_settings(project, "feature_select", {
        "methods": ["variance"],
        "correlation_threshold": 0.99,
        "variance_threshold": 0.0,
        "rfe_n_features": 10,
        "importance_top_k": 20,
    })

    pipe = Pipeline(project)
    for stage in ("profile", "clean"):
        settings = get_stage_settings(project, stage)
        res = pipe.run_stage(stage, settings)
        assert res.status == "success", f"{stage}: {res.error}"

    assert project.get_working_data()["feature_a"].isna().sum() == 0

    for stage in ("feature_eng", "feature_select"):
        settings = get_stage_settings(project, stage)
        res = pipe.run_stage(stage, settings)
        assert res.status == "success", f"{stage}: {res.error}"

    assert "target" in project.get_working_data().columns


def test_stage_order_includes_acquire():
    assert "acquire" in STAGE_ORDER
    assert STAGE_ORDER.index("profile") < STAGE_ORDER.index("model")