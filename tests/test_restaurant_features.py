"""Tests for original POS labor feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.data_loader import load_file, suggest_target
from core.restaurant_features import engineer_labor_features, is_restaurant_labor_export
from core.pipeline import Pipeline
from core.project import Project


def _sample_df() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    return load_file(root / "data" / "sample" / "restaurant_labor_sample.csv")


def test_detects_labor_export():
    assert is_restaurant_labor_export(_sample_df())


def test_engineers_productivity_features():
    df = _sample_df()
    out, summary = engineer_labor_features(df, target="Total Tips")
    assert summary["applied"]
    assert "total_hours" in out.columns
    assert "tips_per_hour" not in out.columns  # leakage drop when target is tips
    assert "sales_per_hour" in out.columns
    assert "Last Name" not in out.columns
    assert "Reg Hrs" not in out.columns
    assert "role_family" in out.columns
    assert "is_tipped_role" in out.columns
    assert "implied_shifts" in out.columns
    assert "n_roles" in out.columns
    assert summary["peer_benchmarks"] is True


def test_peer_benchmark_features():
    df = _sample_df()
    out, summary = engineer_labor_features(df, target=None, add_peer_benchmarks=True)
    assert "tips_per_hour_vs_role_median" in out.columns
    assert "sales_vs_role_median" in out.columns
    assert summary["peer_benchmarks"]


def test_role_mix_for_multi_role_employee():
    df = _sample_df()
    out, _ = engineer_labor_features(df, target=None, aggregate_employee=False)
    multi = out[out["employee_key"].str.contains("Rivera", na=False)]
    assert (multi["is_multi_role"] == 1).any()
    assert (multi["n_roles"] >= 3).any()
    assert (multi["role_hours_share"] > 0).any()


def test_employee_aggregation():
    df = _sample_df()
    out, summary = engineer_labor_features(
        df, target="Total Tips", aggregate_employee=True, add_peer_benchmarks=True
    )
    assert summary["aggregated_employee"]
    assert len(out) < len(df)
    assert "employee_key" in out.columns
    assert "primary_role" in out.columns
    assert "n_roles" in out.columns
    assert "hours_vs_store_median" in out.columns
    assert "Total Tips" in out.columns
    assert "tips_per_hour" not in out.columns
    assert "tips_vs_store_median" not in out.columns  # leakage guard


def test_suggest_total_tips_for_labor_export():
    df = _sample_df()
    suggestion = suggest_target(df, task_type="regression")
    assert suggestion["target_column"] == "Total Tips"


def test_pipeline_feature_eng_on_original_data(tmp_path, monkeypatch):
    import os

    os.chdir(tmp_path)
    df = _sample_df()
    project = Project.create(name="Labor FE pipeline", task_type="regression")
    project.set_raw_data(df)
    from core.data_loader import detect_schema

    project.meta["schema"] = detect_schema(df)
    project.meta["target_column"] = "Total Tips"
    project.save_meta()

    pipeline = Pipeline(project)
    fe_result = None
    for stage in ("clean", "feature_eng", "feature_select"):
        result = pipeline.run_stage(stage)
        assert result.status == "success", result.error
        if stage == "feature_eng":
            fe_result = result

    working = project.get_working_data()
    assert "Total Tips" in working.columns
    assert "total_hours" in working.columns or "implied_shifts" in working.columns
    assert len(working) < len(df)  # employee aggregation
    rf = (fe_result.summary or {}).get("restaurant_features", {})
    assert rf.get("aggregated_employee") is True