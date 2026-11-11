"""Pipeline acquire stage smoke tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("APP_PROJECT_ROOT", str(ROOT / "acquire"))
os.environ.setdefault("APP_DB_PATH", str(ROOT / "acquire" / "db" / "app.duckdb"))

pytest.importorskip("dotenv", reason="python-dotenv required for acquire layer")
pytest.importorskip("duckdb")


def test_acquire_stage_from_existing_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from core.acquire_bridge import ingest_source_id, load_records_df
    from core.pipeline import Pipeline
    from core.project import Project

    ingest_source_id("REAL001")
    assert not load_records_df(source_id="REAL001").empty

    project = Project.create(name="acquire-test", task_type="regression")
    project.meta["acquire_source_id"] = "REAL001"
    project.meta["target_column"] = "primary_value"
    project.save_meta()

    pipe = Pipeline(project)
    result = pipe.run_stage(
        "acquire",
        {"source_id": "REAL001", "use_existing_records": True},
    )
    assert result.status == "success", result.error
    working = project.get_working_data()
    assert len(working) > 0
    assert "primary_value" in working.columns