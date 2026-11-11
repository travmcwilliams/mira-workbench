"""Smoke test: REAL001 ingest → records → acquire stage → profile."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("APP_PROJECT_ROOT", str(ROOT / "acquire"))
os.environ.setdefault("APP_DB_PATH", str(ROOT / "acquire" / "db" / "app.duckdb"))


def main() -> None:
    from core.acquire_bridge import ingest_source_id, load_records_df
    from core.data_loader import detect_schema
    from core.pipeline import Pipeline
    from core.project import Project

    source_id = "REAL001"
    print("=== MIRA smoke: acquire → records → profile ===")

    print("1. Ingest registry source → records …")
    summary = ingest_source_id(source_id)
    print(f"   rows loaded: {summary['rows_loaded']}, records: {summary['records_after']}")

    records = load_records_df(source_id=source_id)
    assert not records.empty, "records empty after ingest"
    print(f"   records query: {len(records)} rows, cols={list(records.columns)}")

    scratch = ROOT / "projects" / "_smoke_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    prev_cwd = Path.cwd()
    try:
        os.chdir(scratch)
        print("2. Project + acquire stage …")
        project = Project.create(name="smoke-acquire", task_type="regression")
        project.meta["acquire_source_id"] = source_id
        project.meta["target_column"] = "primary_value"
        project.save_meta()

        pipe = Pipeline(project)
        acq = pipe.run_stage(
            "acquire",
            {"source_id": source_id, "use_existing_records": True},
        )
        assert acq.status == "success", acq.error
        print(f"   acquire OK: {acq.summary}")

        prof = pipe.run_stage("profile", {"minimal_threshold": 100})
        assert prof.status == "success", prof.error
        print("   profile OK")

        working = project.get_working_data()
        assert "primary_value" in working.columns
        print(f"3. Working data: {len(working)} rows × {len(working.columns)} cols")
        print("=== PASS ===")
    finally:
        os.chdir(prev_cwd)
        from utils.db import close_db

        close_db()


if __name__ == "__main__":
    main()