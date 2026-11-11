"""
stages/acquire.py

Stage 0: Multi-source ingest → canonical `records` table (DuckDB).
Hands the records snapshot to the project working dataset for downstream DS.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.acquire_bridge import ingest_source_id, load_records_df
from core.pipeline import StageResult
from core.project import Project


def default_settings(project: Project) -> dict[str, Any]:
    return {
        "source_id": project.meta.get("acquire_source_id"),
        "dry_run": False,
        "use_existing_records": False,
    }


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    source_id = settings.get("source_id") or project.meta.get("acquire_source_id")
    if not source_id:
        # No acquire source configured. For file/upload-based projects the data is
        # already present, so acquire is a no-op pass-through rather than a failure.
        if df is not None and not df.empty:
            return StageResult(
                status="success",
                output_df=df,
                summary={"source_id": None, "skipped": True, "reason": "no acquire source_id; using existing data"},
            )
        return StageResult(
            status="error",
            error="No acquire source_id. Set project.meta['acquire_source_id'] or pass settings.",
        )

    ingest_summary: dict[str, Any] = {}
    if not settings.get("use_existing_records"):
        try:
            ingest_summary = ingest_source_id(
                str(source_id),
                dry_run=bool(settings.get("dry_run")),
            )
        except Exception as exc:
            return StageResult(status="error", error=f"Acquire ingest failed: {exc}")

    try:
        out_df = load_records_df(source_id=str(source_id))
    except Exception as exc:
        return StageResult(status="error", error=f"Failed to read records: {exc}")

    if out_df.empty:
        return StageResult(
            status="error",
            error=f"No rows in records for source_id={source_id}.",
        )

    project.meta["acquire_source_id"] = source_id
    project.save_meta()

    return StageResult(
        status="success",
        output_df=out_df,
        summary={
            "source_id": source_id,
            "rows_in_working": len(out_df),
            "ingest": ingest_summary,
            "table": "records",
        },
    )


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    return settings


def code_template(settings: dict[str, Any]) -> str:
    sid = settings.get("source_id", "SOURCE_ID")
    return f"""# Acquire → records
# source_id = {sid!r}
print("Ingest complete; query records table for downstream DS.")
"""