"""Load data into a MIRA project (upload, sample, or records)."""

from __future__ import annotations

import pandas as pd

from core.data_loader import detect_schema
from core.pipeline import Pipeline
from core.project import Project


def ingest_dataframe(
    project: Project,
    df: pd.DataFrame,
    *,
    source_name: str = "upload",
    auto_profile: bool = True,
) -> str | None:
    """
    Save dataframe as raw/working, detect schema, optionally run profile.
    Returns error message from profile stage, or None on success.
    """
    project.set_raw_data(df)
    schema = detect_schema(df)
    project.meta["schema"] = schema
    project.meta["data_source_name"] = source_name
    project.save_meta()

    _fire_new_data(project, source_name, len(df))

    if not auto_profile:
        return None

    try:
        pipeline = Pipeline(project)
        result = pipeline.run_stage("profile", {"minimal_threshold": 100})
        if result.status != "success":
            return result.error
    except Exception as exc:
        return str(exc)
    return None


def _fire_new_data(project: Project, source_name: str, rows: int) -> None:
    """Notify the CI/trigger layer that fresh data landed. Best-effort."""
    try:
        from core import ci  # noqa: PLC0415
    except ImportError:
        return
    try:
        ci.on_event(project, "new_data", {"source": source_name, "rows": rows})
    except Exception:
        pass