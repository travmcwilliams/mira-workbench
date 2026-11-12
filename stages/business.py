"""
stages/business.py

Problem framing stage (text only for now).

This is the natural place for single-restaurant context, e.g.:
- Goal: Understand and optimize labor cost % at a full-service restaurant
- Role mix, tipped vs support, tip pool rules, sales attribution, daypart effects, etc.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.pipeline import StageResult
from core.project import Project


def default_settings(project: Project) -> dict[str, Any]:
    return {"notes": ""}


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    """No-op data stage; just records framing notes into project meta."""
    notes = settings.get("notes", "") if settings else ""
    if notes:
        project.meta.setdefault("business", {})["notes"] = notes
        project.save_meta()
    return StageResult(
        status="success",
        output_df=None,
        summary={"notes_length": len(notes) if notes else 0},
    )


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    import streamlit as st

    current = project.meta.get("business", {}).get("notes", "")
    settings["notes"] = st.text_area(
        "Problem statement, goals, success criteria",
        value=settings.get("notes") or current,
        height=120,
    )
    return settings
