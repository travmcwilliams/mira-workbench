"""Unified activity log: acquire scraper events + pipeline stage runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.acquire_bridge import load_scraper_run_log
from core.state import get_all_runs


def _normalize_acquire_events(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rows.append({
            "origin": "acquire",
            "at": entry.get("at", ""),
            "stage": entry.get("event_type", "scrape"),
            "source_id": entry.get("source_id", ""),
            "status": entry.get("event_type", ""),
            "detail": str(entry.get("detail", ""))[:200],
            "run_id": entry.get("run_id", ""),
        })
    return rows


def _normalize_pipeline_events(runs: list[dict[str, Any]], project_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.append({
            "origin": "pipeline",
            "at": run.get("run_at", ""),
            "stage": run.get("stage", ""),
            "source_id": project_id,
            "status": run.get("status", ""),
            "detail": (run.get("error_message") or "")[:200],
            "run_id": str(run.get("id", "")),
        })
    return rows


def load_acquire_activity(*, limit: int = 50) -> list[dict[str, Any]]:
    return _normalize_acquire_events(load_scraper_run_log(limit=limit))


def load_pipeline_activity(project_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    state_path = project_root / "state.db"
    if not state_path.exists():
        return []
    runs = get_all_runs(state_path)[:limit]
    project_id = project_root.name
    return _normalize_pipeline_events(runs, project_id)


def unified_activity_df(
    *,
    project_root: Path | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Merge acquire scraper log with optional project pipeline runs."""
    rows = load_acquire_activity(limit=limit)
    if project_root is not None:
        rows.extend(load_pipeline_activity(project_root, limit=limit))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "at" in df.columns:
        df = df.sort_values("at", ascending=False)
    return df.head(limit).reset_index(drop=True)