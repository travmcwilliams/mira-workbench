"""Read pipeline stage run history from a project's state.db."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.state import get_all_runs, get_last_run


def get_last_stage_run(project_root: Path, stage: str) -> dict[str, Any] | None:
    row = get_last_run(project_root / "state.db", stage)
    if row is None:
        return None
    return _normalize_run(row)


def get_stage_history(project_root: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    runs = get_all_runs(project_root / "state.db")[:limit]
    return [_normalize_run(r) for r in runs]


def get_recent_for_sidebar(project_root: Path, *, limit: int = 8) -> list[dict[str, Any]]:
    history = get_stage_history(project_root, limit=limit)
    return [
        {
            "stage": h["stage"],
            "status": h["status"],
            "at": h["run_at"],
            "ms": h["duration_ms"],
        }
        for h in history
    ]


def _normalize_run(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("settings_json", "artifacts_json", "summary_json"):
        if key in out and isinstance(out[key], str):
            try:
                out[key.replace("_json", "")] = json.loads(out[key])
            except Exception:
                out[key.replace("_json", "")] = {}
    return out