"""Pure data assembly for the Home landing — no Streamlit imports.

Feeds the hero/stat-row/latest-card/recent-grid; everything degrades to
zeros/None so a fresh install renders a sensible empty state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.project import list_project_index


def get_home_summary() -> dict[str, Any]:
    index = list_project_index()

    latest: dict[str, Any] | None = None
    if index:
        e = index[0]
        latest = {
            "id": e["id"],
            "name": e["name"],
            "updated_at": e.get("updated_at"),
            "task_type": e.get("task_type"),
            "n_stages_done": len(e.get("completed_stages") or []),
        }

    record_count = 0
    try:
        from core.acquire_bridge import get_acquire_status

        record_count = int(get_acquire_status().get("record_count") or 0)
    except Exception:
        pass  # acquire DB locked/missing — stat shows 0

    n_sources = 0
    try:
        import json

        reg = Path(__file__).resolve().parents[1] / "acquire" / "data" / "source_registry.json"
        data = json.loads(reg.read_text(encoding="utf-8"))
        n_sources = sum(len(v) for v in data.values() if isinstance(v, list))
    except Exception:
        pass  # registry missing/unreadable — stat shows 0

    return {
        "n_projects": len(index),
        "latest": latest,
        "recent": index[:6],
        "record_count": record_count,
        "n_sources": n_sources,
    }
