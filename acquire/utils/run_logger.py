"""
utils/run_logger.py
Robust run metadata and event logging for the DSI platform.

Generalized from production patterns in the reference CountyIQ system.

Features:
- Per-source run tracking with events.
- Persists to data/scraper_run_log.json (compatible with existing simple format + richer data).
- Event types for fetch, parse, bot blocks, review, dedup, etc.
- start_run / log_event / finish_run lifecycle.

Usage in agents:
    from utils.run_logger import get_run_logger
    logger = get_run_logger()
    run_id = logger.start_run(source_id, source_name)
    logger.log_event(run_id, "fetch_ok", {"url": "..."})
    ...
    logger.finish_run(run_id, rows_loaded=42, errors=0)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .db import PROJECT_ROOT

RUN_LOG_PATH = PROJECT_ROOT / "data" / "scraper_run_log.json"

EVENT_TYPES = frozenset([
    "fetch_ok", "fetch_fail",
    "parse_ok", "parse_fail",
    "records_loaded", "duplicate_removed",
    "blocked_by_bot", "manual_review_required",
    "download_ok", "download_fail",
    "extract_ok", "extract_fail",
])


def _ensure_path() -> Path:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return RUN_LOG_PATH


class RunLogger:
    """Singleton-style run logger with rich events. Safe for the embedded workflow."""

    _instance: Optional["RunLogger"] = None

    def __init__(self, log_path: Optional[Path | str] = None):
        self._path = Path(log_path) if log_path else _ensure_path()
        self._runs: dict[str, dict[str, Any]] = {}
        self._log: list[dict[str, Any]] = []
        self._load()

    @classmethod
    def get(cls) -> "RunLogger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    # Legacy simple list format — migrate
                    self._log = data
                    self._runs = {}
                else:
                    self._runs = data.get("runs") or {}
                    self._log = data.get("log") or []
            except Exception:
                self._runs = {}
                self._log = []
        else:
            self._runs = {}
            self._log = []

    def _save(self) -> None:
        payload = {"runs": self._runs, "log": self._log}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def start_run(self, source_id: str, source_name: str = "") -> str:
        """Start a new run for a source. Returns run_id."""
        now = datetime.now(timezone.utc)
        # run_id = source_id + timestamp (safe for files/JSON)
        ts = now.isoformat(timespec="seconds").replace(":", "-")
        run_id = f"{source_id}_{ts}"
        self._runs[run_id] = {
            "run_id": run_id,
            "source_id": source_id,
            "source_name": source_name,
            "started_at": now.isoformat(),
            "events": [],
            "rows_in": 0,
            "rows_loaded": 0,
            "errors": 0,
            "finished_at": None,
            "status": "running",
        }
        self._save()
        return run_id

    def log_event(self, run_id: str, event_type: str, detail: Optional[dict] = None) -> None:
        if event_type not in EVENT_TYPES:
            # Still allow unknown for flexibility, but prefer the canonical list
            pass
        if run_id not in self._runs:
            self._runs[run_id] = {"run_id": run_id, "events": [], "rows_in": 0, "rows_loaded": 0, "errors": 0}

        event = {
            "event_type": event_type,
            "detail": detail or {},
            "at": datetime.now(timezone.utc).isoformat(),
        }
        self._runs[run_id].setdefault("events", []).append(event)
        self._log.append({"run_id": run_id, "source_id": self._runs[run_id].get("source_id"), **event})
        self._save()

    def finish_run(self, run_id: str, rows_in: int = 0, rows_loaded: int = 0, errors: int = 0, status: str = "ok") -> None:
        if run_id in self._runs:
            run = self._runs[run_id]
            run["finished_at"] = datetime.now(timezone.utc).isoformat()
            run["rows_in"] = rows_in
            run["rows_loaded"] = rows_loaded
            run["errors"] = errors
            run["status"] = status
        self._save()

    def append_legacy_entry(self, entry: dict) -> None:
        """For backward compatibility with the simple list format used by some scripts."""
        self._log.append(entry)
        self._save()

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._runs.get(run_id, {}).copy()


# Convenience singleton accessor (recommended)
def get_run_logger() -> RunLogger:
    return RunLogger.get()
