"""
core/project.py

A single user project. Wraps the folder structure + meta.json + state.db.
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from datetime import UTC
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core.state import init_db

# Anchor at the repo root so the GUI/CLI work from any cwd. MIRA_PROJECTS_ROOT
# overrides for tests: a RELATIVE value stays cwd-relative (legacy semantics —
# tests/conftest.py sets "projects" so chdir(tmp_path) fixtures keep isolating).
_REPO_ROOT = Path(__file__).resolve().parents[1]


def projects_root() -> Path:
    env = os.environ.get("MIRA_PROJECTS_ROOT")
    return Path(env) if env else _REPO_ROOT / "projects"


class Project:
    """A single user project. Wraps the folder structure + meta.json + state.db."""

    def __init__(self, project_id: str):
        self.id = project_id
        self.root = projects_root() / project_id
        self._meta: Optional[dict[str, Any]] = None

    @classmethod
    def create(cls, name: str, task_type: str = "regression") -> "Project":
        pid = str(uuid.uuid4())[:8]
        p = cls(pid)
        p.root.mkdir(parents=True, exist_ok=True)
        (p.root / "data").mkdir(exist_ok=True)
        (p.root / "artifacts").mkdir(exist_ok=True)
        (p.root / "exports").mkdir(exist_ok=True)

        p._meta = {
            "id": pid,
            "name": name,
            "created_at": datetime.datetime.now(UTC).isoformat() + "Z",
            "updated_at": datetime.datetime.now(UTC).isoformat() + "Z",
            "task_type": task_type,
            "target_column": None,
            "enabled_libraries": ["pandas", "sklearn", "xgboost"],
            "current_stage": "business",
            "completed_stages": [],
            "schema": {},
        }
        p.save_meta()

        init_db(p.root / "state.db")
        return p

    @classmethod
    def load(cls, project_id: str) -> "Project":
        p = cls(project_id)
        meta_path = p.root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found for project {project_id}")
        with open(meta_path) as f:
            p._meta = json.load(f)
        return p

    @classmethod
    def list_all(cls) -> list["Project"]:
        root = projects_root()
        if not root.exists():
            return []
        projects: list[Project] = []
        for d in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if d.is_dir():
                try:
                    projects.append(cls.load(d.name))
                except Exception:
                    # Skip corrupt project folders
                    continue
        return projects

    @property
    def meta(self) -> dict[str, Any]:
        if self._meta is None:
            raise RuntimeError("Project not loaded")
        return self._meta

    def save_meta(self) -> None:
        assert self._meta is not None
        self._meta["updated_at"] = datetime.datetime.now(UTC).isoformat() + "Z"
        with open(self.root / "meta.json", "w") as f:
            json.dump(self._meta, f, indent=2)

    def get_raw_data(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "data" / "raw.parquet")

    def get_working_data(self) -> pd.DataFrame:
        path = self.root / "data" / "working.parquet"
        if not path.exists():
            return self.get_raw_data()
        return pd.read_parquet(path)

    def set_raw_data(self, df: pd.DataFrame) -> None:
        """Persist the original uploaded data."""
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.root / "data" / "raw.parquet")
        # On first ingest, working copy starts as the raw data
        self.set_working_data(df)

    def set_working_data(self, df: pd.DataFrame) -> None:
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.root / "data" / "working.parquet")

    def has_raw_data(self) -> bool:
        return (self.root / "data" / "raw.parquet").exists()

    def artifact_path(self, *parts: str) -> Path:
        p = self.root / "artifacts" / Path(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __repr__(self) -> str:
        return f"Project(id={self.id}, name={self.meta.get('name')})"


def list_project_index() -> list[dict[str, Any]]:
    """Lightweight, picklable project index for UI listings (no Project objects).

    One meta.json read per project dir, mtime-sorted newest first; corrupt dirs
    skipped like ``Project.list_all``. Fields: id, name, mtime, completed_stages,
    task_type, updated_at.
    """
    root = projects_root()
    if not root.exists():
        return []
    index: list[dict[str, Any]] = []
    for d in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            continue  # not a project / corrupt — same policy as list_all
        index.append({
            "id": d.name,
            "name": str(meta.get("name") or d.name),
            "mtime": d.stat().st_mtime,
            "completed_stages": list(meta.get("completed_stages") or []),
            "task_type": meta.get("task_type"),
            "updated_at": meta.get("updated_at"),
        })
    return index
