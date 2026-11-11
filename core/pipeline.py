"""
core/pipeline.py

Pipeline orchestration + StageResult.
Supports Phase 2 (data ingest + profile stage execution).
See grok_handoff.md Sections 5 and 6.4. Will be extended in later phases.
"""

from __future__ import annotations

import importlib
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core.project import Project
from core.stage_settings import get_stage_settings
from core.state import log_run


@dataclass
class StageResult:
    status: str  # 'success' | 'error'
    output_df: pd.DataFrame | None = None
    artifacts: list[Path] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# The canonical order of stages
STAGE_ORDER: list[str] = [
    "business",
    "acquire",
    "profile",
    "eda",
    "clean",
    "feature_eng",
    "feature_select",
    "data_prep",
    "model",
    "evaluate",
    "explain",
    "deploy",
]


class Pipeline:
    """Orchestrates stage execution for a Project."""

    def __init__(self, project: Project):
        self.project = project

    def run_stage(self, stage_name: str, settings: dict[str, Any] | None = None) -> StageResult:
        if stage_name not in STAGE_ORDER:
            return StageResult(status="error", error=f"Unknown stage: {stage_name}")

        if settings is None:
            settings = self._get_settings(stage_name)

        try:
            mod = importlib.import_module(f"stages.{stage_name}")
        except ModuleNotFoundError as e:
            return StageResult(status="error", error=f"Stage module missing: {e}")

        df = None
        if stage_name != "business":
            try:
                df = self.project.get_working_data()
            except Exception:
                df = None

        t0 = time.time()
        try:
            result: StageResult = mod.run(self.project, df, settings)
            if result.output_df is not None:
                self.project.set_working_data(result.output_df)
        except Exception as e:
            tb = traceback.format_exc()
            result = StageResult(status="error", error=f"{e}\n{tb}")

        # Log the run (best effort)
        try:
            log_run(
                self.project.root / "state.db",
                stage=stage_name,
                settings=settings or {},
                status=result.status,
                duration_ms=int((time.time() - t0) * 1000),
                artifacts=[str(p) for p in result.artifacts],
                summary=result.summary,
                error=result.error,
            )
        except Exception:
            pass

        # Also append to plain text run.log for human review
        try:
            log_path = self.project.root / "run.log"
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {stage_name}: {result.status} "
                    f"({int((time.time() - t0) * 1000)}ms) {result.error or ''}\n"
                )
        except Exception:
            pass

        if result.status == "success":
            completed = self.project.meta.get("completed_stages", [])
            if stage_name not in completed:
                completed.append(stage_name)
                self.project.meta["completed_stages"] = completed
            self.project.meta["current_stage"] = stage_name
            self.project.save_meta()

        return result

    def run_all(self, until: str | None = None) -> list[StageResult]:
        results: list[StageResult] = []
        for name in STAGE_ORDER:
            settings = self._get_settings(name)
            r = self.run_stage(name, settings)
            results.append(r)
            if r.status == "error":
                break
            if until and name == until:
                break
        return results

    def reset_from(self, stage_name: str) -> None:
        """Invalidate this stage and all downstream stages."""
        if stage_name not in STAGE_ORDER:
            return
        idx = STAGE_ORDER.index(stage_name)
        to_clear = set(STAGE_ORDER[idx:])
        self.project.meta["completed_stages"] = [
            s for s in self.project.meta.get("completed_stages", []) if s not in to_clear
        ]
        self.project.save_meta()

    def _get_settings(self, stage: str) -> dict[str, Any]:
        return get_stage_settings(self.project, stage)
