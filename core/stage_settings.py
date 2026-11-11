"""Load and persist per-stage pipeline settings on a project."""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any

from core.project import Project


def get_stage_settings(project: Project, stage: str) -> dict[str, Any]:
    """Merge saved user settings with stage default_settings."""
    defaults: dict[str, Any] = {}
    try:
        mod = importlib.import_module(f"stages.{stage}")
        if hasattr(mod, "default_settings"):
            result = mod.default_settings(project)
            if isinstance(result, dict):
                defaults = deepcopy(result)
    except Exception:
        pass

    saved = project.meta.get("stage_settings", {}).get(stage, {})
    if not isinstance(saved, dict):
        return defaults
    return _deep_merge(defaults, saved)


def save_stage_settings(project: Project, stage: str, settings: dict[str, Any]) -> None:
    project.meta.setdefault("stage_settings", {})
    project.meta["stage_settings"][stage] = deepcopy(settings)
    project.save_meta()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out