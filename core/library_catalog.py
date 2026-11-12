"""
core/library_catalog.py

Library scope management for the pipeline.

Defines available libraries, which stages they affect, and helpers for PyCaret model inclusion.
"""

from __future__ import annotations

from typing import Any

from core.project import Project

LIBRARIES: dict[str, dict[str, Any]] = {
    "pandas": {"required": True, "stages": ["all"]},
    "sklearn": {"required": True, "stages": ["all"]},
    "xgboost": {"required": False, "stages": ["model"]},
    "lightgbm": {"required": False, "stages": ["model"]},
    "catboost": {"required": False, "stages": ["model"]},
    "statsmodels": {"required": False, "stages": ["model"]},
    "featuretools": {"required": False, "stages": ["feature_eng"]},
    "category_encoders": {"required": False, "stages": ["feature_eng"]},
    "feature_engine": {"required": False, "stages": ["feature_eng", "clean"]},
    "shap": {"required": False, "stages": ["explain"]},
    "autoviz": {"required": False, "stages": ["eda"]},
    "ydata_profiling": {"required": False, "stages": ["profile"]},
    "yellowbrick": {"required": False, "stages": ["evaluate"]},
}

# Map enabled libraries to PyCaret 'include' lists (used in modeling stage)
PYCARET_MODEL_MAP: dict[str, list[str]] = {
    "sklearn": ["lr", "lda", "ridge", "et", "dt", "rf", "knn", "nb", "qda", "svm"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "catboost": ["catboost"],
    "statsmodels": [],  # statsmodels not in PyCaret; flag for separate handling
}


def for_stage(stage: str, enabled: list[str]) -> list[str]:
    """Return libraries that are enabled and apply to the given stage."""
    return [
        name
        for name, meta in LIBRARIES.items()
        if name in enabled and (stage in meta["stages"] or "all" in meta["stages"])
    ]


def pycaret_models_for(enabled: list[str]) -> list[str]:
    """Return the list of PyCaret model shortcuts to include based on enabled libraries."""
    out: list[str] = []
    for lib in enabled:
        out.extend(PYCARET_MODEL_MAP.get(lib, []))
    return out


def get_enabled_libraries(project: Project) -> list[str]:
    """Get currently enabled libraries for the project (with defaults)."""
    enabled = project.meta.get("enabled_libraries", [])
    # Ensure required libraries are always present
    defaults = ["pandas", "sklearn", "xgboost"]
    for lib in defaults:
        if lib not in enabled:
            enabled.append(lib)
    # Filter to only known libraries
    return [lib for lib in enabled if lib in LIBRARIES]


def set_enabled_libraries(project: Project, enabled: list[str]) -> None:
    """Persist the list of enabled libraries to project meta."""
    # Always keep required ones
    required = [name for name, meta in LIBRARIES.items() if meta.get("required")]
    final = sorted(set(enabled) | set(required))
    project.meta["enabled_libraries"] = final
    project.save_meta()


def get_task_config(project: Project) -> dict[str, Any]:
    """Return current task_type and target_column from meta."""
    return {
        "task_type": project.meta.get("task_type", "regression"),
        "target_column": project.meta.get("target_column"),
    }


def set_task_config(project: Project, task_type: str, target_column: str | None) -> None:
    """Persist task type and target column."""
    project.meta["task_type"] = task_type
    project.meta["target_column"] = target_column
    project.save_meta()
