"""Shared model persistence and loading for PyCaret and sklearn backends."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

from core.project import Project

ModelBackend = Literal["pycaret", "sklearn"]


def get_model_backend(project: Project) -> ModelBackend:
    return project.meta.get("model_backend", "pycaret")


def set_model_backend(project: Project, backend: ModelBackend) -> None:
    project.meta["model_backend"] = backend
    project.save_meta()


def model_pkl_path(project: Project) -> Path:
    return project.artifact_path("model.pkl")


def model_stem(project: Project) -> str:
    return str(project.artifact_path("model").with_suffix(""))


def pycaret_supported() -> tuple[bool, str | None]:
    """Return whether PyCaret can run in the current Python interpreter."""
    import sys

    if sys.version_info[:2] not in {(3, 9), (3, 10), (3, 11)}:
        return False, (
            f"PyCaret requires Python 3.9–3.11; current interpreter is "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )
    try:
        import pycaret  # noqa: F401
    except ImportError as exc:
        return False, f"PyCaret is not installed: {exc}"
    return True, None


def save_sklearn_model(project: Project, model: Any) -> Path:
    path = model_pkl_path(project)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)
    set_model_backend(project, "sklearn")
    return path


def load_saved_model(project: Project) -> Any:
    backend = get_model_backend(project)
    if backend == "sklearn":
        path = model_pkl_path(project)
        if not path.exists():
            raise FileNotFoundError(f"Sklearn model not found: {path}")
        with open(path, "rb") as handle:
            return pickle.load(handle)

    task = project.meta.get("task_type", "regression")
    if task == "classification":
        from pycaret.classification import load_model
    else:
        from pycaret.regression import load_model
    return load_model(model_stem(project))


def raw_estimator(model: Any) -> Any:
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def load_prep_splits(project: Project) -> Any:
    """Load the train/test splits persisted by the ``data_prep`` stage.

    Returns ``(X_train, X_test, y_train, y_test)`` (X as DataFrames, y as
    Series) or ``None`` when the artifacts are absent. These splits carry the
    feature representation (scaled or raw) that the ``data_prep`` stage chose,
    and are the single source of truth shared by the model, evaluate, and
    explain stages so all three see identically-transformed features.
    """
    import pandas as pd

    prep = project.artifact_path("prep")
    paths = {
        "X_train": prep / "X_train.parquet",
        "X_test": prep / "X_test.parquet",
        "y_train": prep / "y_train.parquet",
        "y_test": prep / "y_test.parquet",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    X_train = pd.read_parquet(paths["X_train"])
    X_test = pd.read_parquet(paths["X_test"])
    y_train = pd.read_parquet(paths["y_train"]).squeeze("columns")
    y_test = pd.read_parquet(paths["y_test"]).squeeze("columns")
    return X_train, X_test, y_train, y_test