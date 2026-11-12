"""
stages/model.py

PyCaret AutoML modeling stage with sklearn fallback.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.library_catalog import get_enabled_libraries, pycaret_models_for
from core.model_backend import (
    load_prep_splits,
    pycaret_supported,
    save_sklearn_model,
    set_model_backend,
)
from core.pipeline import StageResult
from core.project import Project
from core.sklearn_automl import compare_models as sklearn_compare_models


def default_settings(project: Project) -> dict[str, Any]:
    """Default settings for the modeling stage."""
    return {
        "n_models": 5,
        "cv_folds": 5,
        "prefer_pycaret": True,
    }


def _run_pycaret(
    project: Project,
    df: pd.DataFrame,
    target: str,
    task: str,
    settings: dict[str, Any],
) -> StageResult:
    include = pycaret_models_for(get_enabled_libraries(project))
    n_select = int(settings.get("n_models", 5))
    folds = int(settings.get("cv_folds", 5))

    if task == "classification":
        from pycaret.classification import (  # noqa: I001
            setup as cls_setup,
            compare_models as cls_compare,
            save_model,
            pull as cls_pull,
        )

        cls_setup(df, target=target, session_id=42, verbose=False, n_jobs=-1)
        try:
            best = cls_compare(n_select=n_select, fold=folds, include=include or None)
        except Exception as exc:
            if include and "Not Available" in str(exc):
                best = cls_compare(n_select=n_select, fold=folds, include=None)
            else:
                raise
        leaderboard = cls_pull()
    elif task == "regression":
        from pycaret.regression import (  # noqa: I001
            setup as reg_setup,
            compare_models as reg_compare,
            save_model,
            pull as reg_pull,
        )

        reg_setup(df, target=target, session_id=42, verbose=False, n_jobs=-1)
        try:
            best = reg_compare(n_select=n_select, fold=folds, include=include or None)
        except Exception as exc:
            if include and "Not Available" in str(exc):
                best = reg_compare(n_select=n_select, fold=folds, include=None)
            else:
                raise
        leaderboard = reg_pull()
    else:
        return StageResult(status="error", error=f"Unsupported task type: {task}")

    if leaderboard is None or len(leaderboard) == 0:
        return StageResult(status="error", error="PyCaret returned an empty leaderboard.")

    best_model = best[0] if isinstance(best, list) else best
    model_stem = project.artifact_path("model")
    save_model(best_model, str(model_stem.with_suffix("")))
    set_model_backend(project, "pycaret")

    lb_path = project.artifact_path("leaderboard.csv")
    leaderboard.to_csv(lb_path)

    summary = {
        "backend": "pycaret",
        "n_models": len(leaderboard),
        "best_model": str(leaderboard.index[0]) if len(leaderboard) > 0 else "N/A",
        "task": task,
        "target": target,
        "include": include,
    }

    return StageResult(
        status="success",
        output_df=df,
        artifacts=[project.artifact_path("model.pkl"), lb_path],
        summary=summary,
    )


def _run_sklearn(
    project: Project,
    df: pd.DataFrame,
    target: str,
    task: str,
    settings: dict[str, Any],
    *,
    fallback_reason: str | None = None,
) -> StageResult:
    # Train on the data_prep train split so the model sees the SAME feature
    # representation (scaling) that evaluate/deploy use, and never sees the
    # held-out test rows. Fall back to the full working df only when the
    # data_prep stage has not run.
    splits = load_prep_splits(project)
    trained_on = "full_df"
    if splits is not None:
        X_train, _, y_train, _ = splits
        fit_df = X_train.copy()
        fit_df[target] = y_train
        trained_on = "train_split"
    else:
        fit_df = df

    best_model, leaderboard = sklearn_compare_models(
        project,
        fit_df,
        target,
        n_models=int(settings.get("n_models", 5)),
        cv_folds=int(settings.get("cv_folds", 5)),
    )

    model_path = save_sklearn_model(project, best_model)
    lb_path = project.artifact_path("leaderboard.csv")
    leaderboard.to_csv(lb_path)

    summary = {
        "backend": "sklearn",
        "n_models": len(leaderboard),
        "best_model": str(leaderboard.iloc[0]["Model"]),
        "task": task,
        "target": target,
        "trained_on": trained_on,
    }
    if fallback_reason:
        summary["fallback_reason"] = fallback_reason

    return StageResult(
        status="success",
        output_df=df,
        artifacts=[model_path, lb_path],
        summary=summary,
    )


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    """Run PyCaret compare_models, falling back to sklearn when needed."""
    if df is None or len(df) < 20:
        return StageResult(
            status="error", error="Not enough data for modeling (need at least 20 rows)."
        )

    target = project.meta.get("target_column")
    task = project.meta.get("task_type", "regression")

    if not target or target not in df.columns:
        return StageResult(status="error", error="Target column is not set or missing from data.")

    prefer_pycaret = bool(settings.get("prefer_pycaret", True))
    pycaret_ok, pycaret_reason = pycaret_supported()

    if prefer_pycaret and pycaret_ok:
        try:
            return _run_pycaret(project, df, target, task, settings)
        except Exception as exc:
            return _run_sklearn(
                project,
                df,
                target,
                task,
                settings,
                fallback_reason=f"PyCaret failed: {exc}",
            )

    reason = pycaret_reason or "PyCaret disabled by settings"
    return _run_sklearn(project, df, target, task, settings, fallback_reason=reason)


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    import streamlit as st

    settings["n_models"] = st.slider(
        "Models to compare",
        2, 15,
        int(settings.get("n_models", 5)),
    )
    settings["cv_folds"] = st.slider(
        "CV folds",
        2, 10,
        int(settings.get("cv_folds", 5)),
    )
    settings["prefer_pycaret"] = st.checkbox(
        "Prefer PyCaret (fallback to sklearn)",
        value=bool(settings.get("prefer_pycaret", True)),
    )
    return settings