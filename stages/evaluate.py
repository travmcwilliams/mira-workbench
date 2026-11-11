"""
stages/evaluate.py

Evaluation stage: metrics + plots from the best model.
Supports both PyCaret and sklearn backends.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from core.model_backend import get_model_backend, load_prep_splits, load_saved_model
from core.pipeline import StageResult
from core.project import Project


def default_settings(project: Project) -> dict[str, Any]:
    """Default settings for evaluation stage."""
    return {}


def _evaluate_sklearn(
    project: Project,
    df: pd.DataFrame,
    target: str,
    task: str,
    eval_dir: Path,
) -> StageResult:
    splits = load_prep_splits(project)
    if splits is None:
        return StageResult(
            status="error",
            error="Sklearn evaluation requires data_prep artifacts in artifacts/prep/.",
        )

    _, X_test, _, y_test = splits
    model = load_saved_model(project)
    y_pred = model.predict(X_test)

    metrics: dict[str, Any] = {}
    generated: list[Path] = []

    if task == "classification":
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        }
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.colorbar(im, ax=ax)
        cm_path = eval_dir / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        generated.append(cm_path)

        report_path = eval_dir / "classification_report.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(classification_report(y_test, y_pred, output_dict=True), handle, indent=2)
        generated.append(report_path)
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(mean_squared_error(y_test, y_pred, squared=False))
        r2 = float(r2_score(y_test, y_pred))
        metrics = {"mae": mae, "rmse": rmse, "r2": r2}

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", color="gray")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual")
        residual_path = eval_dir / "residuals.png"
        fig.tight_layout()
        fig.savefig(residual_path, bbox_inches="tight")
        plt.close(fig)
        generated.append(residual_path)

    lb_path = project.artifact_path("leaderboard.csv")
    if lb_path.exists():
        try:
            lb = pd.read_csv(lb_path)
            if len(lb) > 0:
                metrics["leaderboard_top"] = lb.iloc[0].to_dict()
        except Exception:
            pass

    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return StageResult(
        status="success",
        output_df=df,
        artifacts=[metrics_path, *generated],
        summary={
            "backend": "sklearn",
            "task": task,
            "target": target,
            "n_plots": len(generated),
            "metrics_keys": list(metrics.keys()),
        },
    )


def _evaluate_pycaret(
    project: Project,
    df: pd.DataFrame,
    target: str,
    task: str,
    eval_dir: Path,
) -> StageResult:
    model_stem = str(project.artifact_path("model").with_suffix(""))

    if task == "classification":
        plot_list = ["confusion_matrix", "roc", "pr", "error", "class_report"]
        from pycaret.classification import load_model, plot_model, setup
    else:
        plot_list = ["residuals", "error", "cooks", "feature"]
        from pycaret.regression import load_model, plot_model, setup

    setup(df, target=target, session_id=42, verbose=False, n_jobs=-1)
    model = load_model(model_stem)

    original_cwd = os.getcwd()
    os.chdir(eval_dir)
    try:
        for plot_name in plot_list:
            try:
                plot_model(model, plot=plot_name, save=True, verbose=False)
            except Exception:
                continue
        generated = sorted(eval_dir.glob("*.png"))
    finally:
        os.chdir(original_cwd)

    metrics: dict[str, Any] = {}
    lb_path = project.artifact_path("leaderboard.csv")
    if lb_path.exists():
        try:
            lb = pd.read_csv(lb_path, index_col=0)
            if len(lb) > 0:
                metrics = {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in lb.iloc[0].to_dict().items()
                }
        except Exception:
            pass

    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return StageResult(
        status="success",
        output_df=df,
        artifacts=[metrics_path, *generated],
        summary={
            "backend": "pycaret",
            "task": task,
            "target": target,
            "n_plots": len(generated),
            "metrics_keys": list(metrics.keys())[:5] if metrics else [],
        },
    )


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    """Generate evaluation plots and metrics using the saved best model."""
    if df is None or len(df) < 5:
        return StageResult(status="error", error="Insufficient data for evaluation.")

    target = project.meta.get("target_column")
    task = project.meta.get("task_type", "regression")

    if not target or target not in df.columns:
        return StageResult(status="error", error="Target column not set or missing.")

    eval_dir = project.artifact_path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    try:
        if get_model_backend(project) == "sklearn":
            return _evaluate_sklearn(project, df, target, task, eval_dir)
        return _evaluate_pycaret(project, df, target, task, eval_dir)
    except Exception as exc:
        return StageResult(status="error", error=f"Evaluation failed: {exc}")


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    """Placeholder. Full controls (which plots to generate) can be added later."""
    return settings