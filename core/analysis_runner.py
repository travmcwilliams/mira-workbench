"""Run end-to-end analysis pipeline and collect metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.data_loader import detect_schema, load_file
from core.pipeline import Pipeline
from core.project import Project


def run_full_pipeline(
    data_path: Path | str,
    target: str,
    task: str,
    *,
    project_name: str | None = None,
) -> tuple[Project, dict]:
    data_path = Path(data_path)
    df = load_file(data_path)
    name = project_name or f"Analysis — {data_path.stem}"
    project = Project.create(name=name, task_type=task)
    project.set_raw_data(df)
    project.meta["schema"] = detect_schema(df)
    project.meta["target_column"] = target
    project.save_meta()

    pipeline = Pipeline(project)
    results = pipeline.run_all()
    if not results or results[-1].status != "success":
        err = results[-1].error if results else "no results"
        raise RuntimeError(err or "pipeline failed")

    metrics: dict = {}
    metrics_path = project.artifact_path("eval/metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    lb_path = project.artifact_path("leaderboard.csv")
    if lb_path.exists() and not metrics:
        lb = pd.read_csv(lb_path, index_col=0)
        if len(lb):
            metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in lb.iloc[0].to_dict().items()}

    return project, metrics