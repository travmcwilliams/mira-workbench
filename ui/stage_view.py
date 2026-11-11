"""Generic stage card renderer with settings, run controls, and artifacts."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.pipeline import STAGE_ORDER, Pipeline, StageResult
from core.project import Project
from core.stage_history import get_last_stage_run
from core.stage_settings import get_stage_settings, save_stage_settings

STAGE_LABELS: dict[str, str] = {
    "business": "Business Framing",
    "acquire": "Acquire",
    "profile": "Profile",
    "eda": "EDA",
    "clean": "Clean",
    "feature_eng": "Feature Engineering",
    "feature_select": "Feature Selection",
    "data_prep": "Data Prep",
    "model": "Model",
    "evaluate": "Evaluate",
    "explain": "Explain",
    "deploy": "Deploy",
}


def _stage_module(stage: str):
    return importlib.import_module(f"stages.{stage}")


def _render_settings_ui(
    project: Project,
    stage: str,
    settings: dict[str, Any],
    last_result: StageResult | None,
) -> dict[str, Any]:
    mod = _stage_module(stage)
    if hasattr(mod, "render_ui"):
        return mod.render_ui(project, settings, last_result)
    return settings


def _artifact_links(project: Project, stage: str, last_run: dict[str, Any] | None) -> list[Path]:
    paths: list[Path] = []
    if last_run:
        arts = last_run.get("artifacts") or []
        if isinstance(arts, list):
            for a in arts:
                p = Path(a)
                if not p.is_absolute():
                    p = project.root / p
                if p.exists():
                    paths.append(p)

    defaults = {
        "profile": [project.artifact_path("profile_report.html")],
        "eda": list(project.artifact_path("eda").glob("*.png")) if project.artifact_path("eda").exists() else [],
        "model": [project.artifact_path("leaderboard.csv")],
        "evaluate": list(project.artifact_path("eval").glob("*.png")) if project.artifact_path("eval").exists() else [],
        "explain": list(project.artifact_path("shap").glob("*.png")) if project.artifact_path("shap").exists() else [],
        "deploy": [project.artifact_path("pipeline.pkl"), project.root / "exports" / "notebook.ipynb"],
    }
    for p in defaults.get(stage, []):
        if p.exists() and p not in paths:
            paths.append(p)
    return paths


def render_stage_card(project: Project, stage: str, *, key_prefix: str | None = None) -> None:
    """Render one pipeline stage: settings, run, summary, artifacts."""
    key = key_prefix or f"stage_{stage}_{project.id[:8]}"
    label = STAGE_LABELS.get(stage, stage.replace("_", " ").title())
    completed = stage in project.meta.get("completed_stages", [])
    status_icon = "✅" if completed else "○"

    st.subheader(f"{status_icon} {label}")

    settings = get_stage_settings(project, stage)
    last_run = get_last_stage_run(project.root, stage)
    last_result = None
    if last_run and last_run.get("status") == "success":
        last_result = StageResult(
            status="success",
            summary=last_run.get("summary") or {},
        )

    updated = _render_settings_ui(project, stage, settings, last_result)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Save settings", key=f"{key}_save"):
            save_stage_settings(project, stage, updated)
            st.success("Saved.")
            st.rerun()
    with c2:
        run_clicked = st.button("Run stage", type="primary", key=f"{key}_run")
    with c3:
        if last_run:
            st.caption(
                f"Last: **{last_run.get('status', '?')}** "
                f"({last_run.get('duration_ms', 0)} ms) — {last_run.get('run_at', '')}"
            )

    if run_clicked:
        save_stage_settings(project, stage, updated)
        with st.spinner(f"Running {label}…"):
            pipe = Pipeline(project)
            res = pipe.run_stage(stage, updated)
            if res.status == "success":
                st.success(f"{label} complete.")
                if res.summary:
                    st.json(res.summary)
                st.rerun()
            else:
                st.error(res.error or "Stage failed.")

    if last_run and last_run.get("summary"):
        with st.expander("Last run summary", expanded=False):
            st.json(last_run["summary"])

    artifacts = _artifact_links(project, stage, last_run)
    if artifacts:
        st.markdown("**Artifacts**")
        for path in artifacts[:12]:
            rel = path.relative_to(project.root) if path.is_relative_to(project.root) else path.name
            if path.suffix.lower() == ".html":
                size = path.stat().st_size
                st.download_button(
                    f"📄 Download {rel}", data=path.read_bytes(),
                    file_name=path.name, mime="text/html",
                    key=f"dl_html_{stage}_{path.name}",
                )
                if size <= 8_000_000:  # embed like pipeline_nav's profile branch
                    with st.expander(f"Preview {path.name}", expanded=False):
                        st.components.v1.html(
                            path.read_text(encoding="utf-8", errors="replace"),
                            height=700, scrolling=True,
                        )
                else:
                    st.caption(f"(too large to embed inline: {size / 1e6:.1f} MB)")
            elif path.suffix.lower() in {".png", ".jpg"}:
                st.image(str(path), caption=str(rel), use_container_width=True)
            elif path.suffix.lower() == ".csv":
                try:
                    st.dataframe(pd.read_csv(path).head(10), use_container_width=True)
                except Exception:
                    st.caption(f"📊 `{rel}`")
            else:
                if path.stat().st_size < 5_000_000:
                    with open(path, "rb") as handle:
                        st.download_button(
                            f"Download {path.name}",
                            data=handle.read(),
                            file_name=path.name,
                            key=f"{key}_dl_{path.name}",
                        )


def render_stage_status_bar(project: Project) -> str | None:
    """Compact stage picker with completion badges. Returns selected stage name."""
    completed = set(project.meta.get("completed_stages", []))
    options = STAGE_ORDER
    labels = [
        f"{'✓' if s in completed else '·'} {STAGE_LABELS.get(s, s)}"
        for s in options
    ]
    idx = 0
    current = project.meta.get("current_stage")
    if current in options:
        idx = options.index(current)

    picked = st.radio(
        "Pipeline stage",
        options=options,
        index=idx,
        format_func=lambda s: labels[options.index(s)],
        horizontal=True,
        key=f"stage_pick_{project.id[:8]}",
    )
    return picked