"""Pipeline navigator — stage cards grouped by phase."""

from __future__ import annotations

import streamlit as st

from core.pipeline import STAGE_ORDER, Pipeline
from core.project import Project
from ui.components import image_gallery
from ui.preanalysis import render_preanalysis_workbench
from ui.stage_view import STAGE_LABELS, render_stage_card

STAGE_GROUPS: list[tuple[str, list[str]]] = [
    ("Understand", ["profile", "eda"]),
    ("Prepare", ["clean", "feature_eng", "feature_select"]),
    ("Model & Evaluate", ["data_prep", "model", "evaluate", "explain"]),
    ("Ship", ["deploy"]),
]

SPECIAL_VIEWS = {"preanalysis": "Pre-Analysis Workbench"}


def render_pipeline_navigator(project: Project) -> None:
    st.header("Pipeline")
    st.caption("Configure each stage, save settings, and run individually or in sequence.")

    completed = project.meta.get("completed_stages", [])
    prog = len([s for s in STAGE_ORDER if s in completed])
    st.progress(min(prog / max(len(STAGE_ORDER), 1), 1.0), text=f"{prog}/{len(STAGE_ORDER)} stages complete")

    nav_options = ["preanalysis", *STAGE_ORDER]
    nav_labels = {
        "preanalysis": "🔧 Pre-Analysis Workbench",
        **{s: STAGE_LABELS.get(s, s) for s in STAGE_ORDER},
    }

    picked = st.selectbox(
        "Jump to",
        nav_options,
        format_func=lambda x: nav_labels.get(x, x),
        key=f"nav_{project.id[:8]}",
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("Run from here → end", key=f"run_from_{project.id[:8]}"):
            _run_from_stage(project, picked if picked != "preanalysis" else "clean")
    with r2:
        if st.button("Run all remaining", key=f"run_all_{project.id[:8]}"):
            _run_remaining(project)
    with r3:
        reset_stage = st.selectbox(
            "Reset from",
            ["(none)"] + [s for s in STAGE_ORDER if s in completed],
            key=f"reset_{project.id[:8]}",
        )
        if reset_stage != "(none)" and st.button("Reset", key=f"do_reset_{project.id[:8]}"):
            Pipeline(project).reset_from(reset_stage)
            st.success(f"Reset from {reset_stage}")
            st.rerun()

    st.divider()

    if picked == "preanalysis":
        render_preanalysis_workbench(project)
        return

    if picked == "profile":
        profile_path = project.artifact_path("profile_report.html")
        if profile_path.exists():
            with st.expander("Profile report preview", expanded=True):
                try:
                    html = profile_path.read_text(encoding="utf-8", errors="replace")
                    st.components.v1.html(html, height=700, scrolling=True)
                except Exception as exc:
                    st.warning(str(exc))

    if picked == "eda":
        eda_dir = project.artifact_path("eda")
        imgs = sorted(eda_dir.glob("*.png")) if eda_dir.exists() else []
        if imgs:
            image_gallery(imgs, title="EDA plots", max_cols=2)

    if picked in ("model", "evaluate", "explain", "deploy"):
        _render_model_artifacts(project, picked)

    render_stage_card(project, picked)


def _render_model_artifacts(project: Project, stage: str) -> None:
    import pandas as pd

    if stage == "model":
        lb = project.artifact_path("leaderboard.csv")
        if lb.exists():
            st.markdown("**Leaderboard**")
            st.dataframe(pd.read_csv(lb, index_col=0), use_container_width=True)
    if stage == "evaluate":
        eval_dir = project.artifact_path("eval")
        imgs = sorted(eval_dir.glob("*.png")) if eval_dir.exists() else []
        if imgs:
            image_gallery(imgs, title="Evaluation", max_cols=2)
    if stage == "explain":
        shap_dir = project.artifact_path("shap")
        imgs = sorted(shap_dir.glob("*.png")) if shap_dir.exists() else []
        if imgs:
            image_gallery(imgs, title="Explainability", max_cols=2)
    if stage == "deploy":
        exports = project.root / "exports"
        if exports.exists():
            for f in sorted(exports.iterdir()):
                st.caption(f"📦 `{f.name}`")


def _run_from_stage(project: Project, start: str) -> None:
    if start not in STAGE_ORDER:
        start = "profile"
    pipe = Pipeline(project)
    started = False
    for name in STAGE_ORDER:
        if name == start:
            started = True
        if not started:
            continue
        with st.spinner(f"Running {STAGE_LABELS.get(name, name)}…"):
            res = pipe.run_stage(name)
            if res.status != "success":
                st.error(f"{name} failed: {res.error}")
                return
    st.success("Pipeline run complete.")
    st.rerun()


def _run_remaining(project: Project) -> None:
    completed = set(project.meta.get("completed_stages", []))
    pipe = Pipeline(project)
    for name in STAGE_ORDER:
        if name in completed:
            continue
        with st.spinner(f"Running {STAGE_LABELS.get(name, name)}…"):
            res = pipe.run_stage(name)
            if res.status != "success":
                st.error(f"{name} failed: {res.error}")
                return
    st.success("All remaining stages complete.")
    st.rerun()