"""
ui/sidebar.py — MIRA project sidebar: projects, task config, acquire source.
"""

from __future__ import annotations

import streamlit as st

from core.acquire_bridge import get_acquire_status, load_registry_df, pull_source_into_project
from core.ingest import ingest_dataframe
from core.library_catalog import (
    LIBRARIES,
    get_enabled_libraries,
    get_task_config,
    set_enabled_libraries,
    set_task_config,
)
from core.activity_log import unified_activity_df
from core.project import Project
from core.stage_history import get_recent_for_sidebar
from ui.project_picker import clear_project_cache


def render_sidebar() -> None:
    st.sidebar.header("MIRA")
    st.sidebar.caption("Multi-source Ingest & Records Analytics")

    # The records warehouse may be held by another process (a CLI scrape or a
    # second GUI instance) — a busy DB must never take the sidebar down.
    try:
        status = get_acquire_status()
        st.sidebar.metric("Records (global)", f"{status['record_count']:,}")
    except Exception:
        st.sidebar.metric("Records (global)", "—")
        st.sidebar.caption("records store busy (another process has it open)")

    # Project selection (incl. ?pid= deep links) lives in the shared picker,
    # rendered once for every page by the app router.
    st.sidebar.markdown("---")

    with st.sidebar.expander("➕ Create new project", expanded=False):
        with st.form("create_project_form", clear_on_submit=True):
            new_name = st.text_input("Project name", value="New Analysis")
            task_type = st.radio(
                "Task type (default)",
                options=["regression", "classification"],
                horizontal=True,
                index=0,
            )
            submitted = st.form_submit_button("Create Project", type="primary")
            if submitted and new_name.strip():
                try:
                    st.session_state.project = Project.create(
                        name=new_name.strip(), task_type=task_type
                    )
                    clear_project_cache()  # show it without waiting out the TTL
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to create project: {exc}")

    current_project = st.session_state.get("project")

    # --- Acquire source (Phase 1) ---
    st.sidebar.markdown("### Acquire source")
    try:
        reg_df = load_registry_df()
        source_ids = reg_df["source_id"].tolist() if not reg_df.empty else []
    except Exception:
        source_ids = []

    if source_ids:
        saved_id = (
            current_project.meta.get("acquire_source_id") if current_project else None
        )
        default_idx = source_ids.index(saved_id) if saved_id in source_ids else 0
        if "REAL001" in source_ids:
            default_idx = source_ids.index("REAL001")

        picked_source = st.sidebar.selectbox(
            "Registry source",
            source_ids,
            index=default_idx,
            key="sidebar_acquire_source",
            format_func=lambda sid: (
                f"{sid} — {reg_df.loc[reg_df['source_id'] == sid, 'source_name'].iloc[0]}"
                if not reg_df.empty and sid in reg_df["source_id"].values
                else sid
            ),
        )

        skip_scrape = st.sidebar.checkbox(
            "Load existing records only (skip scrape)",
            value=False,
            key="sidebar_skip_scrape",
        )

        if current_project:
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                if st.button("Acquire →", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Acquire…"):
                            result = pull_source_into_project(
                                current_project,
                                picked_source,
                                scrape=not skip_scrape,
                                ingest_fn=lambda p, d, source_name: ingest_dataframe(
                                    p, d, source_name=source_name, auto_profile=False
                                ),
                            )
                            if not current_project.meta.get("target_column"):
                                cols = result.get("columns", [])
                                default_target = (
                                    "primary_value" if "primary_value" in cols else None
                                )
                                set_task_config(
                                    current_project,
                                    current_project.meta.get("task_type", "regression"),
                                    default_target,
                                )
                        st.sidebar.success("Loaded into project")
                        st.rerun()
                    except Exception as exc:
                        st.sidebar.error(str(exc))
            with col_b:
                if st.button("Pipeline", use_container_width=True):
                    try:
                        from core.pipeline import Pipeline

                        current_project.meta["acquire_source_id"] = picked_source
                        current_project.save_meta()
                        pipe = Pipeline(current_project)
                        settings = {
                            "source_id": picked_source,
                            "use_existing_records": skip_scrape,
                        }
                        if not skip_scrape:
                            r0 = pipe.run_stage("acquire", settings)
                            if r0.status != "success":
                                raise RuntimeError(r0.error or "Acquire failed")
                        else:
                            pull_source_into_project(
                                current_project,
                                picked_source,
                                scrape=False,
                                ingest_fn=lambda p, d, source_name: ingest_dataframe(
                                    p, d, source_name=source_name, auto_profile=False
                                ),
                            )
                        r1 = pipe.run_stage("profile", {"minimal_threshold": 100})
                        if r1.status != "success":
                            raise RuntimeError(r1.error or "Profile failed")
                        st.sidebar.success("Acquire + Profile done")
                        st.rerun()
                    except Exception as exc:
                        st.sidebar.error(str(exc))
        else:
            st.sidebar.caption("Create a project to load records.")
    else:
        st.sidebar.caption("No sources in registry.")

    if current_project:
        st.sidebar.markdown("---")
        st.sidebar.caption(f"**{current_project.meta['name']}**")
        st.sidebar.caption(f"id: {current_project.id}")
        has_data = current_project.has_raw_data()
        st.sidebar.caption(f"data: {'✓ loaded' if has_data else '— not loaded'}")
        if current_project.meta.get("acquire_source_id"):
            st.sidebar.caption(f"source: {current_project.meta['acquire_source_id']}")

        st.sidebar.markdown("### Task configuration")
        task_cfg = get_task_config(current_project)
        current_task = task_cfg["task_type"]
        current_target = task_cfg.get("target_column")

        new_task = st.sidebar.radio(
            "Task type",
            options=["regression", "classification"],
            index=0 if current_task == "regression" else 1,
            horizontal=True,
            key=f"task_type_{current_project.id}",
        )

        target_options: list[str] = []
        if has_data:
            try:
                target_options = list(current_project.get_working_data().columns)
            except Exception:
                target_options = []

        if target_options:
            target_index = (
                target_options.index(current_target) if current_target in target_options else 0
            )
            new_target = st.sidebar.selectbox(
                "Target column",
                options=target_options,
                index=target_index,
                key=f"target_col_{current_project.id}",
            )
        else:
            new_target = None
            st.sidebar.caption("Load data to select a target column.")

        if new_task != current_task or new_target != current_target:
            set_task_config(current_project, new_task, new_target)
            st.rerun()

        st.sidebar.markdown("### Enabled libraries")
        current_enabled = set(get_enabled_libraries(current_project))
        groups = {
            "Core (always on)": [name for name, m in LIBRARIES.items() if m.get("required")],
            "ML / AutoML": ["xgboost", "lightgbm", "catboost", "statsmodels"],
            "Feature Engineering": ["featuretools", "category_encoders", "feature_engine"],
            "Explain & Viz": ["shap", "autoviz", "ydata-profiling", "yellowbrick"],
        }
        new_enabled: set[str] = set()
        for group_name, libs in groups.items():
            with st.sidebar.expander(group_name, expanded=(group_name == "Core (always on)")):
                for lib in libs:
                    if lib not in LIBRARIES:
                        continue
                    meta = LIBRARIES[lib]
                    is_required = meta.get("required", False)
                    checked = st.checkbox(
                        lib,
                        value=(lib in current_enabled),
                        disabled=is_required,
                        key=f"lib_{lib}_{current_project.id}",
                        help=f"Affects: {', '.join(meta.get('stages', []))}",
                    )
                    if checked or is_required:
                        new_enabled.add(lib)

        if new_enabled != current_enabled:
            set_enabled_libraries(current_project, sorted(new_enabled))
            st.rerun()

        st.sidebar.caption(f"Active libraries: {len(get_enabled_libraries(current_project))}")

        st.sidebar.markdown("### Recent activity")
        recent = get_recent_for_sidebar(current_project.root, limit=6)
        if recent:
            for entry in recent:
                st.sidebar.text(
                    f"{entry['at'][:16]} | {entry['stage']}: {entry['status']}"
                )
        else:
            st.sidebar.caption("No pipeline runs yet.")

        global_act = unified_activity_df(limit=3)
        if not global_act.empty:
            st.sidebar.caption("Acquire (global)")
            for _, row in global_act.head(3).iterrows():
                st.sidebar.text(f"{str(row.get('at', ''))[:16]} | {row.get('stage', '')}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Sources & Records → registry + scrape")