"""Home page body — moved out of app.py when app.py became the nav router.

`render_home()` keeps the original three states: (a) no project selected →
landing; (b) project without data → load-data workspace; (c) project with
data → preview + dashboards + pipeline navigator. The Home-only sidebar
extras (create form, acquire source, task config, libraries, activity) render
here via `render_sidebar()`; the project picker itself is global (router).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.acquire_bridge import get_acquire_status, load_registry_df, pull_source_into_project
from core.data_loader import load_file, load_from_upload
from core.ingest import ingest_dataframe
from core.library_catalog import get_enabled_libraries, get_task_config, set_task_config
from core.project import Project
from ui.dashboards import render_interactive_dashboard
from ui.dataset_download import render_dataset_download
from ui.pipeline_nav import render_pipeline_navigator
from ui.sidebar import render_sidebar


def _load_sample(project: Project, relative_path: str) -> None:
    full_path = Path(relative_path)
    if not full_path.exists():
        full_path = Path.cwd() / relative_path
    if not full_path.exists():
        st.error(f"Sample not found: {relative_path}")
        return
    df = load_file(full_path)
    ingest_dataframe(project, df, source_name=full_path.name)
    st.success(f"Loaded sample: {full_path.name} ({len(df):,} rows × {len(df.columns)} cols)")


def render_home() -> None:
    render_sidebar()
    project: Project | None = st.session_state.get("project")

    if project is None:
        _render_landing()
        st.stop()

    st.title(f"{project.meta.get('name', 'Project')}")

    has_data = project.has_raw_data()
    task_cfg = get_task_config(project)
    enabled = get_enabled_libraries(project)
    st.caption(
        f"ID: {project.id}  •  Task: {task_cfg['task_type']}  •  "
        f"Target: {task_cfg.get('target_column') or '—'}  •  "
        f"Libraries: {len(enabled)}  •  "
        f"Data: {'loaded' if has_data else 'not loaded yet'}"
    )
    if project.meta.get("completed_stages"):
        link_col1, link_col2, _ = st.columns([1, 1, 3])
        with link_col1:
            st.page_link("pages/0_Results.py", label="Results summary", icon="✅")
        with link_col2:
            st.page_link("pages/3_Board.py", label="Board", icon="🗂️")

    if not has_data:
        _render_load_data(project)
        st.stop()

    df = project.get_working_data()

    with st.expander("Data preview & schema", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, height=280)
        st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
        schema = project.meta.get("schema", {})
        if schema:
            schema_df = pd.DataFrame([{"column": k, "type": v} for k, v in schema.items()])
            st.dataframe(schema_df, use_container_width=True, hide_index=True)
        render_dataset_download(project, key="home_dsdl")

    render_interactive_dashboard(project)
    render_pipeline_navigator(project)

    log_path = project.root / "run.log"
    if log_path.exists():
        with st.expander("run.log (last 30 lines)"):
            lines = log_path.read_text(
                encoding="utf-8", errors="replace").strip().splitlines()[-30:]
            st.code("\n".join(lines))


def _open_project(pid: str) -> None:
    try:
        st.session_state.project = Project.load(pid)
        st.rerun()
    except Exception as exc:
        st.error(f"Failed to open project: {exc}")


def _render_landing() -> None:
    from ui.home_data import get_home_summary
    from ui.project_picker import clear_project_cache
    from ui.theme import page_hero, pill, section_header

    summary = get_home_summary()

    page_hero(
        "MIRA",
        "Multi-source Ingest & Records Analytics — scrape and normalize data "
        "into records, then profile, model, and export. All on your machine.",
    )

    s1, s2, s3, s4 = st.columns(4)
    latest = summary["latest"]
    for col, k, v in (
        (s1, "Projects", f"{summary['n_projects']:,}"),
        (s2, "Records (global)", f"{summary['record_count']:,}"),
        (s3, "Registered sources", f"{summary['n_sources']:,}"),
        (s4, "Last activity", (latest["updated_at"] or "")[:10] if latest else "—"),
    ):
        col.markdown(
            f"<div class='mira-stat'><div class='v'>{v}</div>"
            f"<div class='k'>{k}</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    a1, a2, a3 = st.columns([1, 1, 2])
    with a1:
        with st.popover("➕ New analysis", use_container_width=True):
            with st.form("home_create_project", clear_on_submit=True):
                new_name = st.text_input("Project name", value="New Analysis")
                task_type = st.radio("Task type", ["regression", "classification"],
                                     horizontal=True, index=0)
                if st.form_submit_button("Create", type="primary") and new_name.strip():
                    try:
                        st.session_state.project = Project.create(
                            name=new_name.strip(), task_type=task_type)
                        clear_project_cache()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to create project: {exc}")
    with a2:
        if latest and st.button("Open latest", use_container_width=True):
            _open_project(latest["id"])
    with a3:
        l1, l2 = st.columns(2)
        with l1:
            st.page_link("pages/9_Campaigns.py", label="Browse campaigns", icon="📣")
        with l2:
            st.page_link("pages/8_Investigate.py", label="Start from a question",
                         icon="🔎")

    if latest:
        section_header("Latest project",
                       "Pick up where you left off — or open its results directly.")
        with st.container(border=True):
            c_info, c_open = st.columns([4, 1])
            with c_info:
                stages_pill = pill(f"{latest['n_stages_done']} stages done",
                                   "green" if latest["n_stages_done"] else "gray")
                task_pill = pill(latest.get("task_type") or "—")
                st.markdown(
                    f"**{latest['name']}** &nbsp; {task_pill} {stages_pill}",
                    unsafe_allow_html=True,
                )
                st.caption(f"id: {latest['id']}  ·  updated {latest['updated_at'] or '?'}")
            with c_open:
                if st.button("Open →", key="open_latest_card",
                             use_container_width=True):
                    _open_project(latest["id"])

    if summary["recent"]:
        section_header("Recent projects")
        cols = st.columns(3)
        for i, entry in enumerate(summary["recent"]):
            with cols[i % 3], st.container(border=True):
                st.markdown(f"**{entry['name']}**")
                st.caption(f"{entry['id']}  ·  "
                           f"{len(entry.get('completed_stages') or [])} stages")
                if st.button("Open", key=f"open_recent_{entry['id']}",
                             use_container_width=True):
                    _open_project(entry["id"])
    else:
        st.info("No projects yet — create one with **New analysis**, or scrape "
                "sources on **Sources & Records**.")


def _render_load_data(project: Project) -> None:
    st.header("Load data")
    try:
        acq_status = get_acquire_status()
        st.caption(f"Global **records** table: {acq_status['record_count']:,} rows")
    except Exception:
        st.caption("Global **records** table: unavailable (store busy)")

    with st.container(border=True):
        st.markdown("**No data, and not sure where to start?** Describe what you want "
                    "to investigate and MIRA will search the web, scrape sources, and "
                    "build a project with a consolidated schema.")
        st.page_link("pages/8_Investigate.py", label="🔎 Start from a question")

    col_acquire, col_upload, col_samples = st.columns([2, 2, 1])

    with col_acquire:
        st.subheader("From records (recommended)")
        try:
            reg_df = load_registry_df()
            source_ids = reg_df["source_id"].tolist()
        except Exception:
            source_ids = []

        if source_ids:
            default_idx = source_ids.index("REAL001") if "REAL001" in source_ids else 0
            picked = st.selectbox(
                "Registry source",
                source_ids,
                index=default_idx,
                key="home_acquire_source",
            )
            skip_scrape = st.checkbox("Use existing records (skip scrape)", key="home_skip_scrape")
            if st.button("Acquire → project", type="primary", use_container_width=True):
                try:
                    with st.spinner("Loading from records…"):
                        result = pull_source_into_project(
                            project,
                            picked,
                            scrape=not skip_scrape,
                            ingest_fn=lambda p, d, source_name: ingest_dataframe(
                                p, d, source_name=source_name
                            ),
                        )
                        if not project.meta.get("target_column"):
                            cols = result.get("columns", [])
                            if "primary_value" in cols:
                                set_task_config(
                                    project,
                                    project.meta.get("task_type", "regression"),
                                    "primary_value",
                                )
                    st.success(f"Loaded {result['rows']:,} rows from records")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("Add sources on **Sources & Records** page.")

    with col_upload:
        st.subheader("Upload file")
        uploaded = st.file_uploader(
            "CSV, Excel, Parquet, or JSON",
            type=["csv", "xlsx", "xls", "parquet", "json"],
        )
        if uploaded is not None:
            try:
                df = load_from_upload(uploaded)
                ingest_dataframe(project, df, source_name=uploaded.name)
                st.success(f"Loaded {len(df):,} rows from {uploaded.name}")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load file: {exc}")

    with col_samples:
        st.subheader("Samples")
        if st.button("Boston housing", use_container_width=True):
            _load_sample(project, "data/sample/boston_sample.csv")
            st.rerun()
        if st.button("Restaurant daily ops", use_container_width=True):
            _load_sample(project, "data/sample/restaurant_daily_sample.csv")
            st.rerun()
        if st.button("Stock (10 tickers)", use_container_width=True):
            _load_sample(project, "data/sample/stock_daily_features.csv")
            set_task_config(project, "regression", "fwd_ret_5d")
            st.rerun()
        for label, path in [
            ("Stock top 500", "data/sample/stock_daily_features_500.csv"),
            ("Stock top ~900", "data/sample/stock_daily_features_1000.csv"),
            ("Stock full panel", "data/sample/stock_daily_features_full_1000.parquet"),
        ]:
            if Path(path).exists() and st.button(label, use_container_width=True):
                _load_sample(project, path)
                set_task_config(project, "regression", "fwd_ret_5d")
                st.rerun()
