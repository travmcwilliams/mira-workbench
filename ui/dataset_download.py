"""
ui/dataset_download.py — reusable "download this dataset" widget.

Dataset picker (working vs raw) + format picker + download button. Serialized
bytes are memoized in session_state per (project, dataset, format, data
fingerprint) so a Streamlit rerun doesn't re-serialize a large frame.
"""

from __future__ import annotations

import streamlit as st

from core.dataset_analysis import data_fingerprint
from core.dataset_export import FORMATS, export_bytes, export_filename, get_format
from core.project import Project


def render_dataset_download(project: Project, *, key: str = "dsdl") -> None:
    if not project.has_raw_data():
        return

    has_working = (project.root / "data" / "working.parquet").exists()
    datasets = (["working", "raw"] if has_working else ["raw"])

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        dataset = st.selectbox(
            "Dataset",
            datasets,
            format_func=lambda d: "Working (current)" if d == "working" else "Raw (as loaded)",
            key=f"{key}_dataset",
        )
    with c2:
        fmt_key = st.selectbox(
            "Format",
            [f.key for f in FORMATS],
            format_func=lambda k: get_format(k).label,
            key=f"{key}_format",
        )

    fp = data_fingerprint(project) or "raw"
    cache_key = f"{key}_bytes:{project.id}:{dataset}:{fmt_key}:{fp}"
    cache = st.session_state.setdefault("dataset_export_cache", {})

    if cache_key not in cache:
        df = project.get_raw_data() if dataset == "raw" else project.get_working_data()
        try:
            with st.spinner("Preparing file…"):
                cache.clear()  # keep at most one prepared export around
                cache[cache_key] = export_bytes(df, fmt_key)
        except ValueError as exc:
            st.warning(str(exc))
            return

    with c3:
        st.download_button(
            "⬇ Download",
            data=cache[cache_key],
            file_name=export_filename(
                str(project.meta.get("name") or project.id), dataset, fmt_key
            ),
            mime=get_format(fmt_key).mime,
            key=f"{key}_btn",
        )
