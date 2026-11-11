"""MIRA — Results: what a finished project produced, without re-running anything."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui.results_summary import render_results_summary  # noqa: E402

project = st.session_state.get("project")
if project is None:
    st.info("Select a project from the sidebar to see its results.")
    st.stop()

render_results_summary(project)
