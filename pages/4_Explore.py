"""MIRA — notebook-style dataset exploration with concept-driven charts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui.explore_page import render_explore_page  # noqa: E402

project = st.session_state.get("project")
if project is None:
    st.info("Select a project from the sidebar to explore its dataset.")
    st.stop()

render_explore_page(project)
