"""MIRA — AI notebook: editable code cells over your dataset with a chat copilot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui.notebook_page import render_notebook  # noqa: E402

project = st.session_state.get("project")
if project is None:
    st.info("Select a project from the sidebar to open its notebook.")
    st.stop()

render_notebook(project)
