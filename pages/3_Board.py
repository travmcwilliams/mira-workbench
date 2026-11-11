"""MIRA — per-project Kanban board, sprint planning & CI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui.kanban_board import render_kanban_board  # noqa: E402
from ui.sprint_panel import render_ci_panel, render_sprint_panel  # noqa: E402

project = st.session_state.get("project")
if project is None:
    st.info("Select a project from the sidebar to open its board.")
    st.stop()

board_tab, sprint_tab, ci_tab = st.tabs(["📋 Board", "🏃 Sprint", "⚙️ CI/CD"])
with board_tab:
    render_kanban_board(project)
with sprint_tab:
    render_sprint_panel(project)
with ci_tab:
    render_ci_panel(project)
