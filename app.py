"""
MIRA — Multi-source Ingest & Records Analytics
Streamlit entrypoint: the st.navigation router. Common chrome (page config,
theme, session init, the shared project picker) runs once here; page bodies
live in pages/ (thin wrappers) + ui/ (render functions). Nav structure is
declared in ui/nav.py.
"""

from __future__ import annotations

import streamlit as st

from ui.nav import build_navigation
from ui.project_picker import render_project_picker
from ui.theme import inject_theme

st.set_page_config(
    page_title="MIRA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

if "project" not in st.session_state:
    st.session_state.project = None

render_project_picker()

pg = st.navigation(build_navigation(), expanded=True)
pg.run()
