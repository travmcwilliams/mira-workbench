"""MIRA — discover scrapeable sources from a topic + fields."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from ui.discover_page import render_discover_page
from ui.theme import inject_theme

st.set_page_config(page_title="MIRA — Discover", layout="wide")
inject_theme()

if "project" not in st.session_state:
    st.session_state.project = None

render_discover_page()
