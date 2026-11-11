"""MIRA — interactive whiteboard walkthrough of the pipeline + model selection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui.walkthrough_page import render_walkthrough_page  # noqa: E402

render_walkthrough_page(st.session_state.get("project"))
