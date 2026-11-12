"""Every st.<name> the UI calls must exist in the pinned Streamlit.

The UI modules are only exercised by rendering them, which the test suite
doesn't do, so a call to an API newer than the pin (e.g. st.segmented_control
under streamlit==1.39.*) would otherwise surface only as a crash in the app.
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
_CALL = re.compile(r"\bst\.([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?")
# Namespaces whose second segment is itself part of the public API.
_NAMESPACES = {"column_config"}


def _ui_files():
    yield _ROOT / "app.py"
    for folder in ("ui", "pages"):
        yield from (_ROOT / folder).rglob("*.py")


def test_ui_only_uses_streamlit_apis_that_exist():
    missing = set()
    for f in _ui_files():
        for name, sub in _CALL.findall(f.read_text(encoding="utf-8")):
            if not hasattr(st, name):
                missing.add(f"{f.relative_to(_ROOT)}: st.{name}")
            elif name in _NAMESPACES and sub and not hasattr(getattr(st, name), sub):
                missing.add(f"{f.relative_to(_ROOT)}: st.{name}.{sub}")
    assert not missing, (
        f"Not in streamlit {st.__version__}: " + ", ".join(sorted(missing))
    )
