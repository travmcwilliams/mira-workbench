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


def _st_calls():
    """Yield (file, lineno, func, ast.Call) for every direct st.<func>(...) call."""
    import ast

    for f in _ui_files():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "st"):
                fn = getattr(st, node.func.attr, None)
                if callable(fn):
                    yield f, node.lineno, node.func.attr, fn, node


def test_ui_streamlit_keyword_arguments_exist():
    """Keyword args must exist on the pinned version, and widths must be ints:
    width="stretch"/"content" only arrived in later Streamlit releases."""
    import ast
    import inspect

    bad = []
    for f, line, name, fn, call in _st_calls():
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
        for kw in call.keywords:
            if kw.arg is None:
                continue  # **mapping
            where = f"{f.relative_to(_ROOT)}:{line} st.{name}"
            if kw.arg not in params and not takes_kwargs:
                bad.append(f"{where}({kw.arg}=)")
            if (kw.arg in ("width", "height") and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)):
                bad.append(f"{where}({kw.arg}={kw.value.value!r})")
    assert not bad, f"Unsupported on streamlit {st.__version__}: " + ", ".join(bad)
