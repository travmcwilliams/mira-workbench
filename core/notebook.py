"""
core/notebook.py — a tiny in-process notebook engine for MIRA.

This is the UI-agnostic core behind the Notebook page: a persistent-namespace
"kernel" plus cell persistence and the AI-editing helpers. It deliberately does
**not** import Streamlit so it stays unit-testable and reusable.

The kernel mimics Jupyter/IPython display semantics: a cell's statements are
exec'd in a shared namespace and, if the cell ends in an expression, that value
is captured and returned so the UI can render it (a DataFrame as a table, a
Plotly/Matplotlib figure as a chart, anything else via ``st.write``). Stdout and
tracebacks are captured as strings; nothing raises into the caller.

Persistence is a single ``notebook.json`` under the project folder holding the
ordered cell sources — outputs are runtime-only and never written to disk.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Cells + results
# --------------------------------------------------------------------------- #


@dataclass
class NotebookCell:
    """One editable code segment. ``outputs`` is runtime-only (not persisted)."""

    id: str
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NotebookCell:
        return cls(id=str(d.get("id") or new_cell_id()), source=str(d.get("source", "")))


@dataclass
class CellResult:
    """The captured effect of running one cell."""

    stdout: str = ""
    error: str | None = None
    value: Any = None
    figures: list[Any] = field(default_factory=list)  # matplotlib Figure objects

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def has_output(self) -> bool:
        return bool(self.stdout) or self.error is not None or self.value is not None \
            or bool(self.figures)


def new_cell_id() -> str:
    return uuid.uuid4().hex[:8]


def new_cell(source: str = "") -> NotebookCell:
    return NotebookCell(id=new_cell_id(), source=source)


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #


class Kernel:
    """A persistent execution namespace shared across all cells of a notebook.

    Preloads the analysis toolkit (``pd``, ``np``, ``plt``, and ``px`` if
    Plotly is present) plus the project's working dataframe as ``df``. Because
    the namespace persists between runs, variables defined in one cell are
    visible in the next — exactly like a Jupyter kernel.
    """

    def __init__(self, df: Any = None) -> None:
        self.ns: dict[str, Any] = {}
        self.reset(df)

    def reset(self, df: Any = None) -> None:
        """Rebuild a fresh namespace, seeding ``df`` and the toolkit."""
        import numpy as np
        import pandas as pd

        ns: dict[str, Any] = {"__name__": "__mira_notebook__", "pd": pd, "np": np}
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ns["plt"] = plt
        except Exception:  # pragma: no cover - matplotlib is a hard dep, defensive
            pass
        try:
            import plotly.express as px

            ns["px"] = px
        except Exception:  # pragma: no cover
            pass
        if df is not None:
            ns["df"] = df
        self.ns = ns

    def execute(self, code: str) -> CellResult:
        """Run ``code`` in the shared namespace, capturing output IPython-style."""
        result = CellResult()
        if not code.strip():
            return result

        # Close any figures the *previous* cell left open so we only capture
        # figures this cell actually creates.
        plt = self.ns.get("plt")
        pre_fignums: set[int] = set()
        if plt is not None:
            pre_fignums = set(plt.get_fignums())

        try:
            tree = ast.parse(code, filename="<cell>", mode="exec")
        except SyntaxError:
            result.error = _format_exc()
            return result

        last_expr: ast.Expression | None = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_node = tree.body.pop()
            assert isinstance(last_node, ast.Expr)
            last_expr = ast.Expression(last_node.value)
            ast.copy_location(last_expr, last_node)

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                if tree.body:
                    exec(compile(tree, "<cell>", "exec"), self.ns)  # noqa: S102
                if last_expr is not None:
                    result.value = eval(  # noqa: S307 - local power-user tool
                        compile(last_expr, "<cell>", "eval"), self.ns
                    )
        except Exception:
            result.stdout = buf.getvalue()
            result.error = _format_exc()
            return result

        result.stdout = buf.getvalue()

        # Capture any new matplotlib figures this cell opened.
        if plt is not None:
            new_nums = [n for n in plt.get_fignums() if n not in pre_fignums]
            result.figures = [plt.figure(n) for n in new_nums]
        return result


def _format_exc() -> str:
    """A compact traceback string with the harness frames trimmed away."""
    tb = traceback.format_exc()
    # Drop the two internal frames (execute + compile) so the user sees their code.
    lines = tb.splitlines()
    cleaned = [ln for ln in lines if "core/notebook.py" not in ln
               and "core\\notebook.py" not in ln]
    return "\n".join(cleaned) if cleaned else tb


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

NOTEBOOK_VERSION = 1


def notebook_path(project: Any) -> Path:
    return Path(project.root) / "notebook.json"


def load_cells(project: Any) -> list[NotebookCell]:
    """Load persisted cells, or a friendly starter notebook if none exist."""
    path = notebook_path(project)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cells = [NotebookCell.from_dict(c) for c in data.get("cells", [])]
            if cells:
                return cells
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return default_cells()


def save_cells(project: Any, cells: list[NotebookCell]) -> Path:
    path = notebook_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": NOTEBOOK_VERSION, "cells": [c.to_dict() for c in cells]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def default_cells() -> list[NotebookCell]:
    return [
        new_cell("# `df` is your dataset. `pd`, `np`, `plt`, `px` are ready.\n"
                 "df.head()"),
        new_cell("df.describe(include='all').T"),
    ]


# --------------------------------------------------------------------------- #
# AI editing
# --------------------------------------------------------------------------- #

AI_SYSTEM = (
    "You are MIRA's notebook coding copilot. You edit and write Python cells that "
    "analyze a pandas DataFrame already loaded as `df`. Available names: `df`, `pd` "
    "(pandas), `np` (numpy), `plt` (matplotlib.pyplot, Agg backend), `px` "
    "(plotly.express). Variables persist across cells like a Jupyter kernel.\n\n"
    "RULES:\n"
    "- Reply with ONE fenced ```python code block and nothing else — no prose, no "
    "explanation outside the block. A short leading comment inside the block is fine.\n"
    "- End the cell in a bare expression when you want a result shown (e.g. a "
    "DataFrame, a `px.*` figure, or a value). Assign matplotlib plots to `fig` or "
    "just call plt.* — figures are captured automatically.\n"
    "- Do not read files or hit the network; operate on `df`. Never call plt.show()."
)


def dataset_context(df: Any, schema: Any = None, target: str | None = None) -> str:
    """A compact, token-cheap description of the dataframe for the model."""
    if df is None:
        return "No dataset is loaded."
    try:
        n_rows, n_cols = df.shape
    except Exception:
        return "A dataframe named `df` is available."
    lines = [f"DataFrame `df`: {n_rows:,} rows x {n_cols} columns.",
             "Columns (name: dtype):"]
    for col in list(df.columns)[:60]:
        lines.append(f"  - {col}: {df[col].dtype}")
    if len(df.columns) > 60:
        lines.append(f"  … and {len(df.columns) - 60} more.")
    if target:
        lines.append(f"Target column: {target}")
    return "\n".join(lines)


_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the first fenced code block from an LLM reply, else return it trimmed."""
    m = _CODE_FENCE.search(text)
    if m:
        return m.group(1).strip("\n")
    return text.strip().strip("`").strip()


def build_edit_messages(
    instruction: str,
    current_code: str,
    ctx: str,
    *,
    is_new: bool = False,
) -> list[dict[str, str]]:
    """Construct the chat messages for an AI cell edit / generation request."""
    if is_new or not current_code.strip():
        user = (f"{ctx}\n\nWrite a new notebook cell that does the following:\n"
                f"{instruction}")
    else:
        user = (f"{ctx}\n\nHere is the current cell:\n```python\n{current_code}\n```\n\n"
                f"Revise it so that: {instruction}\n"
                "Return the full updated cell.")
    return [{"role": "user", "content": user}]
