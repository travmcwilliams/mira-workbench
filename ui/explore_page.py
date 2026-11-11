"""
ui/explore_page.py — notebook-style dataset exploration.

Each suggested analysis renders as a notebook cell: the code that produced it
plus the chart most useful for that concept. Results are memoized per (recipe,
data fingerprint) so heavy analyses don't re-run on every Streamlit interaction.
A free-form code cell and a one-click .ipynb export are included.
"""

from __future__ import annotations

import contextlib
import io

import streamlit as st

from core import explore
from core.dataset_analysis import data_fingerprint
from core.project import Project
from ui.dataset_download import render_dataset_download
from ui.theme import page_hero

# Sensible default selection (lighter analyses first).
_DEFAULT_KEYS = ["summary", "correlation", "covariance", "pca", "kmeans"]


def render_explore_page(project: Project) -> None:
    page_hero(f"📓 Explore — {project.meta.get('name', project.id)}",
              "Notebook-style exploration with concept-driven charts.")

    if not project.has_raw_data():
        st.info("This project has no data yet. Load data first.")
        return

    df = project.get_working_data()
    schema = project.meta.get("schema")
    target = project.meta.get("target_column")
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns"
               + (f"  ·  target `{target}`" if target else ""))

    with st.expander("⬇ Download dataset"):
        render_dataset_download(project, key="explore_dsdl")

    recipes = explore.suggest_recipes(df, schema, target)
    by_key = {r.key: r for r in recipes}
    titles = {r.key: r.title for r in recipes}

    chosen = st.multiselect(
        "Analyses (each renders as a notebook cell)",
        options=[r.key for r in recipes],
        default=[k for k in _DEFAULT_KEYS if k in by_key],
        format_func=lambda k: titles.get(k, k),
    )

    top = st.columns(3)
    with top[0]:
        run_all = st.button("▶ Run selected", type="primary")
    with top[1]:
        if st.button("↻ Clear outputs"):
            st.session_state.pop("explore_cache", None)
            st.rerun()
    with top[2]:
        if st.button("⬇ Export .ipynb"):
            path = explore.export_notebook(project, [by_key[k] for k in chosen] or recipes)
            st.session_state["explore_nb_path"] = str(path)

    nb_path = st.session_state.get("explore_nb_path")
    if nb_path:
        try:
            with open(nb_path, "rb") as fh:
                st.download_button("Download exploration.ipynb", fh.read(),
                                   file_name="exploration.ipynb",
                                   mime="application/x-ipynb+json")
        except OSError:
            pass

    fp = data_fingerprint(project) or ""
    cache = st.session_state.setdefault("explore_cache", {})

    for key in chosen:
        recipe = by_key[key]
        _render_cell(df, recipe, cache, fp, run_all)

    st.divider()
    _render_freeform(df)


def _render_cell(df, recipe: explore.Recipe, cache: dict, fp: str, run_all: bool) -> None:
    with st.container(border=True):
        st.markdown(f"**{recipe.title}**  ·  _{recipe.concept}_")
        st.code(recipe.code, language="python")

        cache_key = f"{recipe.key}:{fp}"
        if run_all and cache_key not in cache:
            with st.spinner(f"Running {recipe.title}…"):
                cache[cache_key] = explore.run_recipe(df, recipe)
        elif st.button("▶ Run", key=f"run_{recipe.key}"):
            with st.spinner(f"Running {recipe.title}…"):
                cache[cache_key] = explore.run_recipe(df, recipe)

        res = cache.get(cache_key)
        if res is None:
            st.caption("Not run yet.")
            return
        if res.status == "error":
            st.error(res.error)
            return
        if res.text:
            st.caption(res.text)
        if res.fig is not None:
            st.pyplot(res.fig)
        if res.table is not None:
            st.dataframe(res.table, use_container_width=True)


def _render_freeform(df) -> None:
    st.subheader("🧪 Free-form cell")
    st.caption("Runs locally over `df` (also: `pd`, `np`, `plt`). Assign `fig` to show a chart.")
    code = st.text_area("Code", value="df.describe()", height=120, key="freeform_code")
    if st.button("▶ Run cell", key="run_freeform"):
        _exec_freeform(df, code)


def _exec_freeform(df, code: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ns = {"df": df, "pd": pd, "np": np, "plt": plt, "fig": None}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            # Show the value of a trailing expression, notebook-style.
            try:
                result = eval(code, ns)  # noqa: S307 — local power-user tool
            except SyntaxError:
                result = None
                exec(code, ns)  # noqa: S102
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        return

    out = buf.getvalue()
    if out:
        st.code(out)
    if ns.get("fig") is not None:
        st.pyplot(ns["fig"])
    elif plt.get_fignums():
        st.pyplot(plt.gcf())
    if "result" in dir() and result is not None:
        st.write(result)
