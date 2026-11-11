"""
ui/notebook_page.py — a Jupyter-style notebook with an AI copilot.

Each cell is an editable Python segment that runs over the project's dataset in a
persistent kernel (`df`, `pd`, `np`, `plt`, `px`). Every cell has an inline
"Ask AI" box: describe a change in plain English and the model rewrites that cell
against the live dataset schema; you review and apply. A global chat at the bottom
can spin up a brand-new cell from a description. Cells persist to the project.
"""

from __future__ import annotations

import streamlit as st

from core import chat as chatmod
from core import nb_suggest
from core import notebook as nb
from core.project import Project
from ui.theme import page_hero

# session_state keys are namespaced per project so switching projects is clean.
_KERNEL = "nb_kernel"
_KERNEL_PID = "nb_kernel_pid"
_CELLS = "nb_cells"
_CELLS_PID = "nb_cells_pid"
_RESULTS = "nb_results"          # {cell_id: CellResult}
_PENDING = "nb_pending_edit"     # {cell_id: proposed_code}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def render_notebook(project: Project) -> None:
    page_hero(f"📓 Notebook — {project.meta.get('name', project.id)}",
              "Edit code cells, run them over your dataset, and let AI rewrite any "
              "segment from a plain-English request.")

    if not project.has_raw_data():
        st.info("This project has no data yet. Load a dataset from the Home page first.")
        return

    df = project.get_working_data()
    target = project.meta.get("target_column")
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns"
               + (f"  ·  target `{target}`" if target else ""))

    _ensure_state(project, df)
    chosen, runtime_key = _model_palette()

    _toolbar(project, df)
    _render_toolbelt(project, df, target)
    _render_suggestions(project, df, target, chosen, runtime_key)
    st.divider()

    cells: list[nb.NotebookCell] = st.session_state[_CELLS]
    for i, cell in enumerate(list(cells)):
        _render_cell(project, df, cell, i, len(cells), chosen, runtime_key)

    if st.button("➕ Add cell", key="nb_add"):
        cells.append(nb.new_cell())
        _persist(project)
        st.rerun()

    st.divider()
    _global_chat(project, df, chosen, runtime_key)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #


def _ensure_state(project: Project, df: object) -> None:
    """(Re)initialise kernel + cells when the project changes."""
    if st.session_state.get(_CELLS_PID) != project.id:
        st.session_state[_CELLS] = nb.load_cells(project)
        st.session_state[_CELLS_PID] = project.id
        st.session_state[_RESULTS] = {}
        st.session_state[_PENDING] = {}
    if st.session_state.get(_KERNEL_PID) != project.id \
            or _KERNEL not in st.session_state:
        st.session_state[_KERNEL] = nb.Kernel(df)
        st.session_state[_KERNEL_PID] = project.id


def _persist(project: Project) -> None:
    nb.save_cells(project, st.session_state[_CELLS])


def _run_cell(cell: nb.NotebookCell) -> None:
    kernel: nb.Kernel = st.session_state[_KERNEL]
    st.session_state[_RESULTS][cell.id] = kernel.execute(cell.source)


def _insert_cell(project: Project, code: str, *, run: bool = False) -> None:
    """Append a new cell with `code`, optionally run it, persist, and rerun."""
    cell = nb.new_cell(code)
    st.session_state[_CELLS].append(cell)
    if run:
        _run_cell(cell)
    _persist(project)
    st.rerun()


# --------------------------------------------------------------------------- #
# method pills + AI suggestions
# --------------------------------------------------------------------------- #


def _render_toolbelt(project: Project, df: object, target: str | None) -> None:
    """A palette of common data-analysis methods; click a pill to insert a cell."""
    schema = project.meta.get("schema")
    lib = nb_suggest.library_for(df, schema, target)
    with st.expander("🧰 Insert a method  ·  common data-analysis snippets", expanded=False):
        tabs = st.tabs(list(lib.keys()))
        for tab, (cat, items) in zip(tabs, lib.items(), strict=True):
            with tab:
                cols = st.columns(3)
                for j, m in enumerate(items):
                    if cols[j % 3].button(m["label"], key=f"nb_pill_{cat}_{j}",
                                          use_container_width=True,
                                          help=m["code"][:200]):
                        _insert_cell(project, m["code"])


def _render_suggestions(project: Project, df: object, target: str | None,
                        chosen: chatmod.ChatModel | None,
                        runtime_key: str | None) -> None:
    """Context-aware next-step suggestions (RAG + model, or offline recipes)."""
    with st.expander("✨ Suggest next steps  ·  reads your data + notebook", expanded=False):
        src = (f"tailored by {chosen.label}" if chosen
               else "offline recipes — pick a model in the sidebar for tailored ideas")
        c1, c2 = st.columns([3, 1])
        c1.caption(f"Suggestions {src}.")
        if c2.button("Generate", key="nb_sugg_go", use_container_width=True):
            with st.spinner("Reading your notebook and dataset…"):
                st.session_state["nb_suggestions"] = nb_suggest.suggest(
                    df=df, schema=project.meta.get("schema"), target=target,
                    task_type=project.meta.get("task_type"),
                    cells=st.session_state[_CELLS],
                    results=st.session_state.get(_RESULTS, {}),
                    chat_model=chosen, runtime_key=runtime_key, n=5,
                )

        suggestions = st.session_state.get("nb_suggestions") or []
        if not suggestions:
            st.caption("No suggestions yet — click **Generate**.")
            return
        for k, s in enumerate(suggestions):
            with st.container(border=True):
                st.markdown(f"**{s['title']}**")
                st.code(s["code"], language="python")
                b1, b2 = st.columns(2)
                if b1.button("➕ Insert", key=f"nb_sugg_ins_{k}"):
                    _insert_cell(project, s["code"])
                if b2.button("▶ Insert & run", key=f"nb_sugg_run_{k}"):
                    _insert_cell(project, s["code"], run=True)


# --------------------------------------------------------------------------- #
# model palette (mirrors the Scraping Studio convention)
# --------------------------------------------------------------------------- #


def _model_palette() -> tuple[chatmod.ChatModel | None, str | None]:
    runtime_key = st.session_state.get("nb_api_key") or None
    models = chatmod.available_models(runtime_key=runtime_key)
    with st.sidebar:
        st.markdown("#### 🤖 Notebook copilot")
        if not models:
            st.warning("No chat model. Start Ollama or add an Anthropic key.")
            return None, runtime_key
        ids = [m.id for m in models]
        by_id = {m.id: m for m in models}
        default = chatmod.pick_default(models)
        cur = st.session_state.get("nb_model_id")
        if cur not in ids:
            cur = default.id if default else ids[0]
        st.session_state["nb_model_id"] = st.selectbox(
            "Model", ids, index=ids.index(cur) if cur in ids else 0,
            format_func=lambda mid: f"{by_id[mid].icon} {by_id[mid].label}"
            + ("" if by_id[mid].ready else f"  · {by_id[mid].note}"),
            key="nb_model_select",
        )
        chosen = by_id.get(st.session_state["nb_model_id"])
        if chosen and chosen.provider == "anthropic" and not chosen.ready:
            k = st.text_input("Anthropic API key", type="password", key="nb_key_input")
            if st.button("Use key", key="nb_key_save") and k:
                st.session_state["nb_api_key"] = k.strip()
                st.rerun()
    return chosen, runtime_key


# --------------------------------------------------------------------------- #
# toolbar
# --------------------------------------------------------------------------- #


def _toolbar(project: Project, df: object) -> None:
    c = st.columns(4)
    if c[0].button("▶▶ Run all", key="nb_run_all", use_container_width=True):
        for cell in st.session_state[_CELLS]:
            _run_cell(cell)
        st.rerun()
    if c[1].button("↻ Restart kernel", key="nb_restart", use_container_width=True,
                   help="Clear all variables and outputs, reseed `df`."):
        st.session_state[_KERNEL] = nb.Kernel(df)
        st.session_state[_RESULTS] = {}
        st.rerun()
    if c[2].button("💾 Save", key="nb_save", use_container_width=True):
        _persist(project)
        st.toast("Notebook saved.")
    if c[3].button("⬇ Export .ipynb", key="nb_export", use_container_width=True):
        path = _export_ipynb(project)
        st.session_state["nb_ipynb_path"] = str(path)

    ipynb = st.session_state.get("nb_ipynb_path")
    if ipynb:
        try:
            with open(ipynb, "rb") as fh:
                st.download_button("Download notebook.ipynb", fh.read(),
                                   file_name="notebook.ipynb",
                                   mime="application/x-ipynb+json", key="nb_dl")
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# a single cell
# --------------------------------------------------------------------------- #


def _render_cell(project: Project, df: object, cell: nb.NotebookCell, idx: int,
                 total: int, chosen: chatmod.ChatModel | None,
                 runtime_key: str | None) -> None:
    with st.container(border=True):
        head = st.columns([0.55, 0.09, 0.09, 0.09, 0.09, 0.09])
        head[0].caption(f"Cell [{idx + 1}]")
        run = head[1].button("▶", key=f"nb_run_{cell.id}", help="Run cell")
        up = head[2].button("↑", key=f"nb_up_{cell.id}", help="Move up",
                            disabled=idx == 0)
        down = head[3].button("↓", key=f"nb_down_{cell.id}", help="Move down",
                              disabled=idx == total - 1)
        add = head[4].button("＋", key=f"nb_ins_{cell.id}", help="Insert below")
        delete = head[5].button("🗑", key=f"nb_del_{cell.id}", help="Delete cell")

        new_src = st.text_area("code", value=cell.source, height=140,
                               key=f"nb_src_{cell.id}", label_visibility="collapsed")
        if new_src != cell.source:
            cell.source = new_src  # keep edits without forcing a rerun

        _render_ai_box(cell, df, project, chosen, runtime_key)

        # structural actions
        cells: list[nb.NotebookCell] = st.session_state[_CELLS]
        if up and idx > 0:
            cells[idx - 1], cells[idx] = cells[idx], cells[idx - 1]
            _persist(project)
            st.rerun()
        if down and idx < total - 1:
            cells[idx + 1], cells[idx] = cells[idx], cells[idx + 1]
            _persist(project)
            st.rerun()
        if add:
            cells.insert(idx + 1, nb.new_cell())
            _persist(project)
            st.rerun()
        if delete:
            cells.pop(idx)
            st.session_state[_RESULTS].pop(cell.id, None)
            _persist(project)
            st.rerun()
        if run:
            _run_cell(cell)
            _persist(project)

        _render_result(cell)


def _render_ai_box(cell: nb.NotebookCell, df: object, project: Project,
                   chosen: chatmod.ChatModel | None, runtime_key: str | None) -> None:
    # A proposed edit awaiting review?
    pending = st.session_state[_PENDING].get(cell.id)
    if pending is not None:
        st.caption("✨ Proposed edit — review and apply:")
        st.code(pending, language="python")
        a, b = st.columns(2)
        if a.button("✅ Apply", key=f"nb_apply_{cell.id}"):
            cell.source = pending
            st.session_state[_PENDING].pop(cell.id, None)
            _persist(project)
            st.rerun()
        if b.button("✖ Discard", key=f"nb_discard_{cell.id}"):
            st.session_state[_PENDING].pop(cell.id, None)
            st.rerun()
        return

    with st.expander("✨ Ask AI to edit this cell"):
        instr = st.text_input("Describe the change", key=f"nb_ai_{cell.id}",
                              placeholder="e.g. group by month and plot the mean price")
        if st.button("Rewrite cell", key=f"nb_ai_go_{cell.id}"):
            if not chosen:
                st.error("No chat model selected (see sidebar).")
            elif instr.strip():
                code = _ask_ai(instr, cell.source, df, project, chosen, runtime_key,
                               is_new=not cell.source.strip())
                st.session_state[_PENDING][cell.id] = code
                st.rerun()


def _render_result(cell: nb.NotebookCell) -> None:
    res = st.session_state[_RESULTS].get(cell.id)
    if res is None:
        return
    if res.stdout:
        st.code(res.stdout)
    for fig in res.figures:
        st.pyplot(fig)
    if res.value is not None:
        _render_value(res.value)
    if res.error:
        st.error(res.error)
    elif not res.has_output:
        st.caption("↳ ran (no output)")


def _render_value(value: object) -> None:
    """Render a trailing-expression value by type, notebook-style."""
    import pandas as pd

    try:
        import plotly.graph_objects as go
        if isinstance(value, go.Figure):
            st.plotly_chart(value, use_container_width=True)
            return
    except Exception:
        pass
    if isinstance(value, (pd.DataFrame, pd.Series)):
        st.dataframe(value if isinstance(value, pd.DataFrame) else value.to_frame(),
                     use_container_width=True)
        return
    # matplotlib Axes/Figure returned directly
    fig = getattr(value, "figure", None)
    if fig is not None and value.__class__.__module__.startswith("matplotlib"):
        st.pyplot(fig)
        return
    st.write(value)


# --------------------------------------------------------------------------- #
# global chat → new cell
# --------------------------------------------------------------------------- #


def _global_chat(project: Project, df: object, chosen: chatmod.ChatModel | None,
                 runtime_key: str | None) -> None:
    st.subheader("💬 Ask the notebook copilot")
    st.caption("Describe an analysis and it becomes a new cell at the bottom.")
    prompt = st.chat_input("e.g. show the correlation heatmap of numeric columns",
                           key="nb_global_chat")
    if prompt:
        if not chosen:
            st.error("No chat model selected (see sidebar).")
            return
        with st.spinner("Writing a cell…"):
            code = _ask_ai(prompt, "", df, project, chosen, runtime_key, is_new=True)
        cell = nb.new_cell(code)
        st.session_state[_CELLS].append(cell)
        _run_cell(cell)
        _persist(project)
        st.rerun()


# --------------------------------------------------------------------------- #
# AI + export helpers
# --------------------------------------------------------------------------- #


def _ask_ai(instruction: str, current_code: str, df: object, project: Project,
            chosen: chatmod.ChatModel, runtime_key: str | None, *,
            is_new: bool) -> str:
    ctx = nb.dataset_context(df, project.meta.get("schema"),
                             project.meta.get("target_column"))
    messages = nb.build_edit_messages(instruction, current_code, ctx, is_new=is_new)
    reply = chatmod.complete_chat(
        messages, provider=chosen.provider, model=chosen.model,
        api_key=runtime_key, system=nb.AI_SYSTEM, max_tokens=1200,
    )
    return nb.extract_code(reply)


def _export_ipynb(project: Project) -> object:
    import nbformat

    nbk = nbformat.v4.new_notebook()
    setup = ("import pandas as pd, numpy as np\n"
             "import matplotlib.pyplot as plt\n"
             "try:\n    import plotly.express as px\nexcept Exception:\n    px = None\n"
             "df = pd.read_parquet('data/working.parquet')")
    nbk.cells.append(nbformat.v4.new_code_cell(setup))
    for cell in st.session_state[_CELLS]:
        nbk.cells.append(nbformat.v4.new_code_cell(cell.source))
    out = project.root / "exports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "notebook.ipynb"
    with open(path, "w", encoding="utf-8") as fh:
        nbformat.write(nbk, fh)
    return path
