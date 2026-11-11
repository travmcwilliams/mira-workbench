"""
ui/investigate_page.py — "Start from a question": intent-driven project creation.

A guided wizard for users who have no data and only a goal. It walks:

    1. Intent   — pick a mode + describe the goal
    2. Plan     — review/edit the topic, search queries and fields
    3. Gather   — search the web + scrape candidates (live progress)
    4. Schema   — pick the coherent source cluster (or union all), prune columns,
                  choose a target, preview the data
    5. Create   — build the project and open it in the Notebook

Re-consolidation in step 4 reruns over the already-scraped frames, so changing
cluster/column choices never re-hits the web.
"""

from __future__ import annotations

import streamlit as st

from core import investigate as inv
from core import webschema
from ui.theme import page_hero, pill

# session_state keys
_PLAN = "inv_plan"
_GATHER = "inv_gather"
_CREATED = "inv_created"  # {"project_id", "rows", "cols", "target"}


def render_investigate() -> None:
    page_hero("🔎 Start from a question",
              "Describe what you want to investigate — MIRA finds data on the web, "
              "consolidates a schema, and builds a project.")

    _playwright_note()

    if _CREATED in st.session_state:
        _render_done()
        return
    if _GATHER in st.session_state:
        _render_schema_step()
        return
    if _PLAN in st.session_state:
        _render_plan_step()
        return
    _render_intent_step()


# --------------------------------------------------------------------------- #
# step 1 — intent
# --------------------------------------------------------------------------- #


def _render_intent_step() -> None:
    st.subheader("1 · What do you want to do?")
    mode = st.radio(
        "Intent", list(inv.MODES), horizontal=True, key="inv_mode",
        format_func=lambda m: {
            "investigate": "🔬 Investigate", "experiment": "🧪 Experiment",
            "predict": "🎯 Predict something", "explore": "🧭 Just explore",
        }.get(m, m),
    )
    goal = st.text_area(
        "Your goal (plain English)", key="inv_goal", height=90,
        placeholder="e.g. investigate what drives disc-golf course difficulty ratings",
    )

    with st.expander("Advanced"):
        c = st.columns(3)
        planner = c[0].selectbox("Query planner", ["offline", "ollama", "anthropic"],
                                 help="Use an LLM to expand the search queries "
                                 "(falls back to offline automatically).")
        target_n = c[1].slider("Candidates to gather", 20, 100, 60, step=10,
                               key="inv_target_n")
        max_scrape = c[2].slider("Max sites to scrape", 5, 60, 25, step=5,
                                 key="inv_max_scrape")
        time_budget = st.slider("Scrape time budget (seconds)", 30, 300, 150,
                                step=30, key="inv_time_budget")
        prefer = st.selectbox("Search backend", ["auto", "serper", "duckduckgo", "mojeek"],
                              help="auto = Serper if a key is set, else keyless")
        st.session_state["inv_cfg"] = {
            "planner": planner, "target_n": target_n, "max_scrape": max_scrape,
            "time_budget": float(time_budget),
            "prefer": None if prefer == "auto" else prefer,
        }

    if st.button("Plan →", type="primary", disabled=not goal.strip()):
        cfg = st.session_state.get("inv_cfg", {})
        with st.spinner("Planning search queries…"):
            plan = inv.plan_investigation(goal, mode, provider=cfg.get("planner", "offline"))
        st.session_state[_PLAN] = plan
        st.rerun()


# --------------------------------------------------------------------------- #
# step 2 — plan review
# --------------------------------------------------------------------------- #


def _render_plan_step() -> None:
    plan: inv.InvestigationPlan = st.session_state[_PLAN]
    st.subheader("2 · Review the plan")
    st.caption(f"Mode **{plan.mode}** · planner **{plan.provider}**")

    topic = st.text_input("Search topic", value=plan.topic, key="inv_topic_edit")
    queries = st.text_area("Search queries (one per line)",
                           value="\n".join(plan.queries), height=180,
                           key="inv_queries_edit")
    fields = st.text_input("Fields of interest (comma-separated)",
                           value=", ".join(plan.fields), key="inv_fields_edit")

    c = st.columns(2)
    if c[0].button("← Back"):
        _reset(_PLAN)
        st.rerun()
    if c[1].button("Gather data from the web →", type="primary"):
        plan.topic = topic.strip() or plan.topic
        plan.queries = [q.strip() for q in queries.splitlines() if q.strip()]
        plan.fields = [f.strip() for f in fields.split(",") if f.strip()]
        st.session_state[_PLAN] = plan
        _run_gather(plan)


def _run_gather(plan: inv.InvestigationPlan) -> None:
    cfg = st.session_state.get("inv_cfg", {})
    prog = st.progress(0.0, text="Searching…")

    def cb(msg: str, frac: float) -> None:
        prog.progress(min(max(frac, 0.0), 1.0), text=msg)

    try:
        g = inv.gather_frames(
            plan,
            search_prefer=cfg.get("prefer"),
            target_n=int(cfg.get("target_n", 60)),
            max_scrape=int(cfg.get("max_scrape", 25)),
            time_budget_s=float(cfg.get("time_budget", 150.0)),
            progress_cb=cb,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gather failed: {exc}")
        return
    prog.empty()
    st.session_state[_GATHER] = g
    st.rerun()


# --------------------------------------------------------------------------- #
# step 3 — schema review + consolidation choices
# --------------------------------------------------------------------------- #


def _render_schema_step() -> None:
    g: inv.GatherResult = st.session_state[_GATHER]
    plan: inv.InvestigationPlan = st.session_state[_PLAN]

    st.subheader("3 · Review gathered data")
    st.markdown(
        pill(f"{g.provider}") + pill(f"{g.n_candidates} candidates")
        + pill(f"scraped {g.n_scraped}") + pill(f"{g.n_with_rows} with tables", "green"),
        unsafe_allow_html=True)

    with st.expander(f"Per-source detail ({len(g.per_source)})"):
        st.dataframe(g.per_source, use_container_width=True, hide_index=True)

    if not g.frames:
        st.warning("No source produced a usable table. Go back and broaden the "
                   "topic/queries, raise the candidate count, or add a search key.")
        if st.button("← Back to plan"):
            _reset(_GATHER)
            st.rerun()
        return

    # A first coherent pass gives us the cluster options to choose from.
    _, base = webschema.consolidate(g.frames, task_type=plan.task_type)
    clusters = base.clusters

    st.markdown("**Which sources should form the schema?**")
    opts = [f"Best cluster · {clusters[0]['n_frames']} src, {clusters[0]['rows']} rows"] \
        if clusters else []
    for i, cl in enumerate(clusters[1:], start=1):
        opts.append(f"Cluster {i + 1} · {cl['n_frames']} src, {cl['rows']} rows")
    opts.append("Union ALL sources (messier, wider)")
    choice = st.radio("Source selection", list(range(len(opts))),
                      format_func=lambda i: opts[i], key="inv_cluster_choice")

    if choice == len(opts) - 1:  # union all
        combined, report = webschema.consolidate(
            g.frames, task_type=plan.task_type, strategy="union_all")
    else:
        keep = clusters[choice]["indices"] if clusters else None
        combined, report = webschema.consolidate(
            g.frames, task_type=plan.task_type, keep_indices=keep)

    # Optional column prune.
    all_cols = list(combined.columns)
    with st.expander("Prune columns (optional)"):
        keep_cols = st.multiselect("Columns to keep", all_cols, default=all_cols,
                                   key="inv_keep_cols")
        if keep_cols and set(keep_cols) != set(all_cols):
            if choice == len(opts) - 1:
                combined, report = webschema.consolidate(
                    g.frames, task_type=plan.task_type, strategy="union_all",
                    keep_columns=keep_cols)
            else:
                combined, report = webschema.consolidate(
                    g.frames, task_type=plan.task_type,
                    keep_indices=clusters[choice]["indices"] if clusters else None,
                    keep_columns=keep_cols)

    _schema_preview(combined, report)

    # target override
    cols = list(combined.columns)
    default_t = report.target_column if report.target_column in cols else None
    target = st.selectbox(
        "Target column (what you'd model/measure)", ["(none)"] + cols,
        index=(cols.index(default_t) + 1) if default_t else 0, key="inv_target")
    target_col = None if target == "(none)" else target

    name = st.text_input("Project name",
                         value=f"{plan.mode.title()}: {plan.topic}"[:60],
                         key="inv_proj_name")

    c = st.columns(2)
    if c[0].button("← Back to plan"):
        _reset(_GATHER)
        st.rerun()
    if c[1].button("Create project →", type="primary"):
        if target_col:
            report.target_column = target_col
        pid = inv.build_project(plan, combined, report, name.strip() or None)
        st.session_state[_CREATED] = {
            "project_id": pid, "rows": report.n_rows, "cols": len(cols),
            "target": target_col,
        }
        st.rerun()


def _schema_preview(combined, report: webschema.ConsolidationReport) -> None:
    st.markdown(f"**Proposed schema** — {report.n_rows:,} rows × "
                f"{len(combined.columns)} cols  "
                f"({report.n_sources} of {report.n_source_frames} sources"
                + (f", {report.dropped_incoherent} dropped as incoherent"
                   if report.dropped_incoherent else "") + ")")
    schema_rows = [{"column": c.name, "type": c.dtype, "sources": c.sources}
                   for c in report.columns]
    st.dataframe(schema_rows, use_container_width=True, hide_index=True, height=240)
    with st.expander("Data preview"):
        st.dataframe(combined.head(20), use_container_width=True)


# --------------------------------------------------------------------------- #
# step 4 — done
# --------------------------------------------------------------------------- #


def _render_done() -> None:
    info = st.session_state[_CREATED]
    st.success(f"Created project **{info['project_id']}** — {info['rows']:,} rows × "
               f"{info['cols']} cols"
               + (f", target `{info['target']}`" if info.get("target") else ""))

    from core.project import Project
    proj = Project.load(info["project_id"])

    c = st.columns(3)
    if c[0].button("📓 Open in Notebook", type="primary"):
        st.session_state.project = proj
        st.switch_page("pages/7_Notebook.py")
    if c[1].button("📊 Open project (Home)"):
        st.session_state.project = proj
        st.switch_page("pages/home.py")
    if c[2].button("🔎 Start another"):
        for k in (_PLAN, _GATHER, _CREATED):
            _reset(k)
        st.rerun()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _reset(key: str) -> None:
    st.session_state.pop(key, None)
    # dropping a later stage should also drop everything downstream of it
    order = [_PLAN, _GATHER, _CREATED]
    if key in order:
        for k in order[order.index(key):]:
            st.session_state.pop(k, None)


def _playwright_note() -> None:
    try:
        from core.acquire_bridge import get_fetcher_status
        if not get_fetcher_status().get("playwright_available"):
            st.info("Playwright isn't available — JS-rendered pages will be skipped. "
                    "Install it with `python -m playwright install` for best coverage.")
    except Exception:
        pass
