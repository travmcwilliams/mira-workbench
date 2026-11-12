"""
ui/scraping_studio.py — MIRA "Scraping Studio".

A dark, icon-driven workbench for the *acquire* side of a data-transformation
project, with an embedded AI chat:

  • Smart input  — one box that auto-detects a research topic, a blob of URLs, or
    pasted data, and routes it to the right tool.
  • Discover     — web source discovery (Serper/DDG) with live scraping
    visualizations (relevance score, field coverage, scraper-type & reachability
    mix) → pick sources → build a pipeline-ready project.
  • URL compiler — paste/collect URLs, probe each (status, tables, rows) → compile
    the good ones into one combined dataset/project.
  • Activity / Records — scraper-run health and what's already in the warehouse.
  • Claude chat  — streaming assistant (Claude API when a key is present, else any
    local Ollama chat model) with a model **palette** and the studio state as
    context.

All network/parse calls are guarded so the page degrades to messages, never a
stack trace.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from core import url_compiler as urlc
from core.project import Project
from ui.theme import inject_theme, page_hero

_MIRA_ROOT = Path(__file__).resolve().parents[1]

# Plotly dark template tuned to the studio palette.
_ACCENT = "#4c8dff"
_PLOT_TPL = "plotly_dark"


# --- theme --------------------------------------------------------------------
def _inject_css() -> None:
    # Palette, cards, pills, buttons, metrics all live in the shared theme now.
    inject_theme()


def _px(fig):
    fig.update_layout(template=_PLOT_TPL, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=42, b=8, l=8, r=8),
                      font=dict(color="#c9d1d9"))
    return fig


# --- chat panel ---------------------------------------------------------------
def _context_summary() -> str:
    bits = []
    proj = st.session_state.get("project")
    if proj is not None:
        try:
            bits.append(f"Active project: {proj.meta.get('name')} "
                        f"(task={proj.meta.get('task_type')}, "
                        f"target={proj.meta.get('target_column')}).")
        except Exception:
            pass
    disc = st.session_state.get("studio_discovery")
    if disc:
        bits.append(f"Last discovery: topic='{disc.get('topic')}', "
                    f"{len(disc.get('candidates', []))} candidate sources "
                    f"via {disc.get('provider')}.")
    probes = st.session_state.get("studio_probes")
    if probes:
        ok = sum(1 for p in probes if p.get("kind") not in ("error", "html"))
        bits.append(f"URL compiler: {len(probes)} probed, {ok} with tabular data.")
    return " ".join(bits)


def _open_claude_terminal(path: Path, prompt: str = "") -> tuple[bool, str]:
    """Open a full Claude Code session in a real terminal at `path` (the Mira project).

    The prompt is passed via a throwaway batch file, never on the command line: wt/cmd
    re-tokenize their tail and a literal newline or quote in the prompt otherwise corrupts
    parsing. Runs on the machine hosting this Streamlit app.
    """
    if sys.platform != "win32":
        return False, "Claude terminal launch is wired for Windows on this box."
    try:
        if prompt.strip():
            one_line = " ".join(prompt.split()).replace('"', "'")
            bat = Path(tempfile.gettempdir()) / f"mira_claude_{uuid.uuid4().hex[:8]}.cmd"
            bat.write_text(f'@echo off\r\ncd /d "{path}"\r\nclaude "{one_line}"\r\n',
                           encoding="utf-8")
            launch = ["cmd", "/k", str(bat)]
        else:
            launch = ["cmd", "/k", "claude"]
        wt = shutil.which("wt") or shutil.which("wt.exe")
        if wt:
            subprocess.Popen([wt, "-d", str(path), *launch])
        else:
            subprocess.Popen(launch, cwd=str(path),
                             creationflags=subprocess.CREATE_NEW_CONSOLE)
        return True, f"Opened a Claude terminal in {path}"
    except Exception as exc:  # never crash the page over a spawn failure
        return False, f"Couldn't open a terminal: {exc}"


def _render_terminal() -> None:
    """Right-hand panel: launch the full Claude Code agent in a terminal (not an API chat).

    Claude Code works directly in the Mira project — it can read/write files, run the
    pipeline, discover and vet sources — with the studio's current state seeded into the
    opening prompt. This replaces the constrained API/Ollama chat.
    """
    st.markdown("#### 🖥️ Claude Terminal")
    st.caption("Open the full Claude Code agent in a terminal, in the Mira project — not a "
               "constrained API chat. It can read/write files and run the pipeline; the "
               "studio's current state is seeded into the starting prompt below.")

    seed = (st.session_state.get("studio_input", "") or "").strip()
    ctx = _context_summary()
    parts = []
    if seed:
        parts.append(f"Help me with this scraping-studio input:\n{seed}")
    if ctx:
        parts.append(f"Studio context: {ctx}")
    default_prompt = "\n\n".join(parts)

    prompt = st.text_area("Starting prompt (optional — sent as the first message)",
                          value=default_prompt, height=200, key="studio_term_prompt")
    if st.button("🖥 Open Claude terminal", type="primary", use_container_width=True,
                 key="studio_term_open"):
        ok, msg = _open_claude_terminal(_MIRA_ROOT, prompt)
        (st.success if ok else st.error)(msg)
    st.caption(f"Opens on the machine running Mira, in `{_MIRA_ROOT}`.")


# --- discover -----------------------------------------------------------------
def _candidates_df(cands: list[dict]) -> pd.DataFrame:
    rows = []
    for c in cands:
        rows.append({
            "select": False,
            "title": (c.get("title") or c.get("url", ""))[:60],
            "type": c.get("scraper_type", "?"),
            "format": c.get("data_format") or "—",
            "coverage": round(float(c.get("field_coverage", 0.0)), 2),
            "tables": int(c.get("n_tables", 0) or 0),
            "score": round(float(c.get("score", 0) or 0), 2),
            "reachable": c.get("reachable", None),
            "url": c.get("url", ""),
        })
    return pd.DataFrame(rows)


def _render_discover(query: str) -> None:
    fields_raw = st.text_input(
        "Fields you want (comma-separated)", key="studio_fields",
        placeholder="e.g. team, season, wins, points",
    )
    c1, c2, c3 = st.columns(3)
    prefer = c1.selectbox("Source provider", ["auto", "serper", "duckduckgo"], key="studio_prov")
    max_c = c2.slider("Max candidates", 4, 24, 12, key="studio_maxc")
    do_probe = c3.toggle("Probe reachability", value=True, key="studio_probe")

    if st.button("🔎 Discover sources", type="primary", key="studio_disc_btn"):
        if not query.strip():
            st.warning("Type a topic in the smart input above first.")
        else:
            fields = [f.strip() for f in fields_raw.split(",") if f.strip()]
            from core import acquire_bridge as acbridge
            try:
                # Phase 1: fast search + score (no probing) so we know the candidate sites.
                with st.spinner("Searching the web for scrapeable sources…"):
                    res = acbridge.discover_sources(
                        query, fields,
                        prefer=None if prefer == "auto" else prefer,
                        max_candidates=max_c, do_probe=False,
                    )
                cands = res.get("candidates", [])
                # Phase 2: probe each site one at a time behind a LIVE status indicator —
                # showing which site is being checked and the shape of data found.
                if do_probe and cands:
                    n = len(cands)
                    with st.status(f"Probing {n} sources for data…", expanded=True) as status:
                        for i, c in enumerate(cands, 1):
                            label = (c.get("title") or c.get("url", "source"))[:70]
                            status.update(label=f"🔎 Probing {i}/{n}: {label}")
                            status.write(f"**{i}/{n}** · {label}")
                            acbridge.probe_candidate_live(c, fields)
                            if c.get("reachable"):
                                status.write(
                                    f"&nbsp;&nbsp;✓ {c.get('data_format', 'page') or 'page'}"
                                    f" · {int(c.get('n_tables', 0) or 0)} table(s)"
                                    f" · {c.get('field_coverage', 0)} field coverage"
                                    f" · score {round(float(c.get('score', 0) or 0), 2)}")
                            else:
                                status.write(f"&nbsp;&nbsp;✗ unreachable"
                                             f"{' — ' + c['probe_error'] if c.get('probe_error') else ''}")
                        ok = sum(1 for c in cands if c.get("reachable"))
                        tbl = sum(int(c.get("n_tables", 0) or 0) for c in cands)
                        status.update(
                            label=f"Probed {n} sources · {ok} reachable · {tbl} table(s) found",
                            state="complete", expanded=False)
                res["candidates"] = cands
                st.session_state["studio_discovery"] = res
            except Exception as exc:
                st.error(f"Discovery failed: {exc}")

    disc = st.session_state.get("studio_discovery")
    if not disc:
        return
    cands = disc.get("candidates", [])
    st.markdown(f"<span class='pill'>provider: {disc.get('provider')}</span>"
                f"<span class='pill'>topic: {disc.get('topic')}</span>"
                f"<span class='pill'>{len(cands)} candidates</span>",
                unsafe_allow_html=True)
    if not cands:
        st.info("No scrapeable candidates found. Try broader terms or fewer fields.")
        return

    df = _candidates_df(cands)

    # --- visualizations ---
    v1, v2 = st.columns(2)
    with v1:
        top = df.sort_values("score", ascending=False).head(12)
        fig = px.bar(top, x="score", y="title", orientation="h", color="score",
                     color_continuous_scale="Blues", title="Relevance score")
        fig.update_layout(height=320, yaxis={"autorange": "reversed"},
                          coloraxis_showscale=False)
        st.plotly_chart(_px(fig), use_container_width=True, key="disc_score")
    with v2:
        if df["coverage"].any():
            top = df.sort_values("coverage", ascending=False).head(12)
            fig = px.bar(top, x="coverage", y="title", orientation="h",
                         color="coverage", color_continuous_scale="Tealgrn",
                         title="Requested-field coverage")
            fig.update_layout(height=320, yaxis={"autorange": "reversed"},
                              coloraxis_showscale=False)
            st.plotly_chart(_px(fig), use_container_width=True, key="disc_cov")
        else:
            st.caption("No field-coverage signal (add target fields above).")

    v3, v4 = st.columns(2)
    with v3:
        tc = df["type"].value_counts().reset_index()
        tc.columns = ["type", "count"]
        fig = px.pie(tc, names="type", values="count", hole=0.5,
                     title="Scraper type mix")
        fig.update_layout(height=280)
        st.plotly_chart(_px(fig), use_container_width=True, key="disc_type")
    with v4:
        reach = df["reachable"].map({True: "reachable", False: "unreachable"}).fillna("unknown")
        rc = reach.value_counts().reset_index()
        rc.columns = ["state", "count"]
        fig = px.pie(rc, names="state", values="count", hole=0.5,
                     title="Reachability",
                     color="state",
                     color_discrete_map={"reachable": "#2ea043", "unreachable": "#f85149",
                                         "unknown": "#8b949e"})
        fig.update_layout(height=280)
        st.plotly_chart(_px(fig), use_container_width=True, key="disc_reach")

    # --- pick & build ---
    st.markdown("**Select sources to scrape into a project**")
    edited = st.data_editor(
        df, hide_index=True, use_container_width=True, height=260,
        column_config={
            "select": st.column_config.CheckboxColumn("✓"),
            "coverage": st.column_config.ProgressColumn("coverage", min_value=0, max_value=1),
            "url": st.column_config.LinkColumn("url"),
        },
        key="studio_disc_editor",
    )
    picks = [cands[i] for i in edited.index[edited["select"]].tolist()]
    cta = st.columns([2, 1])
    cta[0].caption(f"{len(picks)} selected")
    if cta[1].button("🧪 Build project", type="primary", disabled=not picks,
                     key="studio_disc_build", use_container_width=True):
        with st.spinner(f"Scraping {len(picks)} source(s)…"):
            try:
                from core.acquire_bridge import scrape_candidates_into_project
                out = scrape_candidates_into_project(
                    picks, topic=disc.get("topic", "discovered"),
                    fields=disc.get("fields", []),
                )
                _after_build(out.get("project_id"), out.get("rows", 0),
                             out.get("target_column"), out.get("sources"))
            except Exception as exc:
                st.error(f"Scrape failed: {exc}")


# --- url compiler -------------------------------------------------------------
def _render_url_compiler(blob: str) -> None:
    urls = urlc.extract_urls(blob)
    if not urls:
        st.info("Paste one or more URLs (or a page of text containing links) into the "
                "smart input above, then probe them here.")
        return
    st.markdown(f"<span class='pill'>{len(urls)} URL(s) detected</span>",
                unsafe_allow_html=True)

    if st.button("📡 Probe URLs", type="primary", key="studio_probe_btn"):
        probes = []
        prog = st.progress(0.0, text="Probing…")
        for i, u in enumerate(urls, 1):
            probes.append(urlc.preview_url(u).__dict__)
            prog.progress(i / len(urls), text=f"Probed {i}/{len(urls)}")
        prog.empty()
        st.session_state["studio_probes"] = probes

    probes = st.session_state.get("studio_probes")
    if not probes:
        return
    pdf = pd.DataFrame(probes)

    m = st.columns(4)
    m[0].metric("URLs", len(pdf))
    m[1].metric("Reachable", int(pdf["ok"].sum()))
    m[2].metric("With tables", int((pdf["kind"] == "html_table").sum()))
    m[3].metric("Total rows", int(pdf["rows"].sum()))

    v1, v2 = st.columns(2)
    with v1:
        kc = pdf["kind"].value_counts().reset_index()
        kc.columns = ["kind", "count"]
        fig = px.pie(kc, names="kind", values="count", hole=0.5, title="Content kind")
        fig.update_layout(height=280)
        st.plotly_chart(_px(fig), use_container_width=True, key="url_kind")
    with v2:
        withrows = pdf[pdf["rows"] > 0]
        if not withrows.empty:
            withrows = withrows.assign(short=withrows["url"].str.slice(0, 42))
            fig = px.bar(withrows.sort_values("rows", ascending=False).head(12),
                         x="rows", y="short", orientation="h", color="rows",
                         color_continuous_scale="Blues", title="Rows available per URL")
            fig.update_layout(height=280, yaxis={"autorange": "reversed"},
                              coloraxis_showscale=False)
            st.plotly_chart(_px(fig), use_container_width=True, key="url_rows")
        else:
            st.caption("No tabular data found at these URLs.")

    show = pdf[["url", "ok", "status", "kind", "n_tables", "rows", "cols", "note", "elapsed_ms"]]
    st.dataframe(show, hide_index=True, use_container_width=True, height=240,
                 column_config={"url": st.column_config.LinkColumn("url")})

    good = [r["url"] for r in probes if r.get("rows", 0) > 0]
    name = st.text_input("Project name", value="URL compile", key="studio_url_name")
    task = st.selectbox("Task", ["regression", "classification"], key="studio_url_task")
    cta = st.columns([2, 1])
    cta[0].caption(f"{len(good)} URL(s) with data will be compiled")
    if cta[1].button("🧪 Compile project", type="primary", disabled=not good,
                     key="studio_url_build", use_container_width=True):
        with st.spinner(f"Loading & combining {len(good)} URL(s)…"):
            res = urlc.build_project_from_urls(good, project_name=name, task_type=task)
            if res.error:
                st.error(res.error)
            else:
                _after_build(res.project_id, res.rows, res.target_column, res.per_url)


# --- activity & records -------------------------------------------------------
def _render_activity() -> None:
    try:
        from core.acquire_bridge import load_scraper_run_log
        log = load_scraper_run_log(limit=200)
    except Exception as exc:
        st.info(f"No scraper activity available ({exc}).")
        return
    if not log:
        st.info("No scraper runs logged yet. Run a discovery scrape to populate this.")
        return
    df = pd.DataFrame(log)
    st.caption(f"{len(df)} recent run record(s)")
    cols = st.columns(2)
    status_col = next((c for c in ("status", "result", "outcome") if c in df.columns), None)
    if status_col:
        sc = df[status_col].astype(str).value_counts().reset_index()
        sc.columns = ["status", "count"]
        with cols[0]:
            fig = px.pie(sc, names="status", values="count", hole=0.5, title="Run outcomes")
            fig.update_layout(height=300)
            st.plotly_chart(_px(fig), use_container_width=True, key="act_status")
    src_col = next((c for c in ("source_id", "source", "url") if c in df.columns), None)
    if src_col:
        bc = df[src_col].astype(str).value_counts().head(12).reset_index()
        bc.columns = [src_col, "runs"]
        with cols[1]:
            fig = px.bar(bc, x="runs", y=src_col, orientation="h", title="Runs by source")
            fig.update_layout(height=300, yaxis={"autorange": "reversed"})
            st.plotly_chart(_px(fig), use_container_width=True, key="act_src")
    with st.expander("Raw run log"):
        st.dataframe(df, hide_index=True, use_container_width=True, height=300)


def _render_records() -> None:
    try:
        from core.acquire_bridge import records_by_source_df, records_stats_df
        stats = records_stats_df()
        by_src = records_by_source_df()
    except Exception as exc:
        st.info(f"Records warehouse unavailable ({exc}).")
        return
    if not stats.empty:
        r = stats.iloc[0]
        m = st.columns(4)
        m[0].metric("Records", f"{int(r.get('total_records', 0)):,}")
        m[1].metric("Sources", int(r.get("sources", 0)))
        m[2].metric("Regions", int(r.get("regions", 0)))
        m[3].metric("Latest period", str(r.get("latest_period", "—")))
    if by_src is not None and not by_src.empty:
        top = by_src.head(15)
        fig = px.bar(top, x="cnt", y="source_name", orientation="h", color="cnt",
                     color_continuous_scale="Blues", title="Records by source")
        fig.update_layout(height=380, yaxis={"autorange": "reversed"},
                          coloraxis_showscale=False)
        st.plotly_chart(_px(fig), use_container_width=True, key="rec_src")
        with st.expander("Source table"):
            st.dataframe(by_src, hide_index=True, use_container_width=True)
    else:
        st.info("No records ingested yet.")


def _after_build(pid: str | None, rows: int, target: str | None,
                 sources: Any) -> None:
    if not pid:
        st.error("No data could be built into a project.")
        if sources:
            st.dataframe(pd.DataFrame(sources), hide_index=True, use_container_width=True)
        return
    st.success(f"✅ Built project `{pid}` — {rows:,} rows"
               + (f", suggested target **{target}**" if target else ""))
    if sources:
        with st.expander("Per-source results"):
            st.dataframe(pd.DataFrame(sources), hide_index=True, use_container_width=True)
    if st.button("➡ Open in pipeline (set active project)", key=f"open_{pid}"):
        try:
            st.session_state.project = Project.load(pid)
            st.toast(f"Active project set to {pid}")
            st.rerun()
        except Exception as exc:
            st.error(f"Couldn't load project: {exc}")


# --- main ---------------------------------------------------------------------
def render_scraping_studio() -> None:
    _inject_css()
    page_hero(
        "🛰️ Scraping Studio",
        "Find sources, compile URLs, and shape raw data for transformation — "
        "with an AI copilot in the room.",
    )

    main, chat_col = st.columns([2, 1], gap="large")

    with main:
        blob = st.text_area(
            "Smart input — a topic, a list of URLs, or pasted data",
            key="studio_input", height=110,
            placeholder="e.g.  'NFL team season stats 2023'   ·   "
                        "https://example.com/table.html   ·   col_a,col_b\\n1,2",
        )
        kind = urlc.classify_input(blob) if blob.strip() else None
        if kind:
            label = {"url": "🔗 looks like URL(s)", "data": "📋 looks like raw data",
                     "topic": "🔎 looks like a search topic"}[kind]
            st.markdown(f"<span class='pill'>{label}</span>", unsafe_allow_html=True)
            if kind == "data":
                if st.button("📋 Load pasted data into a project", key="studio_paste_build"):
                    df = urlc.parse_pasted_data(blob)
                    if df is None or df.empty:
                        st.error("Couldn't parse that as CSV/TSV/JSON.")
                    else:
                        from core.ingest import ingest_dataframe
                        proj = Project.create(name="Pasted data")
                        ingest_dataframe(proj, df, source_name="pasted", auto_profile=False)
                        _after_build(proj.id, len(df), None, None)

        default_mode = 1 if kind == "url" else 0
        # A horizontal radio, not a segmented control: that widget needs
        # Streamlit >= 1.40 and pyproject pins 1.39.*.
        mode = st.radio(
            "Tool",
            ["🔎 Discover", "🔗 URL Compiler", "📊 Activity", "🗄 Records"],
            index=default_mode,
            horizontal=True,
            key="studio_mode",
        )

        st.divider()
        if mode == "🔎 Discover":
            _render_discover(blob)
        elif mode == "🔗 URL Compiler":
            _render_url_compiler(blob)
        elif mode == "📊 Activity":
            _render_activity()
        else:
            _render_records()

    with chat_col:
        _render_terminal()
