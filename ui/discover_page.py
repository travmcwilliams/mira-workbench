"""MIRA Discover page — topic + fields -> find scrapeable sources -> project."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from core.acquire_bridge import discover_sources, scrape_candidates_into_project
from core.project import Project
from ui.theme import page_hero


def _parse_fields(raw: str) -> list[str]:
    parts: list[str] = []
    for chunk in (raw or "").replace("\n", ",").split(","):
        c = chunk.strip()
        if c:
            parts.append(c)
    return parts


def _candidates_table(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        matched = c.get("matched_fields") or []
        rows.append({
            "scrape": False,
            "title": (c.get("title") or c.get("url", ""))[:70],
            "type": c.get("scraper_type", "?"),
            "format": c.get("data_format") or "—",
            "fields": f"{len(matched)}/{len(st.session_state.get('disc_fields', []))}"
                      if st.session_state.get("disc_fields") else str(len(matched)),
            "coverage": c.get("field_coverage", 0.0),
            "tables": c.get("n_tables", 0),
            "score": c.get("score", 0),
            "reachable": c.get("reachable", None),
            "url": c.get("url", ""),
        })
    return pd.DataFrame(rows)


def render_discover_page() -> None:
    page_hero(
        "Discover Sources",
        "Type a topic and the fields you want — MIRA finds scrapeable places, "
        "you pick which to pull, then it builds a project and hands you to the pipeline.",
    )

    serper_on = bool(os.environ.get("SERPER_API_KEY"))
    c1, c2 = st.columns([3, 1])
    with c2:
        st.caption("Search backend")
        st.write("Serper (Google)" if serper_on else "Keyless (Mojeek → DDG)")
        force_keyless = st.checkbox("Force keyless search", value=not serper_on,
                                   help="Use keyless engines even if a Serper key is set.")
        if not serper_on:
            st.caption("Set `SERPER_API_KEY` for more robust, Google-backed results.")

    with c1:
        topic = st.text_input("Topic / prompt",
                              placeholder="e.g. US county population by year")
        fields_raw = st.text_area(
            "Data fields you want (comma or newline separated)",
            placeholder="county, state, population, year",
            height=70,
        )
        task_type = st.selectbox("Task type for the project", ["regression", "classification"])

    if st.button("🔎 Search for sources", type="primary", disabled=not topic.strip()):
        fields = _parse_fields(fields_raw)
        st.session_state.disc_fields = fields
        st.session_state.disc_topic = topic.strip()
        st.session_state.disc_task = task_type
        with st.spinner("Searching and probing candidate sources…"):
            try:
                result = discover_sources(
                    topic.strip(), fields,
                    prefer="keyless" if force_keyless else None,
                )
                st.session_state.disc_candidates = result["candidates"]
                st.session_state.disc_provider = result["provider"]
            except Exception as exc:
                st.error(f"Discovery failed: {exc}")
                st.session_state.disc_candidates = []

    candidates = st.session_state.get("disc_candidates")
    if candidates is None:
        return

    st.divider()
    if not candidates:
        st.warning("No candidates found. Try a broader topic or fewer fields.")
        return

    st.subheader(f"{len(candidates)} candidate sources")
    st.caption(f"via {st.session_state.get('disc_provider', '?')} • "
               "check the rows you want, then scrape. Sorted by field coverage.")

    table = _candidates_table(candidates)
    edited = st.data_editor(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "scrape": st.column_config.CheckboxColumn("Scrape", width="small"),
            "coverage": st.column_config.ProgressColumn(
                "Field match", min_value=0.0, max_value=1.0, format="%.0f%%"),
            "url": st.column_config.LinkColumn("URL"),
        },
        disabled=[c for c in table.columns if c != "scrape"],
        key="disc_editor",
    )

    selected_idx = [i for i, v in enumerate(edited["scrape"].tolist()) if v]
    st.caption(f"{len(selected_idx)} selected")

    cols = st.columns([1, 3])
    with cols[0]:
        go = st.button("⬇️ Scrape selected → project", type="primary",
                       disabled=not selected_idx)
    if go:
        chosen = [candidates[i] for i in selected_idx]
        with st.spinner(f"Scraping {len(chosen)} source(s) and building project…"):
            try:
                summary = scrape_candidates_into_project(
                    chosen,
                    topic=st.session_state.get("disc_topic", "discovered"),
                    fields=st.session_state.get("disc_fields", []),
                    task_type=st.session_state.get("disc_task", "regression"),
                )
            except Exception as exc:
                st.error(f"Scrape failed: {exc}")
                return

        if not summary.get("project_id"):
            st.error(summary.get("error", "No data scraped."))
            with st.expander("Per-source detail"):
                st.dataframe(pd.DataFrame(summary.get("sources", [])),
                             use_container_width=True, hide_index=True)
            return

        st.success(
            f"Built project with {summary['rows']:,} rows × "
            f"{len(summary['columns'])} columns. "
            f"Suggested target: **{summary.get('target_column') or '—'}** "
            f"({summary.get('target_reason', 'n/a')})."
        )
        with st.expander("Per-source detail"):
            st.dataframe(pd.DataFrame(summary.get("sources", [])),
                         use_container_width=True, hide_index=True)

        st.session_state.project = Project.load(summary["project_id"])
        st.info("Project loaded. Opening the pipeline…")
        try:
            st.switch_page("pages/home.py")
        except Exception:
            st.page_link("pages/home.py", label="➡️ Go to the pipeline (Home)")
