"""MIRA Sources & Records page — merged DSI registry, scrape, and activity controls."""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from core.acquire_bridge import (
    ALLOWED_MAPPING_KEYS,
    discover_sources,
    ensure_acquire_env,
    get_acquire_status,
    get_fetcher_status,
    ingest_source_id,
    load_records_df,
    load_registry,
    load_registry_df,
    load_review_queue,
    load_scraper_run_log,
    records_by_source_df,
    records_stats_df,
    run_records_query,
    run_registry_repair,
    save_registry,
    test_fetch_url,
    validate_registry,
    clear_review_queue,
)
from core.activity_log import unified_activity_df
from ui.theme import page_hero


def _acquire_import(module: str, name: str):
    ensure_acquire_env()
    return getattr(__import__(module, fromlist=[name]), name)


# The four scraper types the registry schema accepts (see acquire/scripts/registry_repair).
_SCRAPER_TYPES = ["has_download_link", "has_record_link", "has_search_form", "page_only"]


def _next_source_id(data: dict) -> str:
    """Suggest the next free zero-padded numeric source_id (e.g. 00004 -> 00005)."""
    nums = [
        int(s["source_id"])
        for s in data.get("sources", [])
        if str(s.get("source_id", "")).isdigit()
    ]
    return f"{(max(nums) + 1) if nums else 1:05d}"


def _candidate_to_entry(cand: dict, source_id: str, region: str, region_full: str) -> dict:
    """Map a discovered web candidate onto a valid 8-key registry entry."""
    scraper_type = cand.get("scraper_type")
    if scraper_type not in _SCRAPER_TYPES:
        scraper_type = "page_only"
    return {
        "source_id": source_id,
        "source_name": (cand.get("title") or cand.get("url") or source_id)[:120],
        "region": region,
        "region_full": region_full,
        "source_url": cand.get("url") or None,
        "scraper_type": scraper_type,
        "data_method": "scrape",
        "last_scraped": None,
    }


def _find_candidates_df(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        rows.append({
            "title": (c.get("title") or c.get("url", ""))[:70],
            "type": c.get("scraper_type", "?"),
            "format": c.get("data_format") or "—",
            "coverage": float(c.get("field_coverage", 0.0) or 0.0),
            "reachable": c.get("reachable", None),
            "url": c.get("url", ""),
        })
    return pd.DataFrame(rows)


def _render_find_online() -> None:
    """Search the web for a scrapeable source and pull it straight into the registry.

    Reuses the same discovery backend as the Discover page (Serper/Google, or the keyless
    Mojeek -> DuckDuckGo fallback), but instead of building a whole project it lets the user
    add a single discovered source to the registry (one click) or prefill the form below.
    """
    serper_on = bool(os.environ.get("SERPER_API_KEY"))
    with st.expander("🔎 Find a source online", expanded=False):
        st.caption(
            "Describe the data you want; MIRA searches the web and probes candidate pages. "
            f"Backend: **{'Serper (Google)' if serper_on else 'keyless (Mojeek → DuckDuckGo)'}**."
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            topic = st.text_input(
                "What are you looking for?",
                key="reg_find_topic",
                placeholder="e.g. city of Austin grants open data portal",
            )
            fields_raw = st.text_input(
                "Fields you expect (optional, comma-separated)",
                key="reg_find_fields",
                placeholder="grant_name, amount, agency, year",
            )
        with c2:
            st.text_input("Default region", value="US", key="reg_find_region")
            st.text_input(
                "Default region (full)", value="United States", key="reg_find_region_full"
            )
            force_keyless = st.checkbox(
                "Force keyless", value=not serper_on, key="reg_find_keyless"
            )
            skip_probe = st.checkbox(
                "Skip page probing (faster)", value=False, key="reg_find_skipprobe",
                help="Probing fetches each page to detect the scraper type and field "
                     "coverage — accurate but slower. Skip it for a quick list of links.",
            )

        if st.button("🔎 Search the web", key="reg_find_go", disabled=not topic.strip()):
            fields = [f.strip() for f in fields_raw.replace("\n", ",").split(",") if f.strip()]
            prefer = "keyless" if force_keyless else None
            with st.spinner("Searching the web" + ("…" if skip_probe else " and probing pages…")):
                try:
                    result = discover_sources(
                        topic.strip(), fields, prefer=prefer, do_probe=not skip_probe,
                    )
                    cands = result["candidates"]
                    relaxed = False
                    # Narrow field lists can filter every candidate out — if so, retry
                    # once without the field constraint so the user still gets results.
                    if not cands and fields:
                        result = discover_sources(
                            topic.strip(), [], prefer=prefer, do_probe=not skip_probe,
                        )
                        cands = result["candidates"]
                        relaxed = bool(cands)
                    st.session_state.reg_find_candidates = cands
                    st.session_state.reg_find_provider = result.get("provider", "?")
                    st.session_state.reg_find_relaxed = relaxed
                except Exception as exc:
                    st.error(f"Search failed: {exc}")
                    # None (not []) so the "no sources found" hint below stays hidden:
                    # the search never ran, so "try a broader query" would mislead.
                    st.session_state.reg_find_candidates = None

        candidates = st.session_state.get("reg_find_candidates")
        if candidates is None:
            return
        if not candidates:
            st.warning("No sources found. Try a broader query or drop some fields.")
            return

        st.caption(
            f"{len(candidates)} candidate(s) via {st.session_state.get('reg_find_provider', '?')} "
            "— sorted by field match."
        )
        if st.session_state.get("reg_find_relaxed"):
            st.caption("ℹ️ No pages matched all your fields, so the field filter was relaxed.")
        st.dataframe(
            _find_candidates_df(candidates),
            hide_index=True,
            use_container_width=True,
            column_config={
                "coverage": st.column_config.ProgressColumn(
                    "Field match", min_value=0.0, max_value=1.0, format="%.0f%%"),
                "reachable": st.column_config.CheckboxColumn("Reachable", width="small"),
                "url": st.column_config.LinkColumn("URL", display_text="open ↗"),
            },
        )

        pick = st.selectbox(
            "Pick a source",
            options=list(range(len(candidates))),
            format_func=lambda i: (candidates[i].get("title") or candidates[i].get("url", ""))[:80],
            key="reg_find_pick",
        )
        chosen = candidates[pick]
        b1, b2 = st.columns(2)
        with b1:
            if st.button("⬇️ Add to registry", type="primary", key="reg_find_add"):
                data = load_registry()
                sid = _next_source_id(data)
                entry = _candidate_to_entry(
                    chosen, sid,
                    st.session_state.get("reg_find_region", "US") or "US",
                    st.session_state.get("reg_find_region_full", "United States") or "Web",
                )
                data["sources"].append(entry)
                save_registry(data)
                errs = [e for e in validate_registry() if e.startswith(f"[{len(data['sources']) - 1}]")]
                if errs:
                    st.warning("Added, but with validation issues:\n" + "\n".join(errs[:5]))
                else:
                    st.success(f"Added **{sid}** — {entry['source_name']}")
                st.toast("Source added. Run **Registry Repair** to finalize.", icon="🛠️")
                st.rerun()
        with b2:
            if st.button("✎ Prefill form below", key="reg_find_prefill"):
                data = load_registry()
                st.session_state.reg_prefill = {
                    "source_id": _next_source_id(data),
                    "source_name": (chosen.get("title") or "")[:120],
                    "source_url": chosen.get("url") or "",
                    "scraper_type": chosen.get("scraper_type")
                    if chosen.get("scraper_type") in _SCRAPER_TYPES else "page_only",
                    "region": st.session_state.get("reg_find_region", "US") or "US",
                    "region_full": st.session_state.get("reg_find_region_full", "United States") or "",
                }
                st.toast("Form prefilled below — review and Add/Update.", icon="✎")
                st.rerun()


def render_registry_tab() -> None:
    st.subheader("Source Registry")
    st.caption("Edit sources in `acquire/data/source_registry.json`. Run repair after changes.")

    sources_df = load_registry_df()
    if not sources_df.empty:
        # Map validation errors (prefixed "[<row idx>]") back to their rows so
        # each source shows an at-a-glance ✓ / ⚠ status instead of relying on the
        # table's native search highlight.
        bad_rows: set[int] = set()
        for err in validate_registry():
            if err.startswith("[") and "]" in err:
                try:
                    bad_rows.add(int(err[1 : err.index("]")]))
                except ValueError:
                    pass

        display_df = sources_df.copy()
        display_df.insert(
            0, "status", ["⚠ needs repair" if i in bad_rows else "✓ valid" for i in range(len(display_df))]
        )
        # Drop dict/list columns (e.g. column_mapping) — shown separately below.
        for col in list(display_df.columns):
            if display_df[col].apply(lambda v: isinstance(v, (dict, list))).any():
                display_df = display_df.drop(columns=[col])

        col_cfg = {
            "status": st.column_config.TextColumn("Status", width="small"),
            "source_id": st.column_config.TextColumn("Source ID", width="small"),
            "source_name": st.column_config.TextColumn("Name"),
            "region": st.column_config.TextColumn("Region", width="small"),
            "region_full": st.column_config.TextColumn("Region (full)"),
            "source_url": st.column_config.LinkColumn("URL", display_text="open ↗"),
            "scraper_type": st.column_config.TextColumn("Scraper type"),
            "data_method": st.column_config.TextColumn("Method", width="small"),
            "last_scraped": st.column_config.TextColumn("Last scraped"),
        }
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={k: v for k, v in col_cfg.items() if k in display_df.columns},
        )
        if bad_rows:
            st.caption(
                f"⚠ {len(bad_rows)} source(s) have validation issues — "
                "fix the fields below and click **Run Full Registry Repair**."
            )
        else:
            st.caption("✓ All sources pass validation.")

        has_mappings = False
        for _, row in sources_df.iterrows():
            mapping = row.get("column_mapping")
            if mapping and isinstance(mapping, dict) and mapping:
                if not has_mappings:
                    st.write("**Column mappings:**")
                    has_mappings = True
                st.code(f"{row['source_id']}: {mapping}", language="json")
    else:
        st.info("No sources yet. Add one below.")

    st.divider()
    _render_find_online()

    st.write("**Add / Update Source**")
    pf = st.session_state.get("reg_prefill", {})
    if pf:
        st.caption("✎ Prefilled from a web source — review the region fields, then Add/Update.")

    with st.form("mira_add_source", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            source_id = st.text_input("source_id", value=pf.get("source_id", ""), placeholder="e.g. 00004")
            source_name = st.text_input("source_name", value=pf.get("source_name", ""), placeholder="e.g. City Grants Portal")
            region = st.text_input("region (2-char)", value=pf.get("region", ""), placeholder="e.g. US")
        with c2:
            region_full = st.text_input("region_full", value=pf.get("region_full", ""), placeholder="e.g. United States")
            source_url = st.text_input("source_url (optional)", value=pf.get("source_url", ""), placeholder="https://…")
            _st_default = pf.get("scraper_type", "has_download_link")
            scraper_type = st.selectbox(
                "scraper_type",
                _SCRAPER_TYPES,
                index=_SCRAPER_TYPES.index(_st_default) if _st_default in _SCRAPER_TYPES else 0,
            )
            data_method = st.selectbox("data_method", ["scrape", "api", "manual"])
            mapping_json = st.text_area(
                "Column mapping (optional JSON)",
                value="",
                height=80,
                help=f"Keys must be one of: {sorted(ALLOWED_MAPPING_KEYS)}",
            )

        if st.form_submit_button("Add/Update + Validate"):
            try:
                data = load_registry()
                entry = {
                    "source_id": source_id,
                    "source_name": source_name,
                    "region": region,
                    "region_full": region_full,
                    "source_url": source_url or None,
                    "scraper_type": scraper_type,
                    "data_method": data_method,
                    "last_scraped": None,
                }
                if mapping_json.strip():
                    parsed = json.loads(mapping_json)
                    if not isinstance(parsed, dict):
                        raise ValueError("Mapping must be a JSON object")
                    bad = [k for k in parsed if k not in ALLOWED_MAPPING_KEYS]
                    if bad:
                        st.error(f"Invalid mapping keys: {bad}")
                        st.stop()
                    entry["column_mapping"] = parsed

                existing = next(
                    (s for s in data["sources"] if s["source_id"] == source_id),
                    None,
                )
                if existing:
                    existing.update(entry)
                    st.success(f"Updated {source_id}")
                else:
                    data["sources"].append(entry)
                    st.success(f"Added {source_id}")

                save_registry(data)
                errs = validate_registry()
                if errs:
                    st.warning("Validation issues:\n" + "\n".join(errs[:10]))
                else:
                    st.info("Registry validation passed.")
                st.toast("Saved. Run **Registry Repair** to finalize changes.", icon="🛠️")
                st.session_state.pop("reg_prefill", None)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    if st.button("Run Full Registry Repair"):
        ok, out = run_registry_repair(verify_only=True)
        if ok:
            st.success("Registry repair passed.")
        else:
            st.error(out or "Repair failed.")
        if out:
            st.code(out)
        st.rerun()


def _clear_term_edit_state() -> None:
    """Drop per-row term-edit widget keys so they re-init from ``patch_terms``.

    The inline term editors use index-based keys (``mira_term_edit_<i>``); after
    a removal or a patch switch the indices shift, so stale widget state must be
    cleared or a removed term's value would linger in the next row.
    """
    for k in [k for k in st.session_state if k.startswith("mira_term_edit_")]:
        del st.session_state[k]


def _render_patch_scrape() -> None:
    load_patches = _acquire_import("agents.patch_runner", "load_patches")
    get_patch = _acquire_import("agents.patch_runner", "get_patch")
    expand_patch = _acquire_import("agents.patch_runner", "expand_patch")
    build_temp_source = _acquire_import("agents.patch_runner", "build_temp_source")
    ingest_source = _acquire_import("agents.source_ingestor", "ingest_source")
    get_record_count = _acquire_import("utils.db", "get_record_count")

    st.markdown("### Custom Scrape Patches")
    all_patches = load_patches()
    if not all_patches:
        st.warning("No patches in `acquire/data/scrape_patches.json`.")
        return

    patch_labels = {p["patch_id"]: p["name"] for p in all_patches}
    chosen = st.selectbox(
        "Choose patch",
        [p["patch_id"] for p in all_patches],
        format_func=lambda pid: patch_labels.get(pid, pid),
        key="mira_patch_pick",
    )
    patch = get_patch(chosen) or all_patches[0]

    if "patch_terms" not in st.session_state or st.session_state.get("patch_terms_for") != chosen:
        st.session_state.patch_terms = list(patch.get("term_examples", []))[:4]
        st.session_state.patch_terms_for = chosen
        _clear_term_edit_state()

    new_term = st.text_input("Add search term", key="mira_patch_term")
    if st.button("Add term", key="mira_add_term") and new_term.strip():
        if new_term.strip() not in st.session_state.patch_terms:
            st.session_state.patch_terms.append(new_term.strip())
            st.rerun()

    # Editable/removable active terms. Each row is an inline text field plus a
    # remove button. Widget keys are indexed, so on any length change we wipe the
    # per-row keys and let them re-init from the list (avoids stale values after
    # a removal shifts indices).
    stored_terms = list(st.session_state.get("patch_terms", []))
    if stored_terms:
        st.caption("Active terms — edit inline, or ✕ to remove")
        for i, term in enumerate(stored_terms):
            c_edit, c_del = st.columns([8, 1])
            edited = c_edit.text_input(
                f"term {i + 1}",
                value=term,
                key=f"mira_term_edit_{i}",
                label_visibility="collapsed",
            )
            st.session_state.patch_terms[i] = edited.strip()
            if c_del.button("✕", key=f"mira_term_del_{i}", help="Remove this term"):
                st.session_state.patch_terms.pop(i)
                _clear_term_edit_state()
                st.rerun()
        active_terms = [t for t in st.session_state.get("patch_terms", []) if t]
    else:
        active_terms = []
        st.caption("Active terms: (none)")

    fmts = patch.get("supported_formats", ["csv", "xlsx"])
    selected_fmts = st.multiselect("Formats", fmts, default=fmts[: min(2, len(fmts))], key="mira_fmts")
    patch_dry = st.checkbox("Dry run", value=False, key="mira_patch_dry")
    patch_region = st.text_input("Region code", value="00", key="mira_patch_region")

    if st.button("Run patch scrape", type="primary", key="mira_patch_run"):
        if not active_terms or not selected_fmts:
            st.error("Select at least one term and format.")
        else:
            tasks = expand_patch(patch, active_terms, selected_fmts)
            results = []
            prog = st.progress(0.0)
            for i, task in enumerate(tasks):
                temp_src = build_temp_source(task, region=patch_region or "00")
                try:
                    before = get_record_count()
                    rows = ingest_source(temp_src, local_file=None, dry_run=patch_dry)
                    after = get_record_count()
                    results.append({
                        "task_id": task["task_id"],
                        "term": task["term"],
                        "format": task["fmt"],
                        "status": "ok",
                        "rows_loaded": rows,
                        "db_delta": after - before,
                    })
                except Exception as ex:
                    results.append({
                        "task_id": task["task_id"],
                        "term": task["term"],
                        "status": "error",
                        "error": str(ex)[:120],
                    })
                prog.progress((i + 1) / max(len(tasks), 1))
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


def _render_bulk_scrape() -> None:
    ingest_source = _acquire_import("agents.source_ingestor", "ingest_source")
    ingest_by_id = _acquire_import("agents.source_ingestor", "ingest_by_id")
    get_record_count = _acquire_import("utils.db", "get_record_count")

    st.markdown("### Bulk URL Scrape")
    paste = st.text_area("URLs (one per line)", height=100, key="mira_bulk_urls")
    bulk_urls = [ln.strip() for ln in paste.splitlines() if ln.strip()]
    if not bulk_urls:
        return

    st.caption(f"{len(bulk_urls)} URLs")
    prefix = st.text_input("Batch prefix", value="Bulk", key="mira_bulk_prefix")
    scraper = st.selectbox(
        "scraper_type",
        ["has_download_link", "has_record_link", "has_search_form", "page_only"],
        key="mira_bulk_scraper",
    )
    region = st.text_input("Region", value="00", key="mira_bulk_region")
    dry = st.checkbox("Dry run", key="mira_bulk_dry")
    mapping_json = st.text_area(
        "Column mapping JSON (optional, all URLs)",
        value="",
        height=60,
        key="mira_bulk_mapping",
    )
    bulk_mapping = None
    if mapping_json.strip():
        try:
            bulk_mapping = json.loads(mapping_json)
        except Exception as exc:
            st.error(f"Invalid mapping JSON: {exc}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Quick scrape (temp)", key="mira_bulk_go"):
            results = []
            for idx, url in enumerate(bulk_urls):
                sid = f"{prefix}-{idx:04d}"
                temp = {
                    "source_id": sid,
                    "source_name": f"{prefix} {idx + 1}",
                    "region": region or "00",
                    "region_full": "Bulk Import",
                    "source_url": url,
                    "scraper_type": scraper,
                    "data_method": "scrape",
                    "last_scraped": None,
                }
                if bulk_mapping:
                    temp["column_mapping"] = bulk_mapping
                try:
                    before = get_record_count()
                    rows = ingest_source(temp, local_file=None, dry_run=dry, mapping=bulk_mapping)
                    after = get_record_count()
                    results.append({"source_id": sid, "status": "ok", "rows": rows, "delta": after - before})
                except Exception as ex:
                    results.append({"source_id": sid, "status": "error", "error": str(ex)[:100]})
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    with c2:
        if st.button("Add to registry + scrape", key="mira_bulk_reg"):
            data = load_registry()
            added = 0
            for idx, url in enumerate(bulk_urls):
                sid = f"{prefix}-{idx:04d}"
                entry = {
                    "source_id": sid,
                    "source_name": f"{prefix} {idx + 1}",
                    "region": region or "00",
                    "region_full": "Bulk Import",
                    "source_url": url,
                    "scraper_type": scraper,
                    "data_method": "scrape",
                    "last_scraped": None,
                }
                if bulk_mapping:
                    entry["column_mapping"] = bulk_mapping
                existing = next((s for s in data["sources"] if s["source_id"] == sid), None)
                if existing:
                    existing.update(entry)
                else:
                    data["sources"].append(entry)
                    added += 1
            save_registry(data)
            st.info(f"Registry updated ({added} new). Scraping…")

            results = []
            for idx, url in enumerate(bulk_urls):
                sid = f"{prefix}-{idx:04d}"
                try:
                    before = get_record_count()
                    rows = ingest_by_id(sid, local_file=None, dry_run=dry, mapping=bulk_mapping)
                    after = get_record_count()
                    results.append({"source_id": sid, "status": "ok", "rows": rows, "delta": after - before})
                except Exception as ex:
                    results.append({"source_id": sid, "status": "error", "error": str(ex)[:100]})
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            errs = validate_registry()
            if errs:
                st.warning("Registry validation issues (first 3):\n" + "\n".join(errs[:3]))


def _render_single_source(reg_df: pd.DataFrame) -> None:
    st.markdown("### Single Source")
    if reg_df.empty:
        st.warning("Add sources in Registry first.")
        return
    picked = st.selectbox("Source", reg_df["source_id"].tolist(), key="mira_single_scrape")
    row = reg_df[reg_df["source_id"] == picked].iloc[0]
    st.write(f"**{row['source_name']}** — `{row.get('scraper_type', '')}`")

    c1, c2 = st.columns(2)
    with c1:
        dry = st.checkbox("Dry run", key="mira_single_dry")
        if st.button("Run ingest → records", type="primary", key="mira_single_ingest"):
            with st.spinner("Ingesting..."):
                try:
                    result = ingest_source_id(picked, dry_run=dry)
                    st.success(
                        f"Loaded {result['rows_loaded']} rows; "
                        f"records {result['records_before']:,} → {result['records_after']:,}"
                    )
                except Exception as exc:
                    st.error(str(exc))
    with c2:
        url = row.get("source_url")
        if url and st.button("Test fetcher", key="mira_test_fetch"):
            with st.spinner("Fetching..."):
                content = test_fetch_url(str(url), source_id=picked)
                if content:
                    st.success(f"Fetched {len(content)} chars")
                    with st.expander("Preview"):
                        st.code(content[:2000])
                else:
                    st.warning("Fetch failed.")


def _render_quick_test() -> None:
    st.markdown("### Quick Test URL")
    test_url = st.text_input("Quick test URL", value="https://example.com", key="mira_any_url")
    if st.button("Test URL", key="mira_any_fetch"):
        content = test_fetch_url(test_url)
        if content:
            st.success(f"OK ({len(content)} bytes)")
            st.code(content[:1500])
        else:
            st.error("Fetch failed.")


def render_scrape_tab(reg_df: pd.DataFrame) -> None:
    st.subheader("Scrape Control")
    fetcher = get_fetcher_status()
    st.caption(f"Playwright: {'yes' if fetcher['playwright_available'] else 'no'}")

    # Lateral flow: each scrape method is a step laid out across the top with its
    # own execute button inside — no long vertical scroll to reach a given tool.
    tab_patch, tab_single, tab_quick, tab_bulk = st.tabs(
        ["🧩 Patch Scrape", "📄 Single Source", "🔗 Quick Test", "📦 Bulk URLs"]
    )
    with tab_patch:
        _render_patch_scrape()
    with tab_single:
        _render_single_source(reg_df)
    with tab_quick:
        _render_quick_test()
    with tab_bulk:
        _render_bulk_scrape()


def render_records_tab(reg_df: pd.DataFrame) -> None:
    st.subheader("Records Explorer")

    stats = records_stats_df()
    st.dataframe(stats, hide_index=True)
    st.write("**By source**")
    st.dataframe(records_by_source_df(), use_container_width=True, hide_index=True)

    st.divider()
    source_ids = reg_df["source_id"].tolist() if not reg_df.empty else []
    filter_id = st.selectbox("Filter", ["(all)"] + source_ids, key="mira_rec_filter")
    sid = None if filter_id == "(all)" else filter_id
    recs = load_records_df(source_id=sid, limit=5000)
    st.caption(f"Showing {len(recs):,} rows")
    st.dataframe(recs, use_container_width=True, hide_index=True)

    if not recs.empty:
        st.download_button(
            "Download CSV",
            data=recs.to_csv(index=False).encode("utf-8"),
            file_name=f"records_{filter_id}.csv",
            mime="text/csv",
        )

    st.divider()
    with st.expander("Advanced: SQL query (SELECT only)"):
        query = st.text_area(
            "SQL", value="SELECT * FROM records ORDER BY period DESC LIMIT 20"
        )
        if st.button("Run query", key="mira_sql"):
            if not query.strip().upper().startswith("SELECT"):
                st.warning("Only SELECT queries allowed.")
            else:
                try:
                    result = run_records_query(query)
                    st.dataframe(result, use_container_width=True)
                except Exception as exc:
                    st.error(str(exc))


def render_activity_tab() -> None:
    st.subheader("Activity & Review")

    st.write("**Unified activity** (acquire + pipeline)")
    activity = unified_activity_df(limit=50)
    if activity.empty:
        st.info("No activity logged yet.")
    else:
        st.dataframe(activity, use_container_width=True, hide_index=True)

    st.divider()
    st.write("**Scraper run log**")
    log_entries = load_scraper_run_log(limit=30)
    if log_entries:
        st.dataframe(pd.DataFrame(log_entries), use_container_width=True, hide_index=True)
    else:
        st.info("No scraper runs yet.")

    st.divider()
    st.write("**Review queue**")
    review = load_review_queue()
    if review:
        st.dataframe(pd.DataFrame(review), use_container_width=True, hide_index=True)
        if st.button("Clear review queue"):
            clear_review_queue()
            st.success("Cleared.")
            st.rerun()
    else:
        st.success("Review queue empty.")


def render_sources_page() -> None:
    """Main entry for the Sources & Records Streamlit page."""
    page_hero(
        "Sources & Records",
        "Scrape → canonical records (DuckDB) → load into a Project on Home.",
    )

    status = get_acquire_status()
    reg_df = load_registry_df()
    db_path = str(status["db_path"])
    short_db = db_path if len(db_path) <= 34 else "…" + db_path[-33:]

    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        st.metric("Records", f"{status['record_count']:,}")
    with c2:
        st.metric("Sources", f"{len(reg_df):,}")
    with c3:
        st.metric("Database", short_db, help=db_path)
    with c4:
        st.write("")  # vertical nudge to align button with metrics
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()

    tab_reg, tab_scrape, tab_records, tab_activity = st.tabs(
        ["Registry", "Scrape", "Records", "Activity"]
    )
    with tab_reg:
        render_registry_tab()
    with tab_scrape:
        render_scrape_tab(reg_df)
    with tab_records:
        render_records_tab(reg_df)
    with tab_activity:
        render_activity_tab()