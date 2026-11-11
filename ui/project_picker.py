"""Shared project picker — the one selector, rendered on every page.

Selects by project **id** (labels show "name (id)", so duplicate names can't
collide), reads and re-writes the ``?pid=`` query param so any page's URL is a
shareable deep link, and lists projects from a cached lightweight index rather
than re-loading every meta.json on each rerun.
"""

from __future__ import annotations

import streamlit as st

from core.project import Project, list_project_index

_NONE = "(none)"


@st.cache_data(ttl=30, show_spinner=False)
def _project_index() -> list[dict]:
    return list_project_index()


def clear_project_cache() -> None:
    """Call after creating a project so it appears without waiting out the TTL."""
    _project_index.clear()


def resolve_initial_pid(pid: str | None, ids: list[str]) -> str | None:
    """Pure helper: a ?pid= value is honored only if it names a real project."""
    return pid if pid and pid in ids else None


def render_project_picker() -> Project | None:
    """Render the sidebar selector; returns the active Project (or None)."""
    index = _project_index()
    ids = [e["id"] for e in index]
    labels = {e["id"]: f"{e['name']} ({e['id']})" for e in index}

    # Deep link: ?pid=<id> selects that project on load (from any page).
    if st.session_state.get("project") is None:
        try:
            pid = resolve_initial_pid(st.query_params.get("pid"), ids)
        except Exception:
            pid = None
        if pid:
            try:
                st.session_state.project = Project.load(pid)
            except Exception:
                pass

    current: Project | None = st.session_state.get("project")
    current_id = current.id if current else None

    selected = st.sidebar.selectbox(
        "Select project",
        options=[_NONE] + ids,
        index=(ids.index(current_id) + 1) if current_id in ids else 0,
        format_func=lambda v: v if v == _NONE else labels.get(v, v),
        key="project_selector",
    )

    if selected != _NONE and selected != current_id:
        try:
            st.session_state.project = Project.load(selected)
            st.query_params["pid"] = selected
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Failed to load project: {exc}")

    # Nav clicks / switch_page clear query params — re-stamp pid every render
    # so the address bar stays a shareable deep link on every page.
    if current_id is not None:
        try:
            if st.query_params.get("pid") != current_id:
                st.query_params["pid"] = current_id
        except Exception:
            pass

    return st.session_state.get("project")
