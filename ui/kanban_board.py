"""
ui/kanban_board.py — per-project Kanban board.

Renders board columns as `st.columns` with bordered card containers. Cards move
between columns with buttons (no drag-drop dependency); reordering uses
float-midpoint indexes. Derived pipeline-stage cards are shown read-only.
"""

from __future__ import annotations

import streamlit as st

import json

from core import board
from core import context as context_mod
from ui.theme import page_hero
from core import executor as executor_mod
from core import state
from core.git_sync import ci_status_line, git_status, remote_ci_status, status_line
from core.project import Project

_PRIORITY_BADGE = {
    "urgent": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪",
}


def render_kanban_board(project: Project) -> None:
    page_hero(f"📋 Board — {project.meta.get('name', project.id)}",
              "Plan, track, and ship work for this project.")

    _render_git_header(project)
    _render_add_form(project)

    show_stages = st.toggle("Show pipeline-stage cards", value=True,
                            key="board_show_stages")

    columns = board.board_columns(project)
    full = board.get_board(project, include_stage_cards=show_stages)
    cols = st.columns(len(columns))

    # Project-level fingerprint is the same for every card — compute once (one
    # git call) and compare per card to flag stale context cheaply.
    current_fp = context_mod.fingerprint(project)

    for col_name, col_widget in zip(columns, cols):
        with col_widget:
            cards = full.get(col_name, [])
            st.subheader(f"{col_name.replace('_', ' ').title()}  ({len(cards)})")
            for card in cards:
                _render_card(project, card, columns, current_fp)


def _render_git_header(project: Project) -> None:
    status = git_status(project)
    icon = "🟢" if status.get("is_repo") else "⚪"
    cols = st.columns(2)
    with cols[0]:
        st.caption(f"{icon} git: {status_line(status)}")
    with cols[1]:
        if status.get("is_repo"):
            st.caption(ci_status_line(remote_ci_status(project)))


def _render_add_form(project: Project) -> None:
    with st.expander("➕ Add task", expanded=False):
        with st.form("add_task_form", clear_on_submit=True):
            title = st.text_input("Title")
            description = st.text_area("Description", height=80)
            c1, c2, c3 = st.columns(3)
            with c1:
                column = st.selectbox("Column", board.board_columns(project))
            with c2:
                priority = st.selectbox(
                    "Priority", ["low", "medium", "high", "urgent"], index=1)
            with c3:
                points = st.select_slider(
                    "Points", options=[0, *board.FIBONACCI_POINTS], value=0,
                    help="Fibonacci story points (0 = unestimated)")
            if st.form_submit_button("Add", type="primary") and title.strip():
                board.add_task(
                    project, title.strip(),
                    description=description.strip() or None,
                    column=column, priority=priority,
                    points=points or None,
                )
                st.rerun()


def _render_card(project: Project, card: board.Task, columns: list[str],
                 current_fp: str) -> None:
    is_stage = card.kind == "stage"
    # Executor binding is a pure graph match (no I/O) — cheap to show on every card.
    spec = context_mod.context_spec(card)
    with st.container(border=True):
        badge = _PRIORITY_BADGE.get(card.priority, "")
        pts = f"  ·  {card.points}pt" if card.points else ""
        lock = "🔒 " if is_stage else ""
        stale = (not is_stage and card.context_fingerprint
                 and card.context_fingerprint != current_fp)
        stale_badge = "🟡 " if stale else ""
        st.markdown(f"{lock}{stale_badge}{badge} **{card.title}**{pts}")
        st.caption(f"⚙ {spec['skill']}")
        if card.description:
            st.caption(card.description)
        if card.blockers:
            st.caption("⛔ " + "; ".join(card.blockers))
        if card.risks:
            st.caption("⚠️ " + "; ".join(card.risks))

        if is_stage:
            # Derived cards are read-only; they track pipeline progress.
            return

        _render_context_run(project, card, stale)

        idx = columns.index(card.column) if card.column in columns else 0
        move_cols = st.columns(2)
        with move_cols[0]:
            if idx > 0 and st.button("◀", key=f"l_{card.id}",
                                     help=f"Move to {columns[idx - 1]}"):
                board.move_task(project, card.id, columns[idx - 1])
                st.rerun()
        with move_cols[1]:
            if idx < len(columns) - 1 and st.button(
                    "▶", key=f"r_{card.id}", help=f"Move to {columns[idx + 1]}"):
                board.move_task(project, card.id, columns[idx + 1])
                st.rerun()
        if st.button("🗑", key=f"d_{card.id}", help="Delete"):
            board.delete_task(project, card.id)
            st.rerun()


def _render_context_run(project: Project, card: board.Task, stale: bool) -> None:
    """Context brief + execute controls for a stored card.

    Reads the persisted snapshot directly (no live resolve in the hot path);
    only Run / Dry run / Refresh resolve against current state.
    """
    label = "🧭 Context & run" + ("  · 🟡 stale" if stale else "")
    with st.expander(label, expanded=False):
        prov = st.selectbox("Executor provider", ["offline", "ollama", "anthropic"],
                            key=f"prov_{card.id}",
                            help="offline = hand-off brief; ollama/anthropic = LLM run")
        b1, b2, b3 = st.columns(3)
        with b1:
            dry = st.button("Dry run", key=f"dry_{card.id}")
        with b2:
            run = st.button("Run", type="primary", key=f"run_{card.id}")
        with b3:
            refresh = st.button("Refresh ctx", key=f"ref_{card.id}")

        if refresh:
            context_mod.assign_context(project, card)
            st.rerun()
        if dry or run:
            with st.spinner("Executing…"):
                res = executor_mod.execute_task(project, card, provider=prov,
                                                dry_run=dry)
            {"success": st.success, "ready": st.info, "blocked": st.warning,
             "failed": st.error, "error": st.error}.get(res.status, st.write)(
                f"{res.status.upper()} · {res.skill}")
            if res.output:
                st.code(res.output[:4000])
            if run and res.status == "success":
                st.rerun()  # card may have advanced to done
            return

        # Show the persisted brief, if any.
        row = state.get_task(project.root / "state.db", card.id)
        snap = row.get("context_snapshot_json") if row else None
        if snap:
            st.markdown(context_mod.render_brief(json.loads(snap)))
        else:
            st.caption("No context assigned yet — Dry run or Refresh to assign.")
