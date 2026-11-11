"""
core/sprint.py

Sprint planning workflow: map a project's context ("skills") into a sprint plan
of user stories with Fibonacci points, blockers and risks. The plan can be
persisted, materialized into board tasks, and exported to markdown / JSON.
"""

from __future__ import annotations

import json
from typing import Any

from core import board, llm, state
from core.git_sync import status_line
from core.pipeline import STAGE_ORDER
from core.project import Project


def llm_config(project: Project) -> dict[str, Any]:
    """Self-healing meta["llm"] config (provider/model)."""
    cfg = project.meta.setdefault("llm", {})
    changed = False
    if "provider" not in cfg:
        cfg["provider"] = "offline"
        changed = True
    if "model" not in cfg:
        cfg["model"] = None
        changed = True
    if changed:
        project.save_meta()
    return cfg


def build_context(project: Project, *, capacity: int | None = None) -> dict[str, Any]:
    """Gather the project 'skills'/context the planner reasons over."""
    completed = project.meta.get("completed_stages", []) or []
    remaining = [s for s in STAGE_ORDER if s not in completed]
    open_tasks = [t.title for t in board.list_tasks(project) if t.column != "done"]

    git_summary = None
    try:
        from core.git_sync import git_status
        git_summary = status_line(git_status(project))
    except Exception:
        git_summary = None

    dataset = None
    try:
        from core import dataset_analysis
        dataset = dataset_analysis.compact_summary(
            dataset_analysis.analyze_project(project))
    except Exception:
        dataset = None

    return {
        "project_name": project.meta.get("name", project.id),
        "task_type": project.meta.get("task_type"),
        "target": project.meta.get("target_column"),
        "schema": project.meta.get("schema", {}),
        "dataset": dataset,
        "completed_stages": completed,
        "current_stage": project.meta.get("current_stage"),
        "remaining_stages": remaining,
        "open_tasks": open_tasks,
        "git": git_summary,
        "capacity": capacity,
    }


def plan_sprint(
    project: Project,
    *,
    capacity: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    persist: bool = True,
    materialize: bool = False,
) -> dict[str, Any]:
    """Generate (and optionally persist/materialize) a sprint plan.

    Returns the plan dict. When `materialize`, each story becomes a backlog task
    linked to the created sprint.
    """
    cfg = llm_config(project)
    provider = provider or cfg.get("provider", "offline")
    model = model or cfg.get("model")

    context = build_context(project, capacity=capacity)
    plan = llm.plan_sprint_llm(context, provider=provider, model=model)
    plan["context"] = context

    if persist:
        sprint_id = state.add_sprint(project.root / "state.db", {
            "name": _sprint_name(project),
            "goal": plan.get("goal", ""),
            "state": "planned",
            "capacity_points": capacity,
            "plan_json": plan,
            "provider": plan.get("provider"),
        })
        plan["sprint_id"] = sprint_id
        if materialize:
            _materialize_stories(project, plan, sprint_id)

    return plan


def _sprint_name(project: Project) -> str:
    existing = state.list_sprints(project.root / "state.db")
    return f"Sprint {len(existing) + 1}"


def _materialize_stories(project: Project, plan: dict[str, Any], sprint_id: int) -> None:
    for story in plan.get("stories", []):
        board.add_task(
            project,
            story.get("title", "Untitled story"),
            description=_story_description(story),
            column="backlog",
            priority=story.get("priority", "medium"),
            points=story.get("points"),
            blockers=story.get("blockers") or None,
            risks=story.get("risks") or None,
            labels=["sprint"],
            sprint_id=sprint_id,
            stage=story.get("stage"),
        )


def _story_description(story: dict[str, Any]) -> str:
    lines = []
    if story.get("as_a") or story.get("i_want"):
        lines.append(
            f"As a {story.get('as_a', 'user')}, I want {story.get('i_want', '...')}"
            f"{(' so that ' + story['so_that']) if story.get('so_that') else ''}."
        )
    ac = story.get("acceptance_criteria") or []
    if ac:
        lines.append("Acceptance criteria:")
        lines.extend(f"- {c}" for c in ac)
    return "\n".join(lines)


# --- Export ------------------------------------------------------------------

def export_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, indent=2)


def export_markdown(plan: dict[str, Any]) -> str:
    ctx = plan.get("context", {})
    lines: list[str] = []
    name = ctx.get("project_name", "Project")
    lines.append(f"# Sprint Plan — {name}")
    if plan.get("provider"):
        lines.append(f"_Generated by: {plan['provider']}_")
    lines.append("")
    lines.append(f"**Goal:** {plan.get('goal', '')}")
    total = plan.get("total_points")
    if total is None:
        total = sum(s.get("points") or 0 for s in plan.get("stories", []))
    lines.append(f"**Total points:** {total}")
    if ctx.get("capacity") is not None:
        lines.append(f"**Capacity:** {ctx['capacity']}")
    lines.append("")

    lines.append("## User Stories")
    for i, s in enumerate(plan.get("stories", []), 1):
        pts = s.get("points")
        prio = s.get("priority", "medium")
        lines.append(f"### {i}. {s.get('title', 'Untitled')}  "
                     f"({pts} pts · {prio})")
        if s.get("as_a"):
            lines.append(f"> As a {s.get('as_a')}, I want {s.get('i_want', '')}"
                         f"{(' so that ' + s['so_that']) if s.get('so_that') else ''}.")
        for c in s.get("acceptance_criteria", []):
            lines.append(f"- [ ] {c}")
        if s.get("risks"):
            lines.append(f"- ⚠️ Risks: {', '.join(s['risks'])}")
        if s.get("blockers"):
            lines.append(f"- ⛔ Blockers: {', '.join(s['blockers'])}")
        lines.append("")

    if plan.get("risks"):
        lines.append("## Sprint Risks")
        lines.extend(f"- {r}" for r in plan["risks"])
        lines.append("")
    if plan.get("blockers"):
        lines.append("## Sprint Blockers")
        lines.extend(f"- {b}" for b in plan["blockers"])
        lines.append("")

    return "\n".join(lines)
