"""
core/context.py

Per-task context assignment. Implements the costed-hybrid binding chosen by the
first-principles analysis:

  - SNAPSHOT-PRIMARY: the frozen snapshot is the source of truth (provenance,
    A6) and, because measured intra-life drift is ~0 (A3 falsified), it rarely
    goes stale.
  - SPEC kept: the resolver's selectors are stored so context can be re-resolved
    live when needed (future interactive usage) without per-render cost.
  - FINGERPRINT (A5): cheap signal that detects the rare drift; checked on read,
    refreshed on trigger (new_data / stage re-run).

Assignment is one graph match (core/skills): a task's stage/kind picks a Skill,
the Skill's `consumes` is the context spec, and each consumed resource resolves
to a concrete value pulled from current project state.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from datetime import UTC
from pathlib import Path
from typing import Any

from core import skills, state
from core.project import Project


def _db(project: Project) -> Path:
    return project.root / "state.db"


# --- Fingerprint (cheap staleness signal) ------------------------------------

def fingerprint(project: Project) -> str:
    """Stable hash of the inputs a task's context depends on.

    Deterministic for a given project state; changes when schema, pipeline
    progress, working data, or the repo HEAD changes.
    """
    parts: dict[str, Any] = {
        "schema": sorted((project.meta.get("schema") or {}).keys()),
        "target": project.meta.get("target_column"),
        "completed": sorted(project.meta.get("completed_stages", []) or []),
    }
    wp = project.root / "data" / "working.parquet"
    try:
        parts["data_mtime"] = int(wp.stat().st_mtime) if wp.exists() else None
    except OSError:
        parts["data_mtime"] = None
    try:
        from core.git_sync import git_status
        st = git_status(project)
        lc = st.get("last_commit") or {}
        parts["git"] = lc.get("sha")
    except Exception:
        parts["git"] = None

    blob = json.dumps(parts, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


# --- Resource resolution -----------------------------------------------------

def _resolve_resource(project: Project, resource: str, task: Any) -> Any:
    """Map a symbolic consumed resource to a concrete value from project state."""
    meta = project.meta
    if resource == "facts.schema":
        return meta.get("schema", {})
    if resource == "facts.target":
        return meta.get("target_column")
    if resource == "data.working":
        last = state.get_last_run(_db(project), "profile")
        if last and last.get("summary_json"):
            s = json.loads(last["summary_json"])
            return {"rows": s.get("rows"), "cols": s.get("cols")}
        return {"rows": None, "cols": len(meta.get("schema", {})) or None}
    if resource == "data.analysis":
        from core import dataset_analysis
        return dataset_analysis.compact_summary(
            dataset_analysis.analyze_project(project))
    if resource == "intent.story":
        return {"title": getattr(task, "title", None),
                "description": getattr(task, "description", None)}
    if resource in ("repo.status", "repo.ci"):
        try:
            from core.git_sync import git_status, status_line
            return status_line(git_status(project))
        except Exception:
            return None
    if resource.endswith(".summary"):
        stage = resource.rsplit(".", 1)[0]
        last = state.get_last_run(_db(project), stage)
        if last and last.get("summary_json"):
            return json.loads(last["summary_json"])
        return None
    if resource.endswith(".output"):
        stage = resource.rsplit(".", 1)[0]
        completed = meta.get("completed_stages", []) or []
        return {"stage": stage, "available": stage in completed}
    return None


# --- Spec + assembly ---------------------------------------------------------

def context_spec(task: Any) -> dict[str, Any]:
    """Resolve the executor + needs for a task (the graph match)."""
    skill = skills.match_skill(
        stage=getattr(task, "stage", None),
        kind=getattr(task, "kind", "task"),
        labels=getattr(task, "labels", None),
    )
    return {
        "skill": skill.name,
        "skill_kind": skill.kind,
        "needs": list(skill.consumes),
        "produces": list(skill.produces),
    }


def assemble_context(project: Project, task: Any, spec: dict[str, Any] | None = None
                     ) -> dict[str, Any]:
    """Resolve a task's spec against *current* project state into canonical context."""
    spec = spec or context_spec(task)
    resolved: dict[str, Any] = {}
    for resource in spec["needs"]:
        try:
            resolved[resource] = _resolve_resource(project, resource, task)
        except Exception as exc:  # defensive: one bad resource shouldn't fail assembly
            resolved[resource] = {"error": str(exc)}

    return {
        "task_id": getattr(task, "id", None),
        "title": getattr(task, "title", None),
        "skill": spec["skill"],
        "skill_kind": spec["skill_kind"],
        "needs": spec["needs"],
        "produces": spec["produces"],
        "resolved": resolved,
        "intent": {
            "title": getattr(task, "title", None),
            "description": getattr(task, "description", None),
            "blockers": list(getattr(task, "blockers", []) or []),
            "risks": list(getattr(task, "risks", []) or []),
        },
        "project": {
            "name": project.meta.get("name", project.id),
            "task_type": project.meta.get("task_type"),
            "target": project.meta.get("target_column"),
            "current_stage": project.meta.get("current_stage"),
        },
        "fingerprint": fingerprint(project),
        "assigned_at": datetime.datetime.now(UTC).isoformat() + "Z",
    }


# --- Assignment (snapshot-primary) -------------------------------------------

def assign_context(project: Project, task: Any) -> dict[str, Any]:
    """Resolve live and persist the snapshot for a *stored* task. Returns it."""
    spec = context_spec(task)
    snapshot = assemble_context(project, task, spec)
    task_id = getattr(task, "id", None)
    if task_id is not None:
        state.set_task_context(
            _db(project), task_id,
            spec=spec, snapshot=snapshot, fingerprint=snapshot["fingerprint"],
        )
    return snapshot


def get_context(project: Project, task: Any, *, refresh: bool = False
                ) -> dict[str, Any]:
    """Return a stored task's context.

    Snapshot-primary: returns the frozen snapshot unless it is missing, the
    caller forces `refresh`, or the fingerprint shows drift. Stage cards (no id)
    are always assembled fresh and never persisted.
    """
    task_id = getattr(task, "id", None)
    if task_id is None:
        return assemble_context(project, task)

    row = state.get_task(_db(project), task_id)
    snap_raw = row.get("context_snapshot_json") if row else None
    if not refresh and snap_raw:
        snapshot = json.loads(snap_raw)
        if snapshot.get("fingerprint") == fingerprint(project):
            return snapshot  # fresh; no resolver cost
    return assign_context(project, task)


def is_stale(project: Project, task: Any) -> bool:
    task_id = getattr(task, "id", None)
    if task_id is None:
        return False
    row = state.get_task(_db(project), task_id)
    fp = row.get("context_fingerprint") if row else None
    if not fp:
        return True  # never assigned
    return fp != fingerprint(project)


def refresh_stale(project: Project) -> list[int]:
    """Re-assign context for every stored task whose fingerprint drifted.

    Called from the trigger seam (new_data / stage re-run). Returns refreshed ids.
    """
    from core import board
    current = fingerprint(project)
    refreshed: list[int] = []
    for task in board.list_tasks(project):
        row = state.get_task(_db(project), task.id)
        fp = row.get("context_fingerprint") if row else None
        if fp and fp == current:
            continue
        if fp is None:
            continue  # never assigned -> assign on demand, not on every drift
        assign_context(project, task)
        refreshed.append(task.id)
    return refreshed


# --- Render (C5: one canonical context, projected per consumer) --------------

def render_brief(context: dict[str, Any]) -> str:
    """Markdown brief for a human or an LLM/agent executor."""
    p = context.get("project", {})
    lines = [f"# Task brief — {context.get('title', 'Untitled')}",
             f"_Executor: {context.get('skill')} ({context.get('skill_kind')})_",
             ""]
    intent = context.get("intent", {})
    if intent.get("description"):
        lines += ["## Intent", intent["description"], ""]
    lines += ["## Project",
              f"- **{p.get('name')}** · {p.get('task_type')} · target `{p.get('target')}`",
              f"- current stage: {p.get('current_stage')}", ""]

    resolved = context.get("resolved", {})
    if resolved:
        lines.append("## Context the executor needs")
        for res, val in resolved.items():
            if res == "data.analysis":
                continue  # rendered as its own section below
            lines.append(f"- `{res}`: {_short(val)}")
        lines.append("")

    analysis = resolved.get("data.analysis")
    if isinstance(analysis, dict) and analysis.get("available"):
        lines += [_render_analysis(analysis), ""]
    if context.get("produces"):
        lines += ["## Expected outputs",
                  ", ".join(f"`{x}`" for x in context["produces"]), ""]
    if intent.get("blockers"):
        lines.append("⛔ Blockers: " + "; ".join(intent["blockers"]))
    if intent.get("risks"):
        lines.append("⚠️ Risks: " + "; ".join(intent["risks"]))
    lines.append(f"\n_fingerprint: {context.get('fingerprint')}_")
    return "\n".join(lines)


def _render_analysis(a: dict[str, Any]) -> str:
    """Render the compact dataset-analysis summary inside a brief."""
    lines = ["## Dataset analysis",
             f"- {a.get('rows')} rows × {a.get('cols')} cols"
             + ("  (sampled)" if a.get("sampled") else ""),
             f"- missing cells: {a.get('missing_cell_pct')}%  ·  "
             f"duplicate rows: {a.get('duplicate_rows')}"]
    tgt = a.get("target") or {}
    if tgt.get("kind") == "classification":
        lines.append(f"- target `{tgt.get('name')}`: {tgt.get('n_classes')} classes, "
                     f"imbalance {tgt.get('imbalance_ratio')}×")
    elif tgt.get("kind") == "regression":
        lines.append(f"- target `{tgt.get('name')}`: mean {tgt.get('mean')}, "
                     f"std {tgt.get('std')}, skew {tgt.get('skew')}")
    if a.get("leakage"):
        lines.append("- ⚠️ possible leakage: " + "; ".join(a["leakage"]))
    if a.get("issues"):
        lines.append("- quality flags: " + "; ".join(a["issues"]))
    rec = a.get("recommend", {})
    if rec.get("drop"):
        lines.append("- recommend drop: " + ", ".join(rec["drop"]))
    if rec.get("impute"):
        lines.append("- recommend impute: "
                     + ", ".join(f"{c}→{s}" for c, s in rec["impute"].items()))
    if rec.get("encode"):
        lines.append("- recommend encode: "
                     + ", ".join(f"{c}→{e}" for c, e in rec["encode"].items()))
    return "\n".join(lines)


def _short(val: Any, limit: int = 160) -> str:
    s = json.dumps(val) if not isinstance(val, str) else val
    return s if len(s) <= limit else s[:limit] + "…"
