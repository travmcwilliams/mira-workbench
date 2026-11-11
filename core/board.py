"""
core/board.py

Per-project Kanban board: free-form work-item tasks plus auto-derived cards for
the 12 pipeline stages. Stored tasks live in state.db (see core/state.py); stage
cards are a pure function of the project's pipeline progress so there is no sync
duplication.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from core import state
from core.pipeline import STAGE_ORDER
from core.project import Project

# Default board columns. Stored in meta["board"]["columns"]; overridable per project.
DEFAULT_COLUMNS: list[str] = ["backlog", "todo", "in_progress", "done"]

# Valid Fibonacci story points for prioritization/estimation.
FIBONACCI_POINTS: list[int] = [1, 2, 3, 5, 8, 13, 21]

# Human labels for the derived stage cards (kept local to avoid a UI import).
STAGE_LABELS: dict[str, str] = {
    "business": "Business Understanding",
    "acquire": "Acquire Data",
    "profile": "Profile",
    "eda": "EDA",
    "clean": "Clean",
    "feature_eng": "Feature Engineering",
    "feature_select": "Feature Selection",
    "data_prep": "Data Prep",
    "model": "Model",
    "evaluate": "Evaluate",
    "explain": "Explain",
    "deploy": "Deploy",
}


@dataclass
class Task:
    """A board card. `stage` set => derived pipeline-stage card (read-only)."""

    id: Optional[int] = None
    title: str = ""
    description: Optional[str] = None
    column: str = "backlog"
    priority: str = "medium"
    points: Optional[int] = None
    order_index: float = 0.0
    sprint_id: Optional[int] = None
    blockers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    assignee: Optional[str] = None
    labels: list[str] = field(default_factory=list)
    stage: Optional[str] = None
    kind: str = "task"  # "task" (stored, editable) | "stage" (derived, read-only)
    context_fingerprint: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _row_to_task(row: dict[str, Any]) -> Task:
    return Task(
        id=row.get("id"),
        title=row.get("title", ""),
        description=row.get("description"),
        column=row.get("col", "backlog"),
        priority=row.get("priority", "medium"),
        points=row.get("points"),
        order_index=float(row.get("order_index") or 0.0),
        sprint_id=row.get("sprint_id"),
        blockers=_json_list(row.get("blockers_json")),
        risks=_json_list(row.get("risks_json")),
        assignee=row.get("assignee"),
        labels=_json_list(row.get("labels_json")),
        stage=row.get("stage"),
        kind="task",
        context_fingerprint=row.get("context_fingerprint"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


# --- Board config (lives in meta.json, self-healing) -------------------------

def board_columns(project: Project) -> list[str]:
    board = project.meta.setdefault("board", {})
    cols = board.get("columns")
    if not cols:
        cols = list(DEFAULT_COLUMNS)
        board["columns"] = cols
        project.save_meta()
    return cols


# --- Stored task operations --------------------------------------------------

def _db(project: Project):
    return project.root / "state.db"


def add_task(
    project: Project,
    title: str,
    *,
    description: str | None = None,
    column: str = "backlog",
    priority: str = "medium",
    points: int | None = None,
    blockers: list[str] | None = None,
    risks: list[str] | None = None,
    assignee: str | None = None,
    labels: list[str] | None = None,
    sprint_id: int | None = None,
    stage: str | None = None,
) -> int:
    """Create a free-form task. Returns the new task id."""
    return state.add_task(_db(project), {
        "title": title,
        "description": description,
        "col": column,
        "priority": priority,
        "points": points,
        "blockers_json": blockers or [],
        "risks_json": risks or [],
        "assignee": assignee,
        "labels_json": labels or [],
        "sprint_id": sprint_id,
        "stage": stage,
    })


def update_task(project: Project, task_id: int, **fields: Any) -> None:
    """Update editable fields. Accepts board-level names (column/blockers/...)."""
    mapped: dict[str, Any] = {}
    rename = {"column": "col", "blockers": "blockers_json",
              "risks": "risks_json", "labels": "labels_json"}
    for k, v in fields.items():
        mapped[rename.get(k, k)] = v
    state.update_task(_db(project), task_id, mapped)


def move_task(project: Project, task_id: int, column: str,
              order_index: float | None = None) -> None:
    """Move a card to another column. Fires the task_done trigger when applicable."""
    state.move_task(_db(project), task_id, column, order_index)
    if column == "done":
        _fire_task_done(project, task_id)


def delete_task(project: Project, task_id: int) -> None:
    state.delete_task(_db(project), task_id)


def reorder_task(project: Project, task_id: int, before: Task | None,
                 after: Task | None) -> float:
    """Set a task's order_index to the midpoint of its new neighbors.

    `before`/`after` are the cards that should sit immediately above/below.
    Returns the new order_index.
    """
    lo = before.order_index if before else 0.0
    hi = after.order_index if after else (lo + 2.0)
    new_index = (lo + hi) / 2.0
    state.update_task(_db(project), task_id, {"order_index": new_index})
    return new_index


# --- Derived stage cards -----------------------------------------------------

def _stage_card(project: Project, stage: str, order: int) -> Task:
    completed = project.meta.get("completed_stages", []) or []
    current = project.meta.get("current_stage")
    if stage in completed:
        column = "done"
    elif stage == current:
        column = "in_progress"
    else:
        column = "backlog"
    return Task(
        id=None,
        title=STAGE_LABELS.get(stage, stage.replace("_", " ").title()),
        description=f"Pipeline stage: {stage}",
        column=column,
        priority="medium",
        order_index=float(order),
        stage=stage,
        kind="stage",
        labels=["pipeline"],
    )


def stage_cards(project: Project) -> list[Task]:
    """Derived (read-only) cards mirroring pipeline progress."""
    return [_stage_card(project, s, i) for i, s in enumerate(STAGE_ORDER)]


# --- Board assembly ----------------------------------------------------------

def list_tasks(project: Project, sprint_id: int | None = None) -> list[Task]:
    """Stored free-form tasks only."""
    rows = state.list_tasks(_db(project), sprint_id=sprint_id)
    return [_row_to_task(r) for r in rows]


def get_board(project: Project, *, include_stage_cards: bool = True
              ) -> dict[str, list[Task]]:
    """Return {column: [Task, ...]} merging stored tasks and derived stage cards."""
    columns = board_columns(project)
    board: dict[str, list[Task]] = {c: [] for c in columns}

    cards: list[Task] = list(list_tasks(project))
    if include_stage_cards:
        cards += stage_cards(project)

    for card in cards:
        board.setdefault(card.column, []).append(card)
    for col in board:
        board[col].sort(key=lambda t: t.order_index)
    return board


# --- Trigger seam (wired up in Phase 4; safe no-op until then) ----------------

def _fire_task_done(project: Project, task_id: int) -> None:
    """Notify the CI/trigger layer that a task moved to done.

    Imported lazily so the board has no hard dependency on the CI module, and
    so a missing/failing CI layer never breaks a simple card move.
    """
    try:
        from core import ci  # noqa: PLC0415
    except ImportError:
        return
    on_event = getattr(ci, "on_event", None)
    if on_event is None:
        return
    try:
        on_event(project, "task_done", {"task_id": task_id})
    except Exception:
        # Triggers are best-effort; never let one break a board operation.
        pass
