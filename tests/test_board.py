"""
tests/test_board.py

Phase 1: board data model + core API. All offline/deterministic — no LLM,
network, or git required.
Run with: pytest tests/test_board.py -q
"""

import os
from pathlib import Path

import pytest


@pytest.fixture
def temp_projects_dir(monkeypatch, tmp_path):
    """Isolate Project to a temporary projects/ directory for the test."""
    (tmp_path / "projects").mkdir()
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _new_project(name="Board Test"):
    from core.project import Project
    return Project.create(name=name, task_type="regression")


def test_add_and_list_task(temp_projects_dir):
    from core import board

    p = _new_project()
    tid = board.add_task(p, "Write ingestion adapter", points=5, priority="high")
    assert isinstance(tid, int)

    tasks = board.list_tasks(p)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.title == "Write ingestion adapter"
    assert t.points == 5
    assert t.priority == "high"
    assert t.column == "backlog"
    assert t.kind == "task"


def test_update_task_fields(temp_projects_dir):
    from core import board

    p = _new_project()
    tid = board.add_task(p, "Tune model")
    board.update_task(p, tid, points=8, blockers=["needs GPU"], description="grid search")

    t = board.list_tasks(p)[0]
    assert t.points == 8
    assert t.blockers == ["needs GPU"]
    assert t.description == "grid search"


def test_move_task_changes_column(temp_projects_dir):
    from core import board

    p = _new_project()
    tid = board.add_task(p, "Card")
    board.move_task(p, tid, "in_progress")

    t = board.list_tasks(p)[0]
    assert t.column == "in_progress"


def test_delete_task(temp_projects_dir):
    from core import board

    p = _new_project()
    tid = board.add_task(p, "Temp card")
    board.delete_task(p, tid)
    assert board.list_tasks(p) == []


def test_reorder_task_uses_midpoint(temp_projects_dir):
    from core import board

    p = _new_project()
    a = board.add_task(p, "A", column="todo")
    b = board.add_task(p, "B", column="todo")
    c = board.add_task(p, "C", column="todo")

    tasks = {t.title: t for t in board.list_tasks(p)}
    # Move C between A and B.
    new_index = board.reorder_task(p, c, before=tasks["A"], after=tasks["B"])
    assert tasks["A"].order_index < new_index < tasks["B"].order_index

    ordered = [t.title for t in board.list_tasks(p)]
    assert ordered == ["A", "C", "B"]


def test_stage_cards_reflect_pipeline_progress(temp_projects_dir):
    from core import board
    from core.pipeline import STAGE_ORDER

    p = _new_project()
    p.meta["completed_stages"] = ["business", "profile"]
    p.meta["current_stage"] = "clean"
    p.save_meta()

    cards = {c.stage: c for c in board.stage_cards(p)}
    assert len(cards) == len(STAGE_ORDER)
    assert cards["business"].column == "done"
    assert cards["profile"].column == "done"
    assert cards["clean"].column == "in_progress"
    assert cards["model"].column == "backlog"
    # Stage cards are derived/read-only and carry no db id.
    assert all(c.kind == "stage" and c.id is None for c in cards.values())


def test_get_board_merges_tasks_and_stage_cards(temp_projects_dir):
    from core import board

    p = _new_project()
    p.meta["completed_stages"] = ["business"]
    p.meta["current_stage"] = "profile"
    p.save_meta()
    board.add_task(p, "User story 1", column="todo")

    full = board.get_board(p, include_stage_cards=True)
    assert "todo" in full
    todo_titles = [t.title for t in full["todo"]]
    assert "User story 1" in todo_titles

    # business stage card landed in "done"
    done_stages = [t.stage for t in full["done"] if t.kind == "stage"]
    assert "business" in done_stages

    # Without stage cards, only the stored task remains.
    tasks_only = board.get_board(p, include_stage_cards=False)
    assert all(t.kind == "task" for col in tasks_only.values() for t in col)


def test_ensure_board_schema_self_heals_old_project(temp_projects_dir):
    """A state.db that predates board tables should upgrade on first use."""
    import sqlite3

    from core import board
    from core.state import init_db

    p = _new_project()
    db = p.root / "state.db"
    # Simulate a legacy db: drop the board tables init_db just created.
    conn = sqlite3.connect(db)
    for tbl in ("tasks", "sprints", "gate_runs", "trigger_events"):
        conn.execute(f"DROP TABLE IF EXISTS {tbl}")
    conn.commit()
    conn.close()

    # Board ops must transparently recreate the schema.
    tid = board.add_task(p, "Recovered card")
    assert board.list_tasks(p)[0].id == tid
