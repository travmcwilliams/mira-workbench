"""
tests/test_sprint.py

Phase 5: sprint planning + export. Offline planner is deterministic; LLM paths
fall back to offline when no model is reachable.
Run with: pytest tests/test_sprint.py -q
"""

import os
from pathlib import Path

import pytest


@pytest.fixture
def temp_projects_dir(monkeypatch, tmp_path):
    (tmp_path / "projects").mkdir()
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _project():
    from core.project import Project
    p = Project.create(name="Sprint Test", task_type="regression")
    p.meta["target_column"] = "price"
    p.meta["completed_stages"] = ["business", "acquire"]
    p.meta["current_stage"] = "acquire"
    p.save_meta()
    return p


def test_offline_plan_is_deterministic(temp_projects_dir):
    from core import sprint

    p = _project()
    plan_a = sprint.plan_sprint(p, provider="offline", persist=False)
    plan_b = sprint.plan_sprint(p, provider="offline", persist=False)

    # Identical context -> identical stories (titles + points + order).
    sig_a = [(s["title"], s["points"]) for s in plan_a["stories"]]
    sig_b = [(s["title"], s["points"]) for s in plan_b["stories"]]
    assert sig_a == sig_b
    # One story per remaining (incomplete) stage: 12 - 2 completed = 10.
    assert len(plan_a["stories"]) == 10
    assert all(s["points"] in (1, 2, 3, 5, 8, 13, 21) for s in plan_a["stories"])


def test_capacity_budget_limits_points(temp_projects_dir):
    from core import sprint

    p = _project()
    plan = sprint.plan_sprint(p, provider="offline", capacity=10, persist=False)
    assert plan["total_points"] <= 10
    assert len(plan["stories"]) >= 1


def test_persist_and_materialize(temp_projects_dir):
    from core import board, sprint, state

    p = _project()
    plan = sprint.plan_sprint(p, provider="offline", capacity=15,
                              persist=True, materialize=True)
    assert plan["sprint_id"] == 1

    sprints = state.list_sprints(p.root / "state.db")
    assert len(sprints) == 1 and sprints[0]["state"] == "planned"

    tasks = board.list_tasks(p)
    assert len(tasks) == len(plan["stories"])
    assert all(t.sprint_id == 1 for t in tasks)
    assert all("sprint" in t.labels for t in tasks)


def test_export_markdown_and_json(temp_projects_dir):
    from core import sprint

    p = _project()
    plan = sprint.plan_sprint(p, provider="offline", persist=False)
    md = sprint.export_markdown(plan)
    js = sprint.export_json(plan)

    assert md.startswith("# Sprint Plan — Sprint Test")
    assert "## User Stories" in md
    assert "price" in md  # target threaded through stories
    assert '"stories"' in js


def test_llm_path_falls_back_to_offline(temp_projects_dir, monkeypatch):
    from core import llm, sprint

    p = _project()

    def _boom(*a, **k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(llm, "generate", _boom)
    plan = sprint.plan_sprint(p, provider="ollama", persist=False)
    assert plan["stories"]  # still produced
    assert plan["provider"].startswith("offline (fallback from ollama")


def test_cli_sprint_plan_offline(temp_projects_dir, capsys):
    from mira_cli import main

    p = _project()
    rc = main(["sprint", "plan", "--project-id", p.id, "--provider", "offline",
               "--capacity", "12"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Sprint plan (offline)" in out
    assert "Saved as sprint #1" in out


def test_cli_sprint_export_markdown(temp_projects_dir, capsys):
    from mira_cli import main

    p = _project()
    rc = main(["sprint", "plan", "--project-id", p.id, "--provider", "offline",
               "--export", "md"])
    assert rc == 0
    assert "# Sprint Plan" in capsys.readouterr().out
