"""
tests/test_context.py

Phase 7: context + skills assignment. Verifies the first-principles decisions:
skill contract (C1/C2), snapshot-primary resolve-on-trigger binding (C3), the
one graph-match (C4), and the canonical-context brief (C5). Offline/deterministic.
Run with: pytest tests/test_context.py -q
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
    p = Project.create(name="Ctx Test", task_type="regression")
    p.meta["target_column"] = "price"
    p.meta["schema"] = {"a": "num", "b": "num", "price": "num"}
    p.save_meta()
    return p


# --- Skill contract (C1/C2) --------------------------------------------------

def test_stage_skills_cover_all_stages_with_seeded_produces():
    from core import skills
    from core.pipeline import STAGE_ORDER

    for stage in STAGE_ORDER:
        sk = skills.stage_skill(stage)
        assert sk is not None and sk.kind == "stage"
        # produces always carries output + summary markers
        assert f"{stage}.output" in sk.produces
        assert f"{stage}.summary" in sk.produces
    # produces seeded from observed history (E2): model exposes best_model
    assert "model.best_model" in skills.stage_skill("model").produces
    # lineage: each stage consumes the previous stage's output
    assert "clean.output" in skills.stage_skill("feature_eng").consumes


def test_match_skill_routes_stage_vs_freeform():
    from core import skills

    stage_sk = skills.match_skill(stage="model", kind="stage")
    assert stage_sk.kind == "stage" and stage_sk.stage == "model"

    free_sk = skills.match_skill(stage=None, kind="task")
    assert free_sk.kind == "agent"  # free-form work -> llm_agent


# --- Graph match + assembly (C4) ---------------------------------------------

def test_context_spec_is_skill_consumes(temp_projects_dir):
    from core import board, context

    p = _project()
    tid = board.add_task(p, "Train the model", stage="model")
    task = board.list_tasks(p)[0]
    spec = context.context_spec(task)
    assert spec["skill"] == "run_stage:model"
    assert "data_prep.summary" in spec["needs"]
    assert spec["needs"] == context.skills.stage_skill("model").consumes


def test_assemble_pulls_upstream_summary(temp_projects_dir):
    from core import board, context
    from core.state import log_run

    p = _project()
    # Simulate an upstream data_prep run having produced facts.
    log_run(p.root / "state.db", stage="data_prep", settings={}, status="success",
            duration_ms=5, summary={"n_train": 80, "n_test": 20, "scaler": "none"})
    p.meta["completed_stages"] = ["data_prep"]
    p.save_meta()

    board.add_task(p, "Train", stage="model")
    task = board.list_tasks(p)[0]
    ctx = context.assemble_context(p, task)

    assert ctx["resolved"]["facts.target"] == "price"
    assert ctx["resolved"]["data_prep.summary"]["n_train"] == 80
    assert ctx["resolved"]["data_prep.output"]["available"] is True


# --- Snapshot-primary, resolve-on-trigger (C3) -------------------------------

def test_get_context_is_snapshot_primary(temp_projects_dir, monkeypatch):
    from core import board, context

    p = _project()
    tid = board.add_task(p, "Work item")
    task = board.list_tasks(p)[0]

    first = context.assign_context(p, task)  # resolve + persist
    fp = first["fingerprint"]

    # On a fresh read, the resolver must NOT run again (cost-correct per A3=0).
    monkeypatch.setattr(context, "assemble_context",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("resolver ran")))
    cached = context.get_context(p, task)
    assert cached["fingerprint"] == fp  # served from snapshot, no resolve


def test_fingerprint_changes_and_marks_stale(temp_projects_dir):
    from core import board, context

    p = _project()
    board.add_task(p, "Work item")
    task = board.list_tasks(p)[0]
    context.assign_context(p, task)
    assert context.is_stale(p, task) is False

    # Schema drift -> fingerprint changes -> stale.
    p.meta["schema"]["new_col"] = "num"
    p.save_meta()
    assert context.is_stale(p, task) is True


def test_new_data_trigger_refreshes_stale_context(temp_projects_dir):
    from core import board, context
    from core.ingest import ingest_dataframe
    import pandas as pd

    p = _project()
    board.add_task(p, "Work item")
    task = board.list_tasks(p)[0]
    context.assign_context(p, task)
    old_fp = context.fingerprint(p)

    # Ingest new data -> fires new_data -> refresh_stale re-resolves the snapshot.
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "price": [7, 8, 9]})
    ingest_dataframe(p, df, source_name="new.csv", auto_profile=False)

    row = board.state.get_task(p.root / "state.db", task.id)
    assert row["context_fingerprint"] != old_fp  # snapshot was refreshed on trigger


# --- Brief render (C5) -------------------------------------------------------

def test_render_brief(temp_projects_dir):
    from core import board, context

    p = _project()
    board.add_task(p, "Train the model", description="Fit and validate.", stage="model")
    task = board.list_tasks(p)[0]
    brief = context.render_brief(context.assemble_context(p, task))
    assert "# Task brief — Train the model" in brief
    assert "run_stage:model" in brief
    assert "Expected outputs" in brief


# --- CLI ---------------------------------------------------------------------

def test_cli_board_context(temp_projects_dir, capsys):
    from core import board
    from mira_cli import main

    p = _project()
    board.add_task(p, "Train", stage="model")
    rc = main(["board", "context", "--project-id", p.id, "--task", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_stage:model" in out
    assert "fresh" in out  # freshly assigned on first read

    rc = main(["board", "context", "--project-id", p.id, "--task", "1", "--brief"])
    assert rc == 0
    assert "# Task brief" in capsys.readouterr().out
