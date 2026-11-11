"""
tests/test_ci.py

Phase 4: CI gate engine + gated runner + in-process triggers. The pipeline is
faked so tests are fast and deterministic (no real stages/data).
Run with: pytest tests/test_ci.py -q
"""

import os
import sqlite3
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
    return Project.create(name="CI Test")


class _FakePipeline:
    """Stand-in for core.pipeline.Pipeline with scripted stage summaries."""

    summaries: dict[str, dict] = {}

    def __init__(self, project):
        self.project = project

    def run_stage(self, name, settings=None):
        from core.pipeline import StageResult
        return StageResult(status="success", summary=self.summaries.get(name, {}))


# --- Gate engine (pure) ------------------------------------------------------

def test_evaluate_gate_pass_fail_skip():
    from core.gates import evaluate_gate

    gate = {"stage": "model", "metric": "r2", "op": ">=", "threshold": 0.5}
    assert evaluate_gate("model", {"r2": 0.8}, gate).status == "pass"
    assert evaluate_gate("model", {"r2": 0.2}, gate).status == "fail"
    # missing metric -> skip (non-blocking)
    skip = evaluate_gate("model", {"accuracy": 0.9}, gate)
    assert skip.status == "skip" and not skip.blocking


def test_evaluate_gate_non_numeric_and_bad_op():
    from core.gates import evaluate_gate

    assert evaluate_gate("model", {"r2": "n/a"},
                         {"stage": "model", "metric": "r2", "op": ">=",
                          "threshold": 0.5}).status == "skip"
    assert evaluate_gate("model", {"r2": 0.5},
                         {"stage": "model", "metric": "r2", "op": "≈",
                          "threshold": 0.5}).status == "skip"


def test_evaluate_gates_filters_by_stage():
    from core.gates import evaluate_gates

    gates = [
        {"stage": "model", "metric": "r2", "op": ">=", "threshold": 0.5},
        {"stage": "evaluate", "metric": "rmse", "op": "<=", "threshold": 1.0},
    ]
    res = evaluate_gates("model", {"r2": 0.6}, gates)
    assert len(res) == 1 and res[0].stage == "model"


# --- Gated runner ------------------------------------------------------------

def test_run_ci_blocks_deploy_on_failed_model_gate(temp_projects_dir, monkeypatch):
    import core.ci as ci

    p = _project()
    p.meta["ci"] = {
        "gates": [{"stage": "model", "metric": "r2", "op": ">=", "threshold": 0.5}],
        "triggers": {"task_done": False, "new_data": False},
    }
    p.save_meta()

    _FakePipeline.summaries = {"model": {"r2": 0.2}}  # fails the gate
    monkeypatch.setattr(ci, "Pipeline", _FakePipeline)

    result = ci.run_ci(p)
    assert result.status == "failed"
    assert result.failed_gate.stage == "model"
    assert "model" in result.stages_run
    assert "deploy" not in result.stages_run  # blocked before deploy

    # the failure was logged to gate_runs
    rows = sqlite3.connect(p.root / "state.db").execute(
        "SELECT status FROM gate_runs WHERE stage='model'").fetchall()
    assert ("fail",) in rows


def test_run_ci_passes_through_deploy_when_gate_met(temp_projects_dir, monkeypatch):
    import core.ci as ci

    p = _project()
    p.meta["ci"] = {
        "gates": [{"stage": "model", "metric": "r2", "op": ">=", "threshold": 0.5}],
        "triggers": {"task_done": False, "new_data": False},
    }
    p.save_meta()

    _FakePipeline.summaries = {"model": {"r2": 0.9}}
    monkeypatch.setattr(ci, "Pipeline", _FakePipeline)

    result = ci.run_ci(p)
    assert result.status == "passed"
    assert "deploy" in result.stages_run


# --- Triggers ----------------------------------------------------------------

def test_task_done_logs_trigger_event_disabled_by_default(temp_projects_dir):
    from core import board

    p = _project()
    tid = board.add_task(p, "Trigger me")
    board.move_task(p, tid, "done")

    rows = sqlite3.connect(p.root / "state.db").execute(
        "SELECT event, action, result FROM trigger_events").fetchall()
    assert ("task_done", "noop", "skipped") in rows


def test_enabled_trigger_fires_ci(temp_projects_dir, monkeypatch):
    import core.ci as ci
    from core import board
    from core.ci import CIResult

    p = _project()
    ci.ci_config(p)
    p.meta["ci"]["triggers"]["task_done"] = True
    p.save_meta()

    calls = []
    monkeypatch.setattr(ci, "run_ci",
                        lambda project, **kw: calls.append(project.id) or CIResult(status="passed"))

    tid = board.add_task(p, "Auto-run on done")
    board.move_task(p, tid, "done")

    assert calls == [p.id]
    rows = sqlite3.connect(p.root / "state.db").execute(
        "SELECT result FROM trigger_events WHERE event='task_done'").fetchall()
    assert ("passed",) in rows
