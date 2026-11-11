"""
tests/test_executor.py

Phase 8: executor runtime. The pipeline/CI/LLM are faked so tests are fast and
fully offline. Verifies skill dispatch, lineage gating, card advancement, the
agent hand-off package, and execution logging.
Run with: pytest tests/test_executor.py -q
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
    p = Project.create(name="Exec Test", task_type="regression")
    p.meta["target_column"] = "price"
    p.meta["schema"] = {"a": "num", "price": "num"}
    p.save_meta()
    return p


class _FakePipeline:
    result_status = "success"
    summary = {"best_model": "ridge"}

    def __init__(self, project):
        self.project = project

    def run_stage(self, name, settings=None):
        from core.pipeline import StageResult
        if self.result_status == "success":
            return StageResult(status="success", summary=self.summary)
        return StageResult(status="error", error="boom")


def test_stage_blocked_when_lineage_missing(temp_projects_dir):
    from core import board, executor

    p = _project()
    # 'model' consumes data_prep.output, which is not available.
    board.add_task(p, "Train", stage="model")
    task = board.list_tasks(p)[0]
    res = executor.execute_task(p, task)
    assert res.status == "blocked"
    assert "data_prep.output" in res.output


def test_stage_runs_and_advances_card(temp_projects_dir, monkeypatch):
    import core.executor as executor
    from core import board

    p = _project()
    # Make the model task's lineage available.
    p.meta["completed_stages"] = ["data_prep"]
    p.save_meta()
    board.add_task(p, "Train", stage="model")
    task = board.list_tasks(p)[0]

    monkeypatch.setattr("core.pipeline.Pipeline", _FakePipeline)
    _FakePipeline.result_status = "success"
    res = executor.execute_task(p, task)

    assert res.status == "success"
    assert res.summary.get("best_model") == "ridge"
    # card advanced to done
    assert board.list_tasks(p)[0].column == "done"


def test_stage_failure_reported(temp_projects_dir, monkeypatch):
    import core.executor as executor
    from core import board

    p = _project()
    p.meta["completed_stages"] = ["data_prep"]
    p.save_meta()
    board.add_task(p, "Train", stage="model")
    task = board.list_tasks(p)[0]

    monkeypatch.setattr("core.pipeline.Pipeline", _FakePipeline)
    _FakePipeline.result_status = "error"
    res = executor.execute_task(p, task)
    assert res.status == "failed"
    assert res.error


def test_dry_run_does_not_execute(temp_projects_dir, monkeypatch):
    from core import board, executor

    p = _project()
    board.add_task(p, "Train", stage="model")  # lineage missing, but dry-run skips gate
    task = board.list_tasks(p)[0]

    def _boom(project):
        raise AssertionError("pipeline ran during dry-run")

    monkeypatch.setattr("core.pipeline.Pipeline", _boom)
    res = executor.execute_task(p, task, dry_run=True)
    assert res.status == "ready"
    assert "Would run stage 'model'" in res.output


def test_agent_offline_returns_brief_handoff(temp_projects_dir):
    from core import board, executor

    p = _project()
    board.add_task(p, "Refactor ingestion", description="clean it up")
    task = board.list_tasks(p)[0]
    res = executor.execute_task(p, task, provider="offline")
    assert res.status == "ready"
    assert res.kind == "agent"
    assert "# Task brief" in res.output  # hand-off package is the brief


def test_agent_provider_calls_llm(temp_projects_dir, monkeypatch):
    from core import board, executor, llm

    p = _project()
    board.add_task(p, "Write code")
    task = board.list_tasks(p)[0]
    monkeypatch.setattr(llm, "generate", lambda prompt, **kw: "PLAN: do the thing")
    res = executor.execute_task(p, task, provider="ollama")
    assert res.status == "success"
    assert res.output == "PLAN: do the thing"


def test_agent_llm_failure_degrades_to_handoff(temp_projects_dir, monkeypatch):
    from core import board, executor, llm

    p = _project()
    board.add_task(p, "Write code")
    task = board.list_tasks(p)[0]

    def _boom(prompt, **kw):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(llm, "generate", _boom)
    res = executor.execute_task(p, task, provider="ollama")
    assert res.status == "ready"
    assert "# Task brief" in res.output


def test_execution_is_logged(temp_projects_dir):
    from core import board, executor, state

    p = _project()
    board.add_task(p, "Refactor ingestion")
    task = board.list_tasks(p)[0]
    executor.execute_task(p, task, provider="offline")

    rows = state.get_executions(p.root / "state.db", task_id=task.id)
    assert len(rows) == 1
    assert rows[0]["skill"] == "llm_agent"
    assert rows[0]["status"] == "ready"


def test_cli_board_run_dry(temp_projects_dir, capsys):
    from core import board
    from mira_cli import main

    p = _project()
    board.add_task(p, "Train", stage="model")
    rc = main(["board", "run", "--project-id", p.id, "--task", "1", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_stage:model" in out
    assert "READY" in out
