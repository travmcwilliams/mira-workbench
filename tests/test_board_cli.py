"""
tests/test_board_cli.py

Phase 2: `mira board` CLI subcommands. Offline/deterministic.
Run with: pytest tests/test_board_cli.py -q
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


def _project_id():
    from core.project import Project
    return Project.create(name="CLI Board").id


def test_board_add_and_list(temp_projects_dir, capsys):
    from mira_cli import main

    pid = _project_id()
    rc = main(["board", "add", "--project-id", pid, "--title", "Ship feature",
               "--points", "5", "--priority", "high"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Added task #1" in out

    rc = main(["board", "list", "--project-id", pid])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Ship feature" in out
    assert "5pt" in out
    # stage cards present by default
    assert "Deploy" in out


def test_board_list_tasks_only_hides_stage_cards(temp_projects_dir, capsys):
    from mira_cli import main

    pid = _project_id()
    main(["board", "add", "--project-id", pid, "--title", "Only me"])
    capsys.readouterr()
    main(["board", "list", "--project-id", pid, "--tasks-only"])
    out = capsys.readouterr().out
    assert "Only me" in out
    assert "Deploy" not in out  # no derived stage cards


def test_board_move(temp_projects_dir, capsys):
    from core import board
    from core.project import Project
    from mira_cli import main

    pid = _project_id()
    main(["board", "add", "--project-id", pid, "--title", "Move me"])
    capsys.readouterr()

    rc = main(["board", "move", "--project-id", pid, "--task", "1", "--to", "done"])
    assert rc == 0
    assert "Moved task #1 -> done" in capsys.readouterr().out

    p = Project.load(pid)
    assert board.list_tasks(p)[0].column == "done"


def test_board_move_unknown_task_errors(temp_projects_dir):
    from mira_cli import main

    pid = _project_id()
    rc = main(["board", "move", "--project-id", pid, "--task", "999", "--to", "done"])
    assert rc == 1
