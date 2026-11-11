"""
tests/test_remote_ci.py

Phase 6: remote-CI status via the gh CLI. The gh call is isolated in
`_gh_run_list_json`, monkeypatched here so tests need no network or auth.
Run with: pytest tests/test_remote_ci.py -q
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

GIT = shutil.which("git")

_FAKE_GH_JSON = """
[
  {"databaseId": 101, "workflowName": "CI", "status": "completed",
   "conclusion": "success", "headBranch": "main", "displayTitle": "fix bug",
   "createdAt": "2026-06-26T10:00:00Z", "url": "https://gh/run/101", "event": "push"},
  {"databaseId": 100, "workflowName": "CI", "status": "completed",
   "conclusion": "failure", "headBranch": "main", "displayTitle": "wip",
   "createdAt": "2026-06-26T09:00:00Z", "url": "https://gh/run/100", "event": "push"}
]
"""


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
    return Project.create(name="Remote CI Test")


def _init_repo(repo: Path):
    subprocess.run([GIT, "init", "-b", "main"], cwd=repo, check=True,
                   capture_output=True)


def test_unavailable_when_not_a_repo(temp_projects_dir, monkeypatch):
    from core import git_sync

    monkeypatch.setattr(git_sync, "gh_available", lambda: True)
    p = _project()  # project root is not a git repo
    status = git_sync.remote_ci_status(p)
    assert status["available"] is False
    assert status["runs"] == []
    assert status["error"]


def test_unavailable_when_gh_missing(temp_projects_dir, monkeypatch):
    from core import git_sync

    monkeypatch.setattr(git_sync, "gh_available", lambda: False)
    p = _project()
    status = git_sync.remote_ci_status(p)
    assert status["available"] is False
    assert "gh CLI not installed" in status["error"]


@pytest.mark.skipif(GIT is None, reason="git not installed")
def test_parses_gh_run_list(temp_projects_dir, monkeypatch):
    from core import git_sync

    p = _project()
    _init_repo(p.root)
    git_sync.git_status(p)  # persists repo_path into meta

    monkeypatch.setattr(git_sync, "gh_available", lambda: True)
    monkeypatch.setattr(git_sync, "_gh_run_list_json",
                        lambda repo, limit: _FAKE_GH_JSON)

    status = git_sync.remote_ci_status(p)
    assert status["available"] is True
    assert len(status["runs"]) == 2
    first = status["runs"][0]
    assert first["workflow"] == "CI"
    assert first["conclusion"] == "success"
    assert first["url"] == "https://gh/run/101"


@pytest.mark.skipif(GIT is None, reason="git not installed")
def test_gh_error_degrades_gracefully(temp_projects_dir, monkeypatch):
    from core import git_sync

    p = _project()
    _init_repo(p.root)
    git_sync.git_status(p)

    monkeypatch.setattr(git_sync, "gh_available", lambda: True)

    def _boom(repo, limit):
        raise RuntimeError("gh: no default remote repository")

    monkeypatch.setattr(git_sync, "_gh_run_list_json", _boom)
    status = git_sync.remote_ci_status(p)
    assert status["available"] is False
    assert "no default remote" in status["error"]


def test_ci_status_line_and_icon():
    from core.git_sync import ci_run_icon, ci_status_line

    ok = {"status": "completed", "conclusion": "success", "workflow": "CI",
          "branch": "main"}
    assert ci_run_icon(ok) == "🟢"
    running = {"status": "in_progress", "conclusion": None}
    assert ci_run_icon(running) == "🟡"

    line = ci_status_line({"available": True, "runs": [ok]})
    assert "success" in line and "CI" in line
    assert ci_status_line({"available": False, "error": "no gh"}) == "CI: no gh"
    assert ci_status_line({"available": True, "runs": []}) == "CI: no runs found"


@pytest.mark.skipif(GIT is None, reason="git not installed")
def test_cli_git_ci_status(temp_projects_dir, monkeypatch, capsys):
    from core import git_sync
    from mira_cli import main

    p = _project()
    _init_repo(p.root)
    git_sync.git_status(p)
    monkeypatch.setattr(git_sync, "gh_available", lambda: True)
    monkeypatch.setattr(git_sync, "_gh_run_list_json",
                        lambda repo, limit: _FAKE_GH_JSON)

    rc = main(["git", "ci-status", "--project-id", p.id])
    assert rc == 0
    out = capsys.readouterr().out
    assert "success" in out and "failure" in out
