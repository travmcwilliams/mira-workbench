"""
tests/test_git_sync.py

Phase 3: read-only git sync. Uses a real throwaway git repo when git is
available; otherwise asserts graceful degradation.
Run with: pytest tests/test_git_sync.py -q
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

GIT = shutil.which("git")


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
    return Project.create(name="Git Test")


def test_status_when_not_a_repo(temp_projects_dir):
    from core.git_sync import git_status

    p = _project()
    status = git_status(p)
    assert status["is_repo"] is False
    assert status["branch"] is None
    # never raises, always returns the full dict shape
    assert set(status) >= {"is_repo", "repo_path", "branch", "dirty", "error"}


def test_git_config_defaults_to_project_root(temp_projects_dir):
    from core.git_sync import git_config

    p = _project()
    cfg = git_config(p)
    assert Path(cfg["repo_path"]).name == p.id


@pytest.mark.skipif(GIT is None, reason="git not installed")
def test_status_on_real_repo(temp_projects_dir):
    from core.git_sync import git_status, status_line

    p = _project()
    repo = p.root
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run([GIT, "init", "-b", "main"], cwd=repo, check=True,
                   capture_output=True)
    (repo / "README.md").write_text("hi")
    # First status call persists meta["git"]["repo_path"] into meta.json; do it
    # before committing so the commit captures a clean tree.
    git_status(p)
    subprocess.run([GIT, "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run([GIT, "commit", "-m", "init commit"], cwd=repo, check=True,
                   capture_output=True, env=env)

    status = git_status(p)
    assert status["is_repo"] is True
    assert status["branch"] == "main"
    assert status["dirty"] is False
    assert status["last_commit"]["subject"] == "init commit"

    # dirty after an uncommitted change
    (repo / "README.md").write_text("changed")
    assert git_status(p)["dirty"] is True
    assert "branch main" in status_line(git_status(p))


def test_cli_git_status(temp_projects_dir, capsys):
    from mira_cli import main

    pid = _project().id
    rc = main(["git", "status", "--project-id", pid])
    assert rc == 0
    # Output is the one-line summary; for a fresh non-repo it says so.
    out = capsys.readouterr().out
    assert "git" not in out.lower() or "no git repo" in out or "branch" in out
