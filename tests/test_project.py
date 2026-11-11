"""
tests/test_project.py

Basic tests for project create / load / list / persistence.
Run with: pytest tests/test_project.py -q
"""

# We temporarily change working dir so Project uses a temp projects/ root
import os
from pathlib import Path

import pytest


@pytest.fixture
def temp_projects_dir(monkeypatch, tmp_path):
    """Isolate Project to a temporary projects/ directory for the test."""
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    # conftest.py pins MIRA_PROJECTS_ROOT="projects" (cwd-relative legacy mode),
    # so chdir(tmp_path) isolates this test's projects/ from the real repo.
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def test_create_and_load_project(temp_projects_dir):
    from core.project import Project

    p = Project.create(name="Test Project", task_type="classification")
    assert p.id
    assert p.meta["name"] == "Test Project"
    assert p.meta["task_type"] == "classification"
    assert (p.root / "meta.json").exists()
    assert (p.root / "state.db").exists()

    # Load by id
    loaded = Project.load(p.id)
    assert loaded.meta["name"] == "Test Project"
    assert loaded.id == p.id


def test_list_all_projects(temp_projects_dir):
    from core.project import Project

    Project.create(name="Alpha", task_type="regression")
    Project.create(name="Beta", task_type="classification")

    all_projects = Project.list_all()
    names = [pr.meta["name"] for pr in all_projects]
    assert "Alpha" in names
    assert "Beta" in names


def test_project_persists_across_loads(temp_projects_dir):
    from core.project import Project

    p = Project.create(name="Persist Test")
    pid = p.id

    # Simulate "restart" by loading fresh
    p2 = Project.load(pid)
    assert p2.meta["name"] == "Persist Test"
    assert p2.meta["id"] == pid


def test_projects_root_env_override(monkeypatch, tmp_path):
    from core.project import projects_root

    monkeypatch.setenv("MIRA_PROJECTS_ROOT", str(tmp_path / "elsewhere"))
    assert projects_root() == tmp_path / "elsewhere"


def test_projects_root_default_is_repo_anchored(monkeypatch):
    from core import project as project_mod

    monkeypatch.delenv("MIRA_PROJECTS_ROOT", raising=False)
    root = project_mod.projects_root()
    assert root.is_absolute()
    assert root == project_mod._REPO_ROOT / "projects"


def test_list_project_index_fields_and_order(temp_projects_dir):
    import time

    from core.project import Project, list_project_index

    a = Project.create(name="Alpha")
    time.sleep(0.05)  # distinct dir mtimes for ordering
    b = Project.create(name="Beta")

    index = list_project_index()
    assert [e["name"] for e in index] == ["Beta", "Alpha"]
    entry = index[0]
    assert entry["id"] == b.id
    assert entry["completed_stages"] == []
    assert entry["task_type"] == "regression"
    assert entry["updated_at"]
    assert a.id in {e["id"] for e in index}
