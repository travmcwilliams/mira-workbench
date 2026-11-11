"""ui/home_data.py — pure Home-landing summary."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.home_data import get_home_summary  # noqa: E402


@pytest.fixture
def temp_projects_dir(tmp_path):
    (tmp_path / "projects").mkdir()
    original = Path.cwd()
    os.chdir(tmp_path)  # conftest pins MIRA_PROJECTS_ROOT="projects" (cwd-relative)
    try:
        yield
    finally:
        os.chdir(original)


def test_empty_root(temp_projects_dir):
    s = get_home_summary()
    assert s["n_projects"] == 0
    assert s["latest"] is None
    assert s["recent"] == []


def test_latest_is_most_recent(temp_projects_dir):
    from core.project import Project

    Project.create(name="Older")
    time.sleep(0.05)
    newer = Project.create(name="Newer")

    s = get_home_summary()
    assert s["n_projects"] == 2
    assert s["latest"]["id"] == newer.id
    assert s["latest"]["name"] == "Newer"
    assert s["latest"]["n_stages_done"] == 0
    assert len(s["recent"]) == 2
