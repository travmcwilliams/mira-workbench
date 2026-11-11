"""Suite-wide fixtures.

``core.project`` anchors the projects root at the repo root by default so the
GUI/CLI work from any cwd. The existing suite isolates projects by
``chdir(tmp_path)`` (~17 files) and relies on cwd-relative resolution — this
autouse fixture pins the legacy behavior for every test: a RELATIVE
``MIRA_PROJECTS_ROOT`` stays cwd-relative, so ``chdir`` isolation keeps working
with zero per-file changes.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _legacy_cwd_projects_root(monkeypatch):
    monkeypatch.setenv("MIRA_PROJECTS_ROOT", "projects")
