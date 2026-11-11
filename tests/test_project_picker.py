"""ui/project_picker.py pure helpers + the lightweight project index."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.project_picker import resolve_initial_pid  # noqa: E402


def test_resolve_initial_pid_only_honors_known_ids():
    ids = ["abc", "def"]
    assert resolve_initial_pid("abc", ids) == "abc"
    assert resolve_initial_pid("zzz", ids) is None
    assert resolve_initial_pid(None, ids) is None
    assert resolve_initial_pid("", ids) is None
    assert resolve_initial_pid("abc", []) is None
