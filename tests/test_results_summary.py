"""ui/results_summary.py — pure assembly from a project folder; never raises."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.results_summary import assemble_results  # noqa: E402


def _fake_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "artifacts" / "eval").mkdir(parents=True)
    (root / "meta.json").write_text(json.dumps({
        "id": "abc123", "name": "Fake", "task_type": "regression",
        "target_column": "y", "completed_stages": ["profile", "model", "evaluate"],
    }), encoding="utf-8")
    (root / "artifacts" / "eval" / "metrics.json").write_text(json.dumps({
        "mae": 0.005, "rmse": 0.007, "r2": -0.004,
        "leaderboard_top": {"model": "svm"},
    }), encoding="utf-8")
    (root / "artifacts" / "leaderboard.csv").write_text(
        "model,score\nsvm,0.9\nridge,0.8\n", encoding="utf-8")
    (root / "artifacts" / "eval" / "residuals.png").write_bytes(b"\x89PNG fake")
    return root


def test_assemble_full_project(tmp_path):
    r = assemble_results(_fake_project(tmp_path))
    assert r["completed_stages"] == ["profile", "model", "evaluate"]
    assert r["metrics"]["r2"] == -0.004
    assert [row["model"] for row in r["leaderboard"]] == ["svm", "ridge"]
    # no state.db -> best model falls back to leaderboard row 0
    assert r["best_model"] == "svm"
    assert len(r["eval_images"]) == 1
    assert r["history"] == []  # no state.db


def test_assemble_bare_dir_never_raises(tmp_path):
    r = assemble_results(tmp_path / "empty")
    assert r["meta"] == {}
    assert r["metrics"] is None
    assert r["leaderboard"] is None
    assert r["best_model"] is None
    assert r["eval_images"] == []
    assert r["completed_stages"] == []


def test_assemble_partial_project(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "meta.json").write_text(json.dumps({
        "id": "x", "name": "Partial", "completed_stages": ["profile"],
    }), encoding="utf-8")
    r = assemble_results(root)
    assert r["completed_stages"] == ["profile"]
    assert r["metrics"] is None and r["best_model"] is None
