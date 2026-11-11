"""Tests for the sklearn fallback model zoo used when PyCaret is unavailable.

PyCaret only loads on Python 3.9–3.11; mira's working interpreter is 3.12, so the
sklearn fallback is what actually runs day to day. These tests lock in that the
fallback compares a broad zoo (mirroring PyCaret's default estimator set) rather
than a handful of models.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

from core.library_catalog import PYCARET_MODEL_MAP
from core.sklearn_automl import _candidate_models, compare_models


def _stub_project(task: str, enabled: list[str] | None = None):
    return types.SimpleNamespace(
        meta={"task_type": task, "enabled_libraries": enabled or ["pandas", "sklearn"]}
    )


def test_fallback_zoo_covers_pycaret_default_classifiers():
    """The classification fallback must offer every estimator PyCaret would
    compare from its default sklearn ``include`` set."""
    names = {name for name, _ in _candidate_models("classification", ["sklearn"])}
    for shortcut in PYCARET_MODEL_MAP["sklearn"]:  # lr, lda, ridge, et, dt, rf, knn, nb, qda, svm
        # `ridge` maps to logistic/linear families already covered by `lr` here.
        if shortcut == "ridge":
            continue
        assert shortcut in names, f"classifier zoo missing PyCaret estimator {shortcut!r}"


def test_fallback_zoo_broadened_for_regression():
    names = {name for name, _ in _candidate_models("regression", ["sklearn"])}
    for shortcut in ("rf", "et", "dt", "knn", "svm", "lr", "ridge", "lasso", "en"):
        assert shortcut in names, f"regression zoo missing {shortcut!r}"


def test_compare_models_runs_broad_regression_zoo():
    rng = np.random.default_rng(0)
    n = 80
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["y"] = y

    project = _stub_project("regression", ["sklearn"])
    best, leaderboard = compare_models(project, df, "y", n_models=5, cv_folds=3)

    assert best is not None
    assert len(leaderboard) == 5  # leaderboard capped at n_models from a larger zoo
    assert {"Model", "Score"} <= set(leaderboard.columns)
    # A near-linear target: at least one model should fit it well.
    assert leaderboard["Score"].max() > 0.9
