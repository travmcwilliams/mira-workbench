"""Lightweight sklearn AutoML used when PyCaret is unavailable."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from core.library_catalog import get_enabled_libraries


def _scaled(model: Any) -> Pipeline:
    """Wrap a scale-sensitive estimator (linear / distance / kernel) so it never
    trains on unscaled features when it reaches this fallback directly."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def _optional_boosters(task: str, enabled: list[str]) -> list[tuple[str, Any]]:
    """Gradient-boosting libraries (xgboost/lightgbm/catboost) are OPTIONAL deps. Add each
    only when it is both enabled and importable â€” the default zoo advertises them, but they
    are not guaranteed installed (e.g. CI), so skip gracefully instead of crashing the run."""
    clf = task == "classification"
    out: list[tuple[str, Any]] = []
    if "xgboost" in enabled:
        try:
            from xgboost import XGBClassifier, XGBRegressor

            out.append(("xgboost", XGBClassifier(random_state=42, eval_metric="logloss")
                        if clf else XGBRegressor(random_state=42)))
        except ImportError:
            pass
    if "lightgbm" in enabled:
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor

            out.append(("lightgbm", LGBMClassifier(random_state=42, verbose=-1)
                        if clf else LGBMRegressor(random_state=42, verbose=-1)))
        except ImportError:
            pass
    if "catboost" in enabled:
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            out.append(("catboost", CatBoostClassifier(random_state=42, verbose=0)
                        if clf else CatBoostRegressor(random_state=42, verbose=0)))
        except ImportError:
            pass
    return out


def _candidate_models(task: str, enabled: list[str]) -> list[tuple[str, Any]]:
    """The sklearn fallback model zoo. Mirrors PyCaret's default estimator set
    (lr, lda, ridge, et, dt, rf, knn, nb, qda, svm) so a run on a Python 3.12
    interpreter â€” where PyCaret cannot load â€” still compares a broad zoo, then
    layers on the gradient-boosting libraries when they are enabled."""
    models: list[tuple[str, Any]] = []
    if task == "classification":
        models.extend(
            [
                ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
                ("et", ExtraTreesClassifier(n_estimators=100, random_state=42)),
                ("gbc", GradientBoostingClassifier(random_state=42)),
                ("ada", AdaBoostClassifier(random_state=42)),
                ("dt", DecisionTreeClassifier(random_state=42)),
                ("lr", _scaled(LogisticRegression(max_iter=1000))),
                ("lda", LinearDiscriminantAnalysis()),
                ("qda", QuadraticDiscriminantAnalysis()),
                ("nb", GaussianNB()),
                ("knn", _scaled(KNeighborsClassifier())),
                ("svm", _scaled(SVC())),
            ]
        )
    else:
        models.extend(
            [
                ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
                ("et", ExtraTreesRegressor(n_estimators=100, random_state=42)),
                ("gbr", GradientBoostingRegressor(random_state=42)),
                ("ada", AdaBoostRegressor(random_state=42)),
                ("dt", DecisionTreeRegressor(random_state=42)),
                ("lr", _scaled(LinearRegression())),
                ("ridge", _scaled(Ridge())),
                ("lasso", _scaled(Lasso())),
                ("en", _scaled(ElasticNet())),
                ("knn", _scaled(KNeighborsRegressor())),
                ("svm", _scaled(SVR())),
            ]
        )
    models.extend(_optional_boosters(task, enabled))
    return models


def compare_models(
    project: Any,
    df: pd.DataFrame,
    target: str,
    *,
    n_models: int = 5,
    cv_folds: int = 5,
) -> tuple[Any, pd.DataFrame]:
    task = project.meta.get("task_type", "regression")
    enabled = get_enabled_libraries(project)
    X = df.drop(columns=[target])
    y = df[target]

    if task == "classification":
        scoring = {"score": "accuracy", "f1": "f1_weighted"}
    else:
        scoring = {"score": "r2", "mae": "neg_mean_absolute_error"}

    rows: list[dict[str, Any]] = []
    best_model: Any | None = None
    best_score = float("-inf")

    for name, estimator in _candidate_models(task, enabled):
        try:
            cv = cross_validate(
                clone(estimator),
                X,
                y,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
            )
            score = float(cv["test_score"].mean())
            row: dict[str, Any] = {"Model": name, "Score": score}
            for metric, values in cv.items():
                if metric.startswith("test_") and metric != "test_score":
                    row[metric.removeprefix("test_")] = float(values.mean())
            rows.append(row)
            if score > best_score:
                best_score = score
                best_model = clone(estimator).fit(X, y)
        except Exception:
            continue

    if best_model is None or not rows:
        raise RuntimeError("No sklearn models completed successfully.")

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("Score", ascending=False)
        .head(n_models)
        .reset_index(drop=True)
    )
    leaderboard.index = leaderboard["Model"]
    return best_model, leaderboard
