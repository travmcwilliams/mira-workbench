"""
stages/feature_select.py

Feature selection: correlation, variance, RFE, model importance.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.pipeline import StageResult
from core.project import Project


def default_settings(project: Project) -> dict[str, Any]:
    return {
        "methods": ["variance", "correlation", "mutual_info"],
        "correlation_threshold": 0.9,
        "variance_threshold": 0.01,
        "rfe_n_features": 10,
        "importance_top_k": 20,
        "importance_model": "random_forest",
        "mi_top_k": 20,
        "permutation_top_k": 20,
    }


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    if df is None or len(df) == 0:
        return StageResult(status="error", error="No data for feature selection.")

    target = project.meta.get("target_column")
    if not target or target not in df.columns:
        return StageResult(status="error", error="Target column required for feature selection.")

    X = df.drop(columns=[target])
    y = df[target]

    non_numeric = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    coerced: list[str] = []
    for col in list(non_numeric):
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            continue
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() >= 0.95:
            X[col] = converted
            coerced.append(col)
    remaining_non_numeric = [
        col
        for col in X.columns
        if not pd.api.types.is_numeric_dtype(X[col])
        or pd.api.types.is_datetime64_any_dtype(X[col])
    ]
    if remaining_non_numeric:
        X = X.drop(columns=remaining_non_numeric)

    if X.shape[1] == 0:
        return StageResult(
            status="error",
            error=(
                "No numeric feature columns remain after feature engineering. "
                "Ensure categorical columns are encoded and id/text columns are dropped."
            ),
        )

    original_cols = list(X.columns)

    methods = settings.get("methods", [])
    selected = set(original_cols)

    import numpy as np

    # Correlation filter
    if "correlation" in methods:
        thresh = settings.get("correlation_threshold", 0.9)
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > thresh)]
        selected -= set(to_drop)

    # Variance threshold
    if "variance" in methods:
        from sklearn.feature_selection import VarianceThreshold

        thresh = settings.get("variance_threshold", 0.01)
        try:
            selector = VarianceThreshold(threshold=thresh)
            selector.fit(X)
            kept = X.columns[selector.get_support()].tolist()
            selected &= set(kept)
        except Exception:
            pass

    # RFE (simplified, uses linear model)
    if "rfe" in methods and len(selected) > settings.get("rfe_n_features", 10):
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression, Ridge

        n = min(settings.get("rfe_n_features", 10), len(selected))
        model = (
            LogisticRegression(max_iter=1000)
            if project.meta.get("task_type") == "classification"
            else Ridge()
        )
        try:
            rfe = RFE(model, n_features_to_select=n)
            rfe.fit(X[list(selected)], y)
            kept = [c for c, s in zip(selected, rfe.support_, strict=False) if s]
            selected = set(kept)
        except Exception:
            pass

    # Importance (RandomForest or similar)
    if "importance" in methods:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        top_k = settings.get("importance_top_k", 20)
        model = (
            RandomForestClassifier
            if project.meta.get("task_type") == "classification"
            else RandomForestRegressor
        )(n_estimators=50, random_state=42)
        try:
            model.fit(X[list(selected)], y)
            importances = pd.Series(model.feature_importances_, index=list(selected)).sort_values(
                ascending=False
            )
            kept = importances.head(top_k).index.tolist()
            selected = set(kept)
        except Exception:
            pass

    # Mutual information (sklearn mutual_info_classif / mutual_info_regression)
    if "mutual_info" in methods:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        mi_func = (
            mutual_info_classif
            if project.meta.get("task_type") == "classification"
            else mutual_info_regression
        )
        try:
            cols = list(selected)
            scores = mi_func(X[cols], y, random_state=42)
            mi = pd.Series(scores, index=cols).sort_values(ascending=False)
            mi_threshold = settings.get("mi_threshold")
            if mi_threshold is not None:
                kept = mi[mi >= mi_threshold].index.tolist()
            else:
                kept = mi.head(settings.get("mi_top_k", 20)).index.tolist()
            selected = set(kept)
        except Exception:
            pass

    # Permutation importance on a fitted RandomForest
    if "permutation" in methods:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.inspection import permutation_importance

        top_k = settings.get("permutation_top_k", 20)
        model = (
            RandomForestClassifier
            if project.meta.get("task_type") == "classification"
            else RandomForestRegressor
        )(n_estimators=50, random_state=42)
        try:
            cols = list(selected)
            model.fit(X[cols], y)
            result = permutation_importance(
                model, X[cols], y, n_repeats=5, random_state=42
            )
            importances = pd.Series(result.importances_mean, index=cols).sort_values(
                ascending=False
            )
            kept = importances.head(top_k).index.tolist()
            selected = set(kept)
        except Exception:
            pass

    final_cols = [target] + list(selected)
    df_selected = df[final_cols]

    summary = {
        "original_features": len(original_cols),
        "selected_features": len(selected),
        "dropped": len(original_cols) - len(selected),
        "selected_columns": list(selected)[:15],  # truncate for summary
        "coerced_numeric": coerced,
        "dropped_non_numeric": remaining_non_numeric,
    }

    return StageResult(
        status="success",
        output_df=df_selected,
        artifacts=[],
        summary=summary,
    )


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    return settings


def code_template(settings: dict[str, Any]) -> str:
    return f"""# Feature Selection
# methods = {settings.get("methods", [])}
print("Feature selection applied. Kept top features.")
"""
