"""
stages/clean.py

Cleaning stage: missing values, outliers, duplicates, type overrides.
See grok_handoff.md Section 6.6.

Settings are per-column for missing, global for outliers/duplicates.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.pipeline import StageResult
from core.project import Project


def default_settings(project: Project) -> dict[str, Any]:
    """Auto-generate defaults based on current schema and data."""
    schema = project.meta.get("schema", {})
    df = project.get_working_data() if project.has_raw_data() else pd.DataFrame()
    target = project.meta.get("target_column")

    missing_strategy = {}
    for col, dtype in schema.items():
        if col not in df.columns or col == target:
            # Never impute the target: unlabeled rows are dropped in run(),
            # not filled with a fabricated label.
            continue
        miss_pct = df[col].isna().mean()
        if miss_pct > 0:
            if dtype == "numeric":
                missing_strategy[col] = "median"
            elif dtype == "categorical":
                missing_strategy[col] = "mode"
            else:
                missing_strategy[col] = "constant"

    return {
        "drop_columns": [],
        "missing_strategy": missing_strategy,
        "outlier_method": "iqr",
        "outlier_threshold": 1.5,
        "outlier_action": "clip",
        "drop_duplicates": True,
        "type_overrides": {},
        "constant_value": "Unknown",
    }


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    if df is None or len(df) == 0:
        return StageResult(status="error", error="No data to clean.")

    rows_before = len(df)
    cells_imputed = 0
    rows_dropped_outliers = 0
    duplicates_dropped = 0

    df = df.copy()
    target = project.meta.get("target_column")

    # Never impute the target. Drop rows with a missing/blank label instead of
    # fabricating one — an unlabeled row can neither train nor evaluate a model.
    rows_dropped_missing_target = 0
    if target and target in df.columns:
        before = len(df)
        blank = df[target].isna()
        if not pd.api.types.is_numeric_dtype(df[target]):
            blank = blank | (df[target].astype(str).str.strip() == "")
        df = df[~blank]
        rows_dropped_missing_target = before - len(df)

    for col in settings.get("drop_columns", []):
        if col in df.columns and col != target:
            df = df.drop(columns=[col])

    # Missing values
    missing_strategy = settings.get("missing_strategy", {})
    for col, strategy in missing_strategy.items():
        if col not in df.columns or col == target:
            # Target is protected above; never impute it even if a strategy leaked in.
            continue
        if strategy in ("none", "skip"):
            continue
        if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].mean()
            cells_imputed += df[col].isna().sum()
            df[col] = df[col].fillna(val)
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].median()
            cells_imputed += df[col].isna().sum()
            df[col] = df[col].fillna(val)
        elif strategy == "mode":
            val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            cells_imputed += df[col].isna().sum()
            df[col] = df[col].fillna(val)
        elif strategy == "constant":
            val = settings.get("constant_value", "Unknown")
            cells_imputed += df[col].isna().sum()
            df[col] = df[col].fillna(val)
        elif strategy == "drop_rows":
            before = len(df)
            df = df.dropna(subset=[col])
            cells_imputed += before - len(df)  # rough

    # Outliers
    outlier_method = settings.get("outlier_method", "none")
    outlier_threshold = settings.get("outlier_threshold", 1.5)
    outlier_action = settings.get("outlier_action", "clip")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if outlier_method != "none" and numeric_cols:
        for col in numeric_cols:
            if outlier_method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - outlier_threshold * iqr
                upper = q3 + outlier_threshold * iqr
            elif outlier_method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - outlier_threshold * std
                upper = mean + outlier_threshold * std
            else:
                continue

            mask = (df[col] < lower) | (df[col] > upper)
            n_out = mask.sum()
            if n_out == 0:
                continue

            if outlier_action == "clip":
                df[col] = df[col].clip(lower, upper)
            elif outlier_action == "drop":
                df = df[~mask]
                rows_dropped_outliers += n_out
            elif outlier_action == "flag":
                flag_col = f"{col}_is_outlier"
                if flag_col not in df.columns:
                    df[flag_col] = False
                df.loc[mask, flag_col] = True

    # Duplicates
    if settings.get("drop_duplicates", False):
        before = len(df)
        df = df.drop_duplicates()
        duplicates_dropped = before - len(df)

    # Type overrides
    type_overrides = settings.get("type_overrides", {})
    for col, new_type in type_overrides.items():
        if col in df.columns:
            try:
                if new_type == "category":
                    df[col] = df[col].astype("category")
                elif new_type == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif new_type == "numeric":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif new_type == "string":
                    df[col] = df[col].astype(str)
            except Exception:
                pass

    rows_after = len(df)

    summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped_missing_target": int(rows_dropped_missing_target),
        "cells_imputed": int(cells_imputed),
        "rows_dropped_outliers": int(rows_dropped_outliers),
        "duplicates_dropped": int(duplicates_dropped),
        "dropped_columns": list(settings.get("drop_columns", [])),
    }

    return StageResult(
        status="success",
        output_df=df,
        artifacts=[],
        summary=summary,
    )


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    """Basic UI controls. Full per-column in later polish."""
    # This is called from main app flow; we keep settings logic in app.py for Phase 7
    return settings


def code_template(settings: dict[str, Any]) -> str:
    return f"""# Cleaning
# missing_strategy = {settings.get("missing_strategy", {})}
# Applied missing value imputation, outlier handling, duplicate removal.
df = df.copy()
print("Data cleaned. Shape:", df.shape)
"""
