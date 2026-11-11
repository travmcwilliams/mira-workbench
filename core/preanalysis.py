"""Column-level statistics and recommendations for pre-analysis decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.data_loader import detect_schema

MISSING_STRATEGIES = ("none", "mean", "median", "mode", "constant", "drop_rows")
ENCODING_OPTIONS = ("none", "onehot", "label")
OUTLIER_METHODS = ("none", "iqr", "zscore")
OUTLIER_ACTIONS = ("clip", "drop", "flag")
FS_METHODS = ("correlation", "variance", "rfe", "importance")


def column_summary(df: pd.DataFrame, schema: dict[str, str] | None = None) -> pd.DataFrame:
    """Rich per-column stats for the pre-analysis workbench."""
    if df.empty:
        return pd.DataFrame()

    schema = schema or detect_schema(df)
    rows: list[dict[str, Any]] = []
    n = len(df)

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        schema_type = schema.get(col, "text")
        missing = int(series.isna().sum())
        missing_pct = round(100.0 * missing / n, 2) if n else 0.0
        nunique = int(series.nunique(dropna=True))

        row: dict[str, Any] = {
            "column": col,
            "dtype": dtype,
            "schema_type": schema_type,
            "missing": missing,
            "missing_pct": missing_pct,
            "unique": nunique,
            "sample": _sample_value(series),
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = pd.to_numeric(series, errors="coerce")
            row["mean"] = round(float(clean.mean()), 4) if clean.notna().any() else None
            row["std"] = round(float(clean.std()), 4) if clean.notna().any() else None
            row["min"] = round(float(clean.min()), 4) if clean.notna().any() else None
            row["max"] = round(float(clean.max()), 4) if clean.notna().any() else None
        else:
            row["mean"] = None
            row["std"] = None
            row["min"] = None
            row["max"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def recommend_missing_strategy(
    schema_type: str,
    missing_pct: float,
) -> str:
    if missing_pct <= 0:
        return "none"
    if schema_type == "numeric":
        return "median"
    if schema_type == "categorical":
        return "mode"
    return "constant"


def recommend_encoding(schema_type: str, nunique: int) -> str:
    if schema_type != "categorical":
        return "none"
    return "onehot" if nunique <= 12 else "label"


def correlation_preview(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    max_cols: int = 12,
) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if target and target in numeric.columns:
        cols = [target] + [c for c in numeric.columns if c != target][: max_cols - 1]
        numeric = numeric[cols]
    elif len(numeric.columns) > max_cols:
        numeric = numeric.iloc[:, :max_cols]
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr().round(3)


def missing_matrix_summary(df: pd.DataFrame) -> dict[str, Any]:
    miss = df.isna().sum()
    cols_with_missing = int((miss > 0).sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    return {
        "columns_with_missing": cols_with_missing,
        "missing_cells": missing_cells,
        "missing_cell_pct": round(100.0 * missing_cells / total_cells, 2) if total_cells else 0.0,
        "complete_rows": int(df.dropna().shape[0]),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _sample_value(series: pd.Series) -> str:
    non_null = series.dropna().head(3).tolist()
    if not non_null:
        return ""
    text = ", ".join(str(v)[:40] for v in non_null)
    return text[:120]