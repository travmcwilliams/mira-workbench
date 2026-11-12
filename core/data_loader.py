"""
core/data_loader.py

Data ingestion and schema detection.
Loads tabular files and infers column schemas.
Supports CSV, Excel, Parquet, JSON.
"""

from pathlib import Path
from typing import Any

import pandas as pd


def load_file(path: Path | str) -> pd.DataFrame:
    """Auto-dispatch by extension. Supports common tabular formats.
    Special handling for common restaurant POS labor reports that have 3-4 line preambles.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        # Try UTF-8 first, fall back to latin-1 (common for exports)
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        except Exception:
            # Fallback with error skipping (helps with footer rows in some labor reports)
            df = pd.read_csv(path, on_bad_lines='skip', engine='python')

        # Heuristic: if the first column looks like a title or the header row is shifted,
        # try skipping a few rows (common for "Labor Report - Restaurant Name" exports)
        if len(df) > 0:
            first_val = str(df.iloc[0, 0]) if df.shape[1] > 0 else ""
            if any(x in first_val.lower() for x in ["labor report", "employee totals", "report -"]):
                df = pd.read_csv(path, skiprows=3, on_bad_lines='skip', engine='python')
        return df

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {ext}. Supported: .csv, .xlsx, .xls, .parquet, .json")


_DATE_NAME_HINTS = ("date", "time", "timestamp", "week_start", "trading_day")


def _parse_rate(series: pd.Series) -> float:
    """Share of non-null values that parse as datetimes."""
    sample = series.dropna().head(200)
    if sample.empty:
        return 0.0
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean())


def _looks_like_date_column(col: str, series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    lowered = col.lower()
    if any(hint in lowered for hint in _DATE_NAME_HINTS):
        return _parse_rate(series) >= 0.8
    return _parse_rate(series) >= 0.95


def detect_schema(df: pd.DataFrame) -> dict[str, str]:
    """
    Return {col_name: 'numeric'|'categorical'|'datetime'|'text'|'id'}.
    Heuristic used to drive cleaning, feature engineering, and UI.
    """
    schema: dict[str, str] = {}
    n = len(df)
    if n == 0:
        return {col: "text" for col in df.columns}

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        nn = len(non_null)

        if nn == 0:
            schema[col] = "text"
            continue

        if _looks_like_date_column(col, s):
            schema[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            # Only integer-like numeric columns may be treated as identifiers.
            if pd.api.types.is_integer_dtype(s) and nn > 10 and s.nunique() / nn > 0.95:
                schema[col] = "id"
            else:
                schema[col] = "numeric"
        elif s.nunique() == nn and nn > 1:
            schema[col] = "id"
        elif s.nunique() / nn < 0.5:
            schema[col] = "categorical"
        else:
            # Default to text for strings with high cardinality or free text
            schema[col] = "text"
    return schema


# Columns that are common regression targets in bundled / ops datasets
_KNOWN_REGRESSION_TARGETS = {
    "medv",
    "labor_pct",
    "tips",
    "total_tips",
    "net_sales",
    "sales_per_labor_hour",
    "fare",
    "total_pay",
}
_OPS_TARGET_PRIORITY = [
    "labor_pct",
    "total_tips",
    "sales_per_labor_hour",
    "total_pay",
    "tips",
    "net_sales",
    "medv",
    "fare",
]
_KNOWN_CLASSIFICATION_TARGETS = {"survived", "churn", "target", "label"}
_TARGET_BLOCKLIST = {
    "day_of_week",
    "month",
    "week_start",
    "trading_day_date",
    "passengerid",
    "employee id",
    "last name",
    "first name",
    "1.5x_hrs",
    "reg_hrs",
    "role",
}


def suggest_target(
    df: pd.DataFrame,
    schema: dict[str, str] | None = None,
    task_type: str | None = None,
) -> dict[str, str | None]:
    """
    Suggest a task type and target column from schema + column names.
    Prefers known business metrics over id/date fields.
    """
    if schema is None:
        schema = detect_schema(df)

    lowered_map = {col: col.lower().replace(" ", "_") for col in df.columns}

    if task_type is None:
        for col, key in lowered_map.items():
            if key in _KNOWN_CLASSIFICATION_TARGETS:
                task_type = "classification"
                return {"task_type": task_type, "target_column": col, "reason": "known classification target"}
        task_type = "regression"

    candidates: list[tuple[int, str, str]] = []
    for col, dtype in schema.items():
        key = lowered_map[col]
        if key in _TARGET_BLOCKLIST:
            continue
        if dtype not in {"numeric", "categorical"}:
            continue
        score = 0
        reason = "numeric feature column"
        if task_type == "classification":
            if dtype != "categorical":
                continue
            score += 5
            reason = "low-cardinality categorical"
        elif dtype != "numeric":
            continue
        else:
            score += 5
            reason = "numeric measure"
        if key in _KNOWN_REGRESSION_TARGETS and task_type == "regression":
            score += 20
            reason = "known ops/benchmark target"
            if key in _OPS_TARGET_PRIORITY:
                score += 30 - _OPS_TARGET_PRIORITY.index(key) * 2
                reason = "restaurant ops / benchmark priority target"
        if key in _KNOWN_CLASSIFICATION_TARGETS and task_type == "classification":
            score += 20
            reason = "known classification target"
        if dtype == "id" or col.lower() in _DATE_NAME_HINTS:
            score -= 50
            reason = "identifier/date — poor target"
        if df[col].nunique(dropna=False) <= 1:
            score -= 100
        candidates.append((score, col, reason))

    if not candidates:
        return {"task_type": task_type, "target_column": None, "reason": "no suitable target found"}

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_col, reason = candidates[0]
    if best_score < 0:
        return {"task_type": task_type, "target_column": None, "reason": reason}
    return {"task_type": task_type, "target_column": best_col, "reason": reason}


def preview(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Return first n rows for UI preview."""
    return df.head(n)


def load_from_upload(uploaded_file: Any) -> pd.DataFrame:
    """
    Convenience for Streamlit st.file_uploader.
    uploaded_file is a UploadedFile-like object with .name and .getbuffer() or .read().
    """
    name = getattr(uploaded_file, "name", "uploaded")
    ext = Path(name).suffix.lower()

    if ext == ".csv":
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(uploaded_file)
    if ext == ".parquet":
        return pd.read_parquet(uploaded_file)
    if ext == ".json":
        return pd.read_json(uploaded_file)

    # Fallback: try reading as csv
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")
