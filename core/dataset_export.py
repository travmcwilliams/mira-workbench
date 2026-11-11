"""
core/dataset_export.py

Serialize a project's dataset (raw or working) to downloadable bytes in a
user-chosen format. Pure functions over a DataFrame — the Streamlit widget
that fronts this lives in ui/dataset_download.py.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pandas as pd

# Hard limit of the .xlsx format itself (2^20 rows incl. header).
EXCEL_MAX_ROWS = 1_048_575


@dataclass(frozen=True)
class ExportFormat:
    key: str
    label: str
    extension: str
    mime: str


FORMATS: tuple[ExportFormat, ...] = (
    ExportFormat("csv", "CSV", "csv", "text/csv"),
    ExportFormat("tsv", "TSV (tab-separated)", "tsv", "text/tab-separated-values"),
    ExportFormat(
        "xlsx",
        "Excel (.xlsx)",
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    ExportFormat("parquet", "Parquet", "parquet", "application/vnd.apache.parquet"),
    ExportFormat("json", "JSON (records)", "json", "application/json"),
    ExportFormat("jsonl", "JSON Lines", "jsonl", "application/x-ndjson"),
)

_BY_KEY = {f.key: f for f in FORMATS}


def get_format(key: str) -> ExportFormat:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown export format: {key!r}") from None


def export_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    """Serialize *df* to bytes in the given format. Raises ValueError on an
    unknown format or a dataset that exceeds the .xlsx row limit."""
    key = get_format(fmt).key

    text: str
    if key == "csv":
        text = df.to_csv(index=False)
        return text.encode("utf-8")
    if key == "tsv":
        text = df.to_csv(index=False, sep="\t")
        return text.encode("utf-8")
    if key == "xlsx":
        if len(df) > EXCEL_MAX_ROWS:
            raise ValueError(
                f"Dataset has {len(df):,} rows; Excel supports at most "
                f"{EXCEL_MAX_ROWS:,}. Choose CSV or Parquet instead."
            )
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue()
    if key == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue()
    if key == "json":
        text = df.to_json(orient="records", date_format="iso")
        return text.encode("utf-8")
    # jsonl
    text = df.to_json(orient="records", lines=True, date_format="iso")
    return text.encode("utf-8")


def export_filename(project_name: str, dataset: str, fmt: str) -> str:
    """Safe download filename like ``my_project_working.csv``."""
    f = get_format(fmt)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", project_name).strip("_") or "dataset"
    return f"{slug}_{dataset}.{f.extension}"
