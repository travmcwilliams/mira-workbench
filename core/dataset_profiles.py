"""Detect dataset shape for interactive dashboard routing."""

from __future__ import annotations

from typing import Literal

import pandas as pd

DatasetProfile = Literal["restaurant_daily", "employee_labor", "canonical_records", "generic"]

_CANONICAL_RECORDS_REQUIRED = {"entity_name", "primary_value", "source_id"}
_CANONICAL_RECORDS_OPTIONAL = {
    "record_id", "location_text", "period", "source_name", "region",
}
_RESTAURANT_DAILY_REQUIRED = {"net_sales", "labor_hours", "tips"}
_RESTAURANT_DAILY_OPTIONAL = {"labor_pct", "trading_day_date", "day_of_week", "wages"}
_EMPLOYEE_LABOR_REQUIRED = {"role"}
_EMPLOYEE_LABOR_MEASURES = {"reg_hrs", "total_tips", "net_sales", "total_pay"}


def normalize_column(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def column_set(df: pd.DataFrame) -> set[str]:
    return {normalize_column(c) for c in df.columns}


def detect_dataset_profile(df: pd.DataFrame) -> DatasetProfile:
    """Classify loaded data for dashboard selection."""
    cols = column_set(df)
    if _CANONICAL_RECORDS_REQUIRED.issubset(cols) and len(cols & _CANONICAL_RECORDS_OPTIONAL) >= 2:
        return "canonical_records"
    if _RESTAURANT_DAILY_REQUIRED.issubset(cols) and (
        "labor_pct" in cols or "wages" in cols or "trading_day_date" in cols
    ):
        return "restaurant_daily"
    if _EMPLOYEE_LABOR_REQUIRED.issubset(cols) and len(cols & _EMPLOYEE_LABOR_MEASURES) >= 2:
        return "employee_labor"
    return "generic"


def profile_label(profile: DatasetProfile) -> str:
    labels = {
        "restaurant_daily": "Restaurant daily ops",
        "employee_labor": "Employee / role labor",
        "canonical_records": "MIRA records",
        "generic": "Generic tabular",
    }
    return labels[profile]