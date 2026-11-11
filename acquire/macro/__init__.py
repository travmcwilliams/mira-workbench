"""MIRA-owned point-in-time macro data with release vintages.

Leak-safety starts in the data layer: every observation carries
``available_for_signal_at`` (when a signal was allowed to know it) and a
``vintage`` (which ALFRED/ECB release edition produced the value). As-of
queries never return rows that were not yet public.

Public surface:
    from macro.store import open_db, upsert_rows, as_of, list_series
    from macro.fred import fetch_fred_series, DEFAULT_FRED_SERIES
    from macro.ecb import fetch_ecb_fx, DEFAULT_ECB_CURRENCIES
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_DB",
    "DEFAULT_ECB_CURRENCIES",
    "DEFAULT_FRED_SERIES",
    "as_of",
    "fetch_ecb_fx",
    "fetch_fred_series",
    "list_series",
    "open_db",
    "upsert_rows",
]

from .ecb import DEFAULT_ECB_CURRENCIES, fetch_ecb_fx
from .fred import DEFAULT_FRED_SERIES, fetch_fred_series
from .store import DEFAULT_DB, as_of, list_series, open_db, upsert_rows
