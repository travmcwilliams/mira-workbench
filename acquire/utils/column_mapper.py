"""
utils/column_mapper.py
Lightweight explicit + heuristic column mapper.

Generalized from CountyIQ's extractors/column_mapper.py but made domain-neutral
and suitable for any record type.

Use this when you have known per-source column aliases and want deterministic mapping
before or alongside the heuristic logic in agents/schema_mapper.py.

Example:
    mapper = ColumnMapper({
        "grantid": "record_id",
        "programname": "entity_name",
        "state": "location_text",
        "awardamount": "primary_value",
        "fiscalyear": "period",
    })
    canonical_row = mapper.map_row(raw_row)
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ColumnMapper:
    def __init__(self, alias_map: Optional[Dict[str, str]] = None):
        """
        alias_map: lowercased_raw_name -> canonical_field
        Example: {"pin": "parcel_id", "amount": "primary_value"}
        """
        self.alias_map: Dict[str, str] = {}
        if alias_map:
            for k, v in alias_map.items():
                self.alias_map[self._norm(k)] = v

    def _norm(self, s: str) -> str:
        return (s or "").strip().lower().replace(" ", "_").replace("-", "_")

    def add_aliases(self, aliases: Dict[str, str]):
        for k, v in aliases.items():
            self.alias_map[self._norm(k)] = v

    def map_row(self, raw: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Map a single raw dict to a dict using only the known canonical keys you care about.
        Unmapped fields are dropped (use schema_mapper for full heuristic fallback).
        """
        out: Dict[str, Any] = dict(defaults) if defaults else {}
        for raw_key, value in raw.items():
            if value is None:
                continue
            norm_key = self._norm(str(raw_key))
            canonical = self.alias_map.get(norm_key)
            if canonical:
                # Coerce common numeric fields
                if canonical in ("primary_value", "amount", "value") and isinstance(value, str):
                    try:
                        value = float(value.replace(",", "").replace("$", "").strip())
                    except Exception:
                        pass
                out[canonical] = value
        return out

    def map_rows(self, rows: list[Dict[str, Any]], defaults: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
        return [self.map_row(r, defaults) for r in rows]


# Example usage for the grant demo (can be expanded per source in registry extras or a per-source file)
GRANT_DEMO_ALIASES = {
    "grantid": "record_id",
    "programname": "entity_name",
    "state": "location_text",
    "awardamount": "primary_value",
    "fiscalyear": "period",
    "agency": "source_name",  # will be overridden by registry value usually
}
