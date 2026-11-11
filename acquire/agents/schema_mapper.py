"""
agents/schema_mapper.py
Schema Mapper Agent (Agent 10).

Maps heterogeneous source columns onto the single canonical `records` schema (§3.3).

Conventions (MUST):
- Print column names + first 3 rows of input BEFORE any parsing logic.
- Print final record count after producing canonical rows.
- Never open DB conn itself; return df for caller to load.

For a new domain you primarily edit the MAPPING RULES + any LLM assist here (§13).
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Canonical columns (exact order/names from spec)
CANONICAL_COLS = [
    "record_id", "entity_name", "location_text", "primary_value",
    "period", "source_name", "source_id", "region"
]


def print_input_debug(df: pd.DataFrame, source_label: str = "input"):
    print(f"\n[schema_mapper] DEBUG {source_label}:")
    print("  Columns:", list(df.columns))
    print("  First 3 rows:")
    print(df.head(3).to_string())
    print(f"  Total rows in raw: {len(df)}")


def map_to_canonical(
    raw_df: pd.DataFrame,
    source_id: str,
    source_name: str,
    region: str,
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Transform raw_df -> canonical df.

    mapping: optional explicit {canonical_col: raw_col} override from registry or CLI.
    Explicit mapping takes precedence. Unmapped fields fall back to heuristics.

    Improvements:
    - Much broader heuristics for public/open data (iata, airport, cnt, etc.)
    - Explicit mapping support for per-source control without touching this file
    - Composite location_text (city, state, country)
    - First-column fallback for record_id on arbitrary datasets
    - Better logging of chosen columns and fallbacks
    """
    print_input_debug(raw_df, f"source_id={source_id}")

    df = raw_df.copy()
    out = pd.DataFrame(columns=CANONICAL_COLS)

    col_map = mapping or {}

    # Heuristics (order matters; first match wins)
    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # case-insensitive fallback
        lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    # --- record_id (stable identifier) ---
    # Explicit mapping wins
    rid_col = col_map.get("record_id") or find_col([
        "GrantID", "id", "record_id", "opportunity_id", "uid",
        "iata", "code", "airport_code", "airport_id", "key", "uid", "identifier"
    ])
    if rid_col:
        out["record_id"] = df[rid_col].astype(str).str.strip()
        if "record_id" in col_map:
            print(f"[schema_mapper] record_id from explicit mapping: {rid_col}")
    else:
        # Fallback for arbitrary public data
        if len(df.columns) > 0:
            first_col = df.columns[0]
            out["record_id"] = df[first_col].astype(str).str.strip()
            print(f"[schema_mapper] Using first column '{first_col}' as record_id fallback.")

    # --- entity_name (primary named entity) ---
    ent_col = col_map.get("entity_name") or find_col([
        "ProgramName", "entity_name", "title", "name", "program", "grant_name",
        "airport", "facility", "location_name", "description", "org", "organization"
    ])
    if ent_col:
        out["entity_name"] = df[ent_col].astype(str).str.strip()
        if "entity_name" in col_map:
            print(f"[schema_mapper] entity_name from explicit mapping: {ent_col}")

    # --- location_text ---
    loc_col = col_map.get("location_text") or find_col([
        "State", "location", "location_text", "jurisdiction", "county", "city", "region",
        "address", "place"
    ])
    if loc_col:
        out["location_text"] = df[loc_col].astype(str).str.strip()
        if "location_text" in col_map:
            print(f"[schema_mapper] location_text from explicit mapping: {loc_col}")
    else:
        # Composite from common geo fields
        parts = []
        for field in ["city", "state", "country", "region"]:
            c = find_col([field])
            if c:
                parts.append(df[c].astype(str).str.strip())
        if parts:
            out["location_text"] = pd.concat(parts, axis=1).agg(", ".join, axis=1)
            print("[schema_mapper] Built composite location_text from available geo columns.")

    # --- primary_value (main quantitative) ---
    val_col = col_map.get("primary_value") or find_col([
        "AwardAmount", "amount", "primary_value", "value", "funding", "award", "dollars",
        "cnt", "count", "traffic", "passengers", "volume", "number", "total"
    ])
    if val_col:
        out["primary_value"] = pd.to_numeric(df[val_col], errors="coerce")
        if "primary_value" in col_map:
            print(f"[schema_mapper] primary_value from explicit mapping: {val_col}")

    # --- period (year/cycle) ---
    per_col = col_map.get("period") or find_col(["FiscalYear", "period", "year", "fy", "cycle", "date", "year"])
    if per_col:
        out["period"] = pd.to_numeric(df[per_col], errors="coerce").astype("Int64")
        if "period" in col_map:
            print(f"[schema_mapper] period from explicit mapping: {per_col}")
    else:
        out["period"] = pd.NA
        print("[schema_mapper] No period/year column found — leaving as null (common for snapshot data).")

    # Fill constants from registry
    out["source_id"] = source_id
    out["source_name"] = source_name
    out["region"] = region

    # Drop rows with no usable record_id
    before = len(out)
    out = out[out["record_id"].notna() & (out["record_id"].astype(str).str.len() > 0)].copy()
    dropped = before - len(out)
    if dropped:
        print(f"[schema_mapper] Dropped {dropped} rows without usable record_id.")
    else:
        print(f"[schema_mapper] All {len(out)} rows kept (record_id present).")

    # Ensure column order and dtypes
    out = out[CANONICAL_COLS]
    out["primary_value"] = out["primary_value"].astype(float)
    out["period"] = out["period"].astype("Int64")

    print(f"[schema_mapper] Produced {len(out)} canonical rows.")
    print("  Canonical columns:", CANONICAL_COLS)
    if len(out) > 0:
        print(out.head(3).to_string())
    return out


def map_csv(
    csv_path: Path,
    source_id: str,
    source_name: str,
    region: str,
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Convenience: read CSV then map."""
    df = pd.read_csv(csv_path)
    return map_to_canonical(df, source_id, source_name, region, mapping)
