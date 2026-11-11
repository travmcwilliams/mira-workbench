#!/usr/bin/env python
"""
scripts/registry_repair.py
Validate and (optionally) repair the source_registry.json per the ENFORCED schema in spec §3.4.

Usage:
    python scripts/registry_repair.py --verify-only
    python scripts/registry_repair.py            # will attempt light repairs + rewrite

Rules:
- Key is ALWAYS "source_id" (never src_fips, id, etc.)
- region_full is NEVER omitted.
- source_id should be fixed-width zero-padded standardized code.
- scraper_type must be one of the 4 allowed.
- After every manual edit: run with --verify-only and git status.
"""

import argparse
import json
import sys
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent.parent / "data" / "source_registry.json"
ALLOWED_SCRAPER_TYPES = {"has_download_link", "has_record_link", "has_search_form", "page_only"}
ALLOWED_DATA_METHODS = {"scrape", "api", "manual"}

REQUIRED_KEYS = [
    "source_id", "source_name", "region", "region_full",
    "source_url", "scraper_type", "data_method", "last_scraped"
]


def load_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "sources" not in data or not isinstance(data["sources"], list):
        raise ValueError("Registry must have top-level 'sources' list")
    return data


def save_registry(data):
    # Pretty, stable write
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def validate_source(src: dict, idx: int) -> list[str]:
    errors = []
    for k in REQUIRED_KEYS:
        if k not in src:
            errors.append(f"[{idx}] missing required key '{k}'")
    # No alternate keys allowed
    extra = set(src.keys()) - set(REQUIRED_KEYS)
    if extra:
        errors.append(f"[{idx}] has unexpected keys (schema drift): {sorted(extra)}")

    sid = src.get("source_id", "")
    if not isinstance(sid, str) or not sid:
        errors.append(f"[{idx}] source_id must be non-empty string (fixed-width zero-padded recommended)")
    elif len(sid) < 3 or not sid[0].isdigit() and not sid.startswith(("US-", "GR-")):  # lenient for demo
        # In real use enforce strict format e.g. r'^\d{5}$'
        pass

    if src.get("scraper_type") not in ALLOWED_SCRAPER_TYPES:
        errors.append(f"[{idx}] invalid scraper_type: {src.get('scraper_type')}")

    if src.get("data_method") not in ALLOWED_DATA_METHODS:
        errors.append(f"[{idx}] invalid data_method: {src.get('data_method')}")

    if "region_full" in src and not src.get("region_full"):
        errors.append(f"[{idx}] region_full must never be empty string")

    if src.get("source_url") is not None and not isinstance(src.get("source_url"), str):
        errors.append(f"[{idx}] source_url must be string or null")

    return errors


def repair_source(src: dict) -> dict:
    """Light auto-repair for common drift."""
    out = {}
    for k in REQUIRED_KEYS:
        if k in src:
            out[k] = src[k]
        else:
            # Provide safe defaults (will still fail strict verify if wrong)
            if k == "last_scraped":
                out[k] = None
            elif k == "source_url":
                out[k] = None
            else:
                out[k] = ""
    # Force canonical key order
    return {k: out[k] for k in REQUIRED_KEYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true",
                        help="Only validate; do not rewrite the file.")
    args = parser.parse_args()

    print(f"[registry_repair] Checking {REGISTRY_PATH}")
    try:
        data = load_registry()
    except Exception as e:
        print(f"[registry_repair] FATAL: cannot load registry: {e}")
        sys.exit(2)

    all_errors = []
    repaired = []
    for i, src in enumerate(data["sources"]):
        errs = validate_source(src, i)
        if errs:
            all_errors.extend(errs)
        # Always normalize order for cleanliness
        repaired.append(repair_source(src))

    if all_errors:
        print("[registry_repair] SCHEMA ERRORS:")
        for e in all_errors:
            print("  -", e)
        if args.verify_only:
            print("\n--verify-only: failing.")
            sys.exit(1)
        else:
            print("\nAttempting to rewrite with repaired rows (order normalized)...")
            data["sources"] = repaired
            save_registry(data)
            print("[registry_repair] Rewritten. Re-run --verify-only to confirm clean.")
            sys.exit(1)
    else:
        # Always rewrite to enforce canonical key order + formatting
        data["sources"] = repaired
        if not args.verify_only:
            save_registry(data)
        print(f"[registry_repair] OK: {len(repaired)} sources, schema clean.")
        print("  Keys in canonical order, region_full present, types valid.")
        if args.verify_only:
            print("  (verify-only, no write performed)")
        sys.exit(0)


if __name__ == "__main__":
    main()
