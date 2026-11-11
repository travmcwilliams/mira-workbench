#!/usr/bin/env python
"""
scripts/run_demo_ingest.py
End-to-end demo of the first real dataset milestone (spec Phase 2).

- Rebuilds clean DB
- Runs registry repair
- Ingests the sample direct-download source (00001) using the local CSV
- Shows dedup, counts, debug prints from mapper + ingestor

Run from project root:
    python scripts/run_demo_ingest.py
"""

import os
import sys
from pathlib import Path

# Make imports work when run directly
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# For local Windows dev demo, force a sane project-local DB unless APP_ already set by user.
# This keeps the file inside the checkout (C:\...dsi-platform\db\...) instead of the linux default.
if os.name == "nt" and not os.environ.get("APP_PROJECT_ROOT"):
    os.environ["APP_PROJECT_ROOT"] = str(ROOT)
    os.environ["APP_DB_PATH"] = str(ROOT / "db" / "app.duckdb")

from agents.source_ingestor import ingest_by_id
from utils.db import DB_PATH, get_db, get_record_count, init_db_schema


def main():
    print("=" * 60)
    print("DSI SCAFFOLD - DEMO INGEST (Grant Opportunities sample)")
    print(f"DB will be at: {DB_PATH}")
    print("=" * 60)

    # 1. Clean slate
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db_schema()
    print("[demo] DB schema initialized.")

    # 2. Verify registry
    print("\n[demo] Running registry_repair --verify-only ...")
    try:
        # call the function logic without sys.exit
        from scripts.registry_repair import load_registry, validate_source
        data = load_registry()
        errs = []
        for i, s in enumerate(data["sources"]):
            errs += validate_source(s, i)
        if errs:
            print("  REGISTRY ERRORS (fix before demo):")
            for e in errs:
                print("   ", e)
            sys.exit(1)
        print("  Registry OK.")
    except SystemExit:
        raise
    except Exception as e:
        print(f"  Repair check failed: {e}")
        sys.exit(1)

    # 3. Ingest the demo source using bundled sample CSV
    sample = ROOT / "data" / "raw" / "00001" / "grants_alpha_2025.csv"
    print(f"\n[demo] Ingesting source 00001 from {sample} ...")
    n = ingest_by_id("00001", local_file=sample, dry_run=False)
    print(f"\n[demo] Ingest complete. Rows from mapper: {n}")

    # 4. Show state
    final = get_record_count()
    print(f"\n[demo] Total records in DB after ingest + dedup: {final}")
    print("\n[demo] Sample of loaded records:")
    conn = get_db()
    print(conn.execute("SELECT * FROM records LIMIT 5;").df().to_string())

    print("\n" + "=" * 60)
    print("DEMO SUCCESS. Next: implement more agents, wire real discovery, etc.")
    print("Run: python scripts/registry_repair.py --verify-only")
    print("=" * 60)


if __name__ == "__main__":
    main()
