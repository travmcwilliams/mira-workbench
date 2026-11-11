#!/usr/bin/env python
"""
scripts/rebuild_db.py
Clean rebuild of the embedded DB from scratch + run dedup.
Useful after schema changes or when you want a pristine start.
"""

import sys
from pathlib import Path

# Ensure project root on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import DB_PATH, get_db, get_record_count, init_db_schema, run_dedup


def main():
    print(f"[rebuild_db] Using DB at {DB_PATH}")
    if DB_PATH.exists():
        print("  Removing existing DB file...")
        DB_PATH.unlink()

    get_db()
    init_db_schema()
    print("  Schema initialized (records + calendar_events + lifecycle_states).")

    deleted = run_dedup()
    print(f"  Dedup run (deleted {deleted}).")
    print(f"  Current record count: {get_record_count()}")
    print("[rebuild_db] Done. DB is clean.")


if __name__ == "__main__":
    main()
