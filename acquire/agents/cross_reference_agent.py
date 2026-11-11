"""
agents/cross_reference_agent.py
Cross-Reference Agent (Agent 6).

Match primary records against a secondary authoritative dataset (e.g. another registry, census, paid list).

Uses pandas + rapidfuzz for fuzzy matching.

Stub.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.db import get_db


def cross_reference(secondary_source: str = "secondary.csv"):
    print(f"[cross_reference] Matching against {secondary_source} ...")
    conn = get_db()
    # TODO: load secondary (could be another table or CSV), fuzzy join on entity_name/location,
    # set flags or write match table.
    print("[cross_reference] (stub) Implement rapidfuzz matching + flag writing.")
    n = conn.execute("SELECT COUNT(*) FROM records;").fetchone()[0]
    print(f"[cross_reference] Primary records available for matching: {n}")
    return 0
