"""
agents/enrichment_agent.py
Enrichment + Scoring Agent (Agent 5).

- Calls 3rd-party enrichment API (ENRICHMENT-API-KEY) to add attributes.
- Computes domain score / ranking (the main "intelligence" product).
- For tax liens / grants / etc. this is where the secret sauce (yield/risk/liquidity or equivalent) lives.

Stub. See spec §13: only scoring model is domain-specific.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.db import get_db
from utils.secrets import get_enrichment_api_key


def enrich_and_score(limit: int = 100):
    key = get_enrichment_api_key()
    if not key:
        print("[enrichment] No ENRICHMENT-API-KEY. Skipping external calls (scaffold mode).")

    print("[enrichment] Running enrichment + scoring pass...")
    conn = get_db()
    # TODO: select un-enriched records, call provider, compute score, write back
    # (consider adding an 'enriched' JSON column or separate table in real impl)
    count = conn.execute("SELECT COUNT(*) FROM records;").fetchone()[0]
    print(f"[enrichment] (stub) Scored/enriched pass complete. Base records: {count}")
    # In real run: print final enriched count or updated rows.
    return count
