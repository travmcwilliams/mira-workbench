"""
agents/lifecycle_engine.py
Lifecycle / Deadline Engine (Agent 7).

Track time-sensitive deadlines; compute derived temporal/financial metrics (aging, days-to-close, expected yield over time, etc.).

Per spec §13 this is one of the 4 things you touch when adapting to a new domain.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.db import get_db


def compute_lifecycle():
    print("[lifecycle_engine] Updating lifecycle states + metrics...")
    conn = get_db()
    # TODO: for each record derive state (open/closing/closed), deadline, metrics JSON.
    # Write to lifecycle_states table.
    print("[lifecycle_engine] (stub) Domain deadline rules + derived metrics go here.")
    n = conn.execute("SELECT COUNT(*) FROM lifecycle_states;").fetchone()[0]
    print(f"[lifecycle_engine] lifecycle_states rows: {n}")
    return n
