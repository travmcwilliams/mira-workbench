"""
agents/holdings_agent.py
Holdings / Watchlist Agent (Agent 8).

Manage each end-user's tracked items through their lifecycle.
Per-user state, alerts on changes/deadlines, portfolio view.

Stub for product layer.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.db import get_db


def update_watchlist(user_id: str = "demo-user"):
    print(f"[holdings_agent] Updating watchlist for {user_id} ...")
    get_db()
    # TODO: read user actions (watch/unwatch), join to records + lifecycle, compute status.
    print("[holdings_agent] (stub) Per-user holdings + notifications not implemented.")
    return 0
