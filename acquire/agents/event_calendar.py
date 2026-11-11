"""
agents/event_calendar.py
Event / Deadline Calendar Agent (Agent 3).

Builds normalized time-based schedule of events tied to records (e.g. application open/close, award dates).

Reads: records + sources
Writes: calendar_events table (see utils/db.py)

Stub.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.db import get_db


def build_calendar(source_id: str = None):
    print("[event_calendar] Building calendar events...")
    conn = get_db()
    # TODO: domain logic - parse dates from entity_name / extras, or from source pages.
    # Example skeleton:
    # rows = conn.execute("SELECT record_id, source_id, ... FROM records WHERE ...").df()
    # for each derive event_date etc.
    print("[event_calendar] (stub) No events generated. Implement deadline extraction for your domain.")
    count = conn.execute("SELECT COUNT(*) FROM calendar_events;").fetchone()[0]
    print(f"[event_calendar] Calendar table now has {count} events.")
    return count
