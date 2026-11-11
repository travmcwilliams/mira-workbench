"""
utils/db.py
Singleton DuckDB connection manager for the DSI platform.

Per spec:
- Agents NEVER open their own connections.
- Use env APP_PROJECT_ROOT / APP_DB_PATH with sane Linux default.
- Single connection to avoid file corruption on the embedded DB.
"""

import os
from pathlib import Path
from typing import Optional

import duckdb

# Path conventions (critical - see spec §2)
# Default is the Linux VM path. On Windows dev without explicit APP_* env vars,
# we fall back to the directory containing this package (so DB lands inside checkout).
_default_root = "/home/app/project"
PROJECT_ROOT = Path(os.environ.get("APP_PROJECT_ROOT", _default_root))
# Windows dev convenience: if the caller did not explicitly set APP_PROJECT_ROOT
# (so we got the linux default), override using location of this utils/ package.
# This keeps the DB inside the checkout tree. Agents still only read the final values.
if (not os.environ.get("APP_PROJECT_ROOT")) and os.name == "nt":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("APP_DB_PATH", str(PROJECT_ROOT / "db/app.duckdb")))

_conn: Optional[duckdb.DuckDBPyConnection] = None


def get_db() -> duckdb.DuckDBPyConnection:
    """Return the singleton DuckDB connection. Initialize on first use."""
    global _conn
    if _conn is None:
        # Ensure parent dir exists (works on Linux and Windows)
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(DB_PATH))
        # Pragmas for analytics workload (safe defaults)
        _conn.execute("PRAGMA threads=4;")
        _conn.execute("PRAGMA enable_object_cache;")
    return _conn


def init_db_schema() -> None:
    """Create the canonical records table and supporting tables if they do not exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
            record_id      TEXT,
            entity_name    TEXT,
            location_text  TEXT,
            primary_value  DOUBLE,
            period         INTEGER,
            source_name    TEXT,
            source_id      TEXT,
            region         TEXT
        );
    """)
    # Intelligence layer tables (minimal for scaffold)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calendar_events (
            event_id TEXT PRIMARY KEY,
            record_id TEXT,
            source_id TEXT,
            event_type TEXT,
            event_date DATE,
            description TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lifecycle_states (
            record_id TEXT,
            source_id TEXT,
            state TEXT,
            deadline DATE,
            computed_at TIMESTAMP,
            metrics JSON
        );
    """)
    # Ensure at least the core table is queryable
    conn.execute("SELECT 1 FROM records LIMIT 0;")


def run_dedup() -> int:
    """
    Deduplicate records on the natural key after every ingest.
    Returns number of rows deleted.
    """
    conn = get_db()
    before = conn.execute("SELECT COUNT(*) FROM records;").fetchone()[0]
    conn.execute("""
        DELETE FROM records
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM records
            GROUP BY record_id, source_id, period
        );
    """)
    after = conn.execute("SELECT COUNT(*) FROM records;").fetchone()[0]
    deleted = before - after
    return deleted


def get_record_count() -> int:
    conn = get_db()
    return conn.execute("SELECT COUNT(*) FROM records;").fetchone()[0]


def close_db() -> None:
    """Close the singleton connection (call at process end if needed)."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
