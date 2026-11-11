"""DuckDB store for prediction-market backfill — append-only, campaign-style.

Same discipline as the news campaign store: durable state lives here (not in any
conversation), rows are appended (never mutated), and resume-ability comes from
reading what's already present. fetch_log records every attempt so a market is
never silently skipped — a failed fetch is a visible row, not an absence.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parents[2] / "var" / "predmarkets" / "predmarkets.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    platform      VARCHAR NOT NULL,
    market_id     VARCHAR NOT NULL,
    question      VARCHAR,
    tag_slug      VARCHAR,
    end_date      VARCHAR,
    closed        BOOLEAN,
    volume_usd    DOUBLE,
    clob_token_yes VARCHAR,
    outcomes      VARCHAR,
    inserted_at   TIMESTAMP DEFAULT current_timestamp
);
CREATE TABLE IF NOT EXISTS prices (
    platform   VARCHAR NOT NULL,
    market_id  VARCHAR NOT NULL,
    ts         BIGINT NOT NULL,
    date       VARCHAR NOT NULL,
    prob       DOUBLE,
    inserted_at TIMESTAMP DEFAULT current_timestamp
);
CREATE TABLE IF NOT EXISTS fetch_log (
    platform   VARCHAR NOT NULL,
    market_id  VARCHAR NOT NULL,
    stage      VARCHAR NOT NULL,
    status     VARCHAR NOT NULL,
    n_points   INTEGER,
    note       VARCHAR,
    fetched_at TIMESTAMP DEFAULT current_timestamp
);
"""


def connect(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """DuckDB is single-writer: concurrent campaigns need separate files
    (e.g. kalshi.duckdb next to the default polymarket predmarkets.duckdb)."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_SCHEMA)
    return con


def done_market_ids(con: duckdb.DuckDBPyConnection, platform: str, stage: str) -> set[str]:
    """Markets already attempted at this stage (any status) — the resume skip-set."""
    rows = con.execute(
        "SELECT DISTINCT market_id FROM fetch_log WHERE platform = ? AND stage = ?",
        [platform, stage]).fetchall()
    return {r[0] for r in rows}
