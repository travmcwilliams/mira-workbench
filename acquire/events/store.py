"""Durable macro-event store — dedicated DuckDB, append-only, idempotent ingest.

One table in ``var/macro_events/macro_events.duckdb`` (NOT the demo
``acquire/db/app.duckdb`` — that stays untouched, same isolation the news
campaign store uses):

  * ``events`` — one row per (source, event_type, country, release_datetime,
    revision), keyed by a sha1 of exactly those fields (``event_id``). Re-runs
    of any ingest are no-ops (``ON CONFLICT DO NOTHING`` — first write wins), so
    a calendar refresh can always be replayed safely.

Append-only is a hard rule here: this module contains no UPDATE or DELETE. A
corrected or enriched observation (e.g. the *actual* print arriving days after
the calendar row, or a consensus revision) is a SUCCESSOR row ingested with
``revision`` bumped — readers disambiguate via ``revision``/``ingested_at``.
That matches the platform's artifact-chain doctrine: change by writing a
successor, never by mutating.

``surprise`` is computed as ``actual - consensus`` at ingest when both are
present and no explicit value was given. ``asset_classes`` is stored as a JSON
array string (same shape as ``articles.tickers`` in the news store).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb
import pandas as pd

DEFAULT_DB = "var/macro_events/macro_events.duckdb"

_DDL = """
CREATE TABLE IF NOT EXISTS events (
    event_id         TEXT PRIMARY KEY,
    event_type       TEXT NOT NULL,
    release_datetime TIMESTAMP NOT NULL,
    country          TEXT NOT NULL,
    actual           DOUBLE,
    consensus        DOUBLE,
    previous         DOUBLE,
    surprise         DOUBLE,
    source           TEXT NOT NULL,
    asset_classes    TEXT,
    revision         INTEGER NOT NULL DEFAULT 0,
    ingested_at      TIMESTAMP DEFAULT current_timestamp
);
"""


def open_db(db_path: str | Path = DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    """Open (creating parents + schema if needed) the macro-events DuckDB."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_DDL)
    return con


def event_id(source: str, event_type: str, country: str,
             release_datetime: str, revision: int = 0) -> str:
    """Deterministic identity key — the append-only chain's dedup anchor."""
    key = f"{source}|{event_type}|{country}|{release_datetime}|{revision}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# ingest (append-only; re-runs are no-ops)
# --------------------------------------------------------------------------- #

_INSERT = """
INSERT INTO events (event_id, event_type, release_datetime, country, actual,
                    consensus, previous, surprise, source, asset_classes, revision)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (event_id) DO NOTHING
"""


def _surprise(row: Mapping[str, Any]) -> float | None:
    if row.get("surprise") is not None:
        return float(row["surprise"])
    actual, consensus = row.get("actual"), row.get("consensus")
    if actual is not None and consensus is not None:
        return float(actual) - float(consensus)
    return None


def _asset_classes(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(list(value))


def ingest_events(con: duckdb.DuckDBPyConnection,
                  rows: Iterable[Mapping[str, Any]]) -> int:
    """Idempotently append event rows; returns the number of genuinely NEW rows.

    Each row needs ``event_type``, ``release_datetime`` and ``source``;
    ``event_id`` is derived from the identity fields when absent. Value fields
    (``actual``/``consensus``/``previous``/``surprise``) are optional — a
    calendar-only ingest leaves them NULL and a later enrichment appends a
    successor row with ``revision`` bumped (never an UPDATE).
    """
    rows = list(rows)
    if not rows:
        return 0
    params = []
    for r in rows:
        country = r.get("country", "US")
        revision = int(r.get("revision", 0))
        eid = r.get("event_id") or event_id(
            r["source"], r["event_type"], country,
            str(r["release_datetime"]), revision)
        params.append((
            eid, r["event_type"], r["release_datetime"], country,
            r.get("actual"), r.get("consensus"), r.get("previous"),
            _surprise(r), r["source"], _asset_classes(r.get("asset_classes")),
            revision,
        ))
    before = con.execute("SELECT count(*) FROM events").fetchone()[0]
    con.executemany(_INSERT, params)
    after = con.execute("SELECT count(*) FROM events").fetchone()[0]
    return int(after - before)


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #

def fetch_events(
    con: duckdb.DuckDBPyConnection,
    *,
    event_type: str | None = None,
    country: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Events as a DataFrame, oldest first, with a study-ready ISO ``date`` column.

    ``start``/``end`` filter on the release DATE (inclusive). When successor
    revisions exist for one identity, ALL are returned — callers pick a policy
    (``revision``/``ingested_at``); the store never hides history.
    """
    sql = "SELECT * FROM events"
    clauses, args = [], []
    if event_type is not None:
        clauses.append("event_type = ?")
        args.append(event_type)
    if country is not None:
        clauses.append("country = ?")
        args.append(country)
    if start is not None:
        clauses.append("CAST(release_datetime AS DATE) >= CAST(? AS DATE)")
        args.append(start)
    if end is not None:
        clauses.append("CAST(release_datetime AS DATE) <= CAST(? AS DATE)")
        args.append(end)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY release_datetime, event_type, revision"
    df = con.execute(sql, args).df()
    if not df.empty:
        df["date"] = pd.to_datetime(df["release_datetime"]).dt.strftime("%Y-%m-%d")
    else:
        df["date"] = pd.Series(dtype=str)
    return df


def event_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Row count per event_type (quick coverage check)."""
    return {
        str(t): int(n)
        for t, n in con.execute(
            "SELECT event_type, count(*) FROM events "
            "GROUP BY event_type ORDER BY event_type").fetchall()
    }
