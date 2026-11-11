"""DuckDB-backed point-in-time macro series store — MIRA-owned, no foreign paths.

Table ``macro_series`` is the sole durable home for FRED/ALFRED + ECB FX rows.
Every row carries:

  * ``observation_date`` — the economic period the value describes
  * ``available_for_signal_at`` — first instant a signal may use this value
  * ``vintage`` — which release edition produced the value (ALFRED vintage date,
    or ``current`` for a latest-only pull)

Idempotent upsert key: ``(series_id, observation_date, vintage)``. Re-running a
backfill never duplicates; revised values for the same vintage overwrite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb

DEFAULT_DB = "var/macro/macro_series.duckdb"

_DDL = """
CREATE TABLE IF NOT EXISTS macro_series (
    series_id               TEXT NOT NULL,
    observation_date        DATE NOT NULL,
    value                   DOUBLE,
    release_timestamp       TIMESTAMP,
    available_for_signal_at TIMESTAMP NOT NULL,
    vintage                 TEXT NOT NULL,
    source                  TEXT NOT NULL,
    frequency               TEXT,
    PRIMARY KEY (series_id, observation_date, vintage)
);

CREATE INDEX IF NOT EXISTS idx_macro_series_asof
    ON macro_series (series_id, available_for_signal_at);
"""

_UPSERT = """
INSERT INTO macro_series (
    series_id, observation_date, value, release_timestamp,
    available_for_signal_at, vintage, source, frequency
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (series_id, observation_date, vintage) DO UPDATE SET
    value                   = excluded.value,
    release_timestamp       = excluded.release_timestamp,
    available_for_signal_at = excluded.available_for_signal_at,
    source                  = excluded.source,
    frequency               = excluded.frequency
"""

_REQUIRED = (
    "series_id",
    "observation_date",
    "value",
    "available_for_signal_at",
    "vintage",
    "source",
)


def open_db(db_path: str | Path = DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    """Open (creating parents + schema if needed) the macro DuckDB."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_DDL)
    return con


def upsert_rows(
    con: duckdb.DuckDBPyConnection,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    """Idempotent upsert. Returns number of input rows accepted (post-validation).

    Re-inserting the same ``(series_id, observation_date, vintage)`` overwrites
    value / timestamps / source / frequency; row count stays stable.
    """
    params: list[tuple[Any, ...]] = []
    for r in rows:
        missing = [k for k in _REQUIRED if k not in r or r[k] is None]
        if missing:
            raise ValueError(f"macro row missing required fields: {missing}")
        if r.get("value") is None:
            continue
        params.append(
            (
                str(r["series_id"]),
                str(r["observation_date"])[:10],
                float(r["value"]),
                r.get("release_timestamp"),
                str(r["available_for_signal_at"]),
                str(r["vintage"]),
                str(r["source"]),
                r.get("frequency"),
            )
        )
    if not params:
        return 0
    con.executemany(_UPSERT, params)
    return len(params)


def row_count(con: duckdb.DuckDBPyConnection) -> int:
    return int(con.execute("SELECT count(*) FROM macro_series").fetchone()[0])


def list_series(
    con: duckdb.DuckDBPyConnection,
    *,
    series_id: str | None = None,
) -> list[dict[str, Any]]:
    """Summary per series: counts, date span, sources, vintage span."""
    if series_id:
        sql = """
            SELECT series_id,
                   count(*) AS n_rows,
                   min(observation_date) AS first_obs,
                   max(observation_date) AS last_obs,
                   min(available_for_signal_at) AS first_available,
                   max(available_for_signal_at) AS last_available,
                   count(DISTINCT vintage) AS n_vintages,
                   any_value(source) AS source,
                   any_value(frequency) AS frequency
            FROM macro_series
            WHERE series_id = ?
            GROUP BY series_id
            ORDER BY series_id
        """
        raw = con.execute(sql, [series_id]).fetchall()
    else:
        sql = """
            SELECT series_id,
                   count(*) AS n_rows,
                   min(observation_date) AS first_obs,
                   max(observation_date) AS last_obs,
                   min(available_for_signal_at) AS first_available,
                   max(available_for_signal_at) AS last_available,
                   count(DISTINCT vintage) AS n_vintages,
                   any_value(source) AS source,
                   any_value(frequency) AS frequency
            FROM macro_series
            GROUP BY series_id
            ORDER BY series_id
        """
        raw = con.execute(sql).fetchall()

    cols = (
        "series_id", "n_rows", "first_obs", "last_obs",
        "first_available", "last_available", "n_vintages", "source", "frequency",
    )
    return [dict(zip(cols, row, strict=True)) for row in raw]


def as_of(
    con: duckdb.DuckDBPyConnection,
    as_of_ts: str,
    *,
    series_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Latest known value per series at ``as_of_ts`` (point-in-time safe).

    Only rows with ``available_for_signal_at <= as_of_ts`` are eligible. Among
    those, pick the most recent availability (then most recent observation_date)
    so revised vintages surface only after they were public.
    """
    params: list[Any] = [as_of_ts]
    series_filter = ""
    if series_ids:
        placeholders = ", ".join("?" * len(series_ids))
        series_filter = f"AND series_id IN ({placeholders})"
        params.extend(series_ids)

    sql = f"""
        SELECT series_id, observation_date, value, release_timestamp,
               available_for_signal_at, vintage, source, frequency
        FROM macro_series
        WHERE available_for_signal_at <= ?
          {series_filter}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY series_id
            ORDER BY available_for_signal_at DESC, observation_date DESC
        ) = 1
        ORDER BY series_id
    """
    cols = (
        "series_id", "observation_date", "value", "release_timestamp",
        "available_for_signal_at", "vintage", "source", "frequency",
    )
    return [dict(zip(cols, row, strict=True)) for row in con.execute(sql, params).fetchall()]


def as_of_history(
    con: duckdb.DuckDBPyConnection,
    as_of_ts: str,
    *,
    series_id: str | None = None,
) -> list[dict[str, Any]]:
    """All observation points known by ``as_of_ts``, PIT-revision-safe.

    For each ``(series_id, observation_date)`` keeps only the latest vintage
    whose ``available_for_signal_at`` was still ``<= as_of_ts``. A later revision
    that had not been released yet is invisible — the whole point of vintages.
    """
    params: list[Any] = [as_of_ts]
    series_filter = ""
    if series_id:
        series_filter = "AND series_id = ?"
        params.append(series_id)

    sql = f"""
        SELECT series_id, observation_date, value, release_timestamp,
               available_for_signal_at, vintage, source, frequency
        FROM macro_series
        WHERE available_for_signal_at <= ?
          {series_filter}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY series_id, observation_date
            ORDER BY available_for_signal_at DESC, vintage DESC
        ) = 1
        ORDER BY series_id, observation_date
    """
    cols = (
        "series_id", "observation_date", "value", "release_timestamp",
        "available_for_signal_at", "vintage", "source", "frequency",
    )
    return [dict(zip(cols, row, strict=True)) for row in con.execute(sql, params).fetchall()]
