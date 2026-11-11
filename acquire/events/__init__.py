"""mira macro-event acquisition — structured releases as first-class citizens.

The news package reconstructs a *continuous* daily sentiment series; this package
holds the *discrete* counterpart: scheduled macro events (CPI, NFP, GDP prints,
FOMC-adjacent releases) as dated, typed rows a study can condition on.

Two modules, mirroring the news layout:

    events.store          — dedicated append-only DuckDB (idempotent ingest;
                            NOT the demo ``acquire/db/app.duckdb``)
    events.fred_releases  — keyless FRED release-calendar adapter behind an
                            injectable transport seam (offline-testable)

Joining events onto returns is deliberately NOT done here — that lives in
``core/timejoin.py`` (shared leak-safe date infrastructure) and
``core/event_study.py`` (the generalized study runner).
"""

from __future__ import annotations

__all__ = ["fred_releases", "store"]
