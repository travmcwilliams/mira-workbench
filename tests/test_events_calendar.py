"""Offline tests for the macro-events store + keyless FRED calendar adapter.

No network: the FRED transport seam is exercised with fixture HTML; the store
runs against a tmp DuckDB. Mirrors the test_news_wayback.py layout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# acquire/ is a sys.path root (mira convention), so `events` imports as top-level.
_ACQUIRE = Path(__file__).resolve().parents[1] / "acquire"
if str(_ACQUIRE) not in sys.path:
    sys.path.insert(0, str(_ACQUIRE))

from events import fred_releases, store  # noqa: E402

# --- store: append-only, idempotent ------------------------------------------

ROW = {
    "event_type": "cpi",
    "release_datetime": "2026-06-10 00:00:00",
    "country": "US",
    "actual": 2.0,
    "consensus": 1.5,
    "previous": 1.8,
    "source": "test",
    "asset_classes": ["equities", "rates"],
}


@pytest.fixture()
def con(tmp_path):
    con = store.open_db(tmp_path / "events.duckdb")
    yield con
    con.close()


def test_ingest_is_idempotent(con):
    assert store.ingest_events(con, [ROW]) == 1
    assert store.ingest_events(con, [ROW]) == 0          # replay: no-op
    assert store.event_counts(con) == {"cpi": 1}


def test_surprise_computed_when_absent(con):
    store.ingest_events(con, [ROW])
    df = store.fetch_events(con)
    assert df.loc[0, "surprise"] == pytest.approx(0.5)   # actual - consensus


def test_explicit_surprise_wins(con):
    row = {**ROW, "surprise": -9.0}
    store.ingest_events(con, [row])
    assert store.fetch_events(con).loc[0, "surprise"] == pytest.approx(-9.0)


def test_calendar_only_row_has_null_values(con):
    row = {"event_type": "nfp", "release_datetime": "2026-07-02 00:00:00",
           "source": "test"}
    store.ingest_events(con, [row])
    df = store.fetch_events(con)
    assert pd.isna(df.loc[0, "actual"]) and pd.isna(df.loc[0, "surprise"])
    assert df.loc[0, "country"] == "US"                  # default


def test_revision_is_a_successor_not_an_update(con):
    store.ingest_events(con, [{k: v for k, v in ROW.items() if k != "actual"}])
    successor = {**ROW, "revision": 1}
    assert store.ingest_events(con, [successor]) == 1    # new row, not clobber
    df = store.fetch_events(con, event_type="cpi")
    assert len(df) == 2
    assert df["revision"].tolist() == [0, 1]
    first = df[df["revision"] == 0].iloc[0]
    assert pd.isna(first["actual"])                      # original row untouched


def test_fetch_events_filters_and_date_column(con):
    rows = [
        ROW,
        {**ROW, "release_datetime": "2026-07-15 00:00:00"},
        {**ROW, "event_type": "nfp", "release_datetime": "2026-07-02 00:00:00"},
    ]
    assert store.ingest_events(con, rows) == 3
    df = store.fetch_events(con, event_type="cpi", start="2026-07-01", end="2026-07-31")
    assert len(df) == 1
    assert df.loc[0, "date"] == "2026-07-15"
    assert json.loads(df.loc[0, "asset_classes"]) == ["equities", "rates"]


def test_event_id_deterministic():
    a = store.event_id("s", "cpi", "US", "2026-06-10 00:00:00")
    b = store.event_id("s", "cpi", "US", "2026-06-10 00:00:00")
    c = store.event_id("s", "cpi", "US", "2026-06-10 00:00:00", revision=1)
    assert a == b and a != c


# --- FRED adapter: fixture HTML through the transport seam ---------------------

CPI_HTML = """<!doctype html>
<html><head>
<title>Consumer Price Index | Release Dates | FRED | St. Louis Fed</title>
</head><body>
<h1>Release Dates: Consumer Price Index</h1>
<table>
  <tr><td>June 10, 2026</td></tr>
  <tr><td>May 13, 2026</td></tr>
  <tr><td>Sept. 11, 2026</td></tr>
  <tr><td>Sep 12, 2025</td></tr>
  <tr><td>2026-04-10</td></tr>
  <tr><td>June 31, 2026</td></tr>   <!-- impossible date: must be skipped -->
</table>
<p>Copyright 2026 — some chrome text with a bare year.</p>
</body></html>"""


def test_parse_release_dates_tolerant_and_sorted():
    got = fred_releases.parse_release_dates(CPI_HTML)
    assert got == ["2025-09-12", "2026-04-10", "2026-05-13",
                   "2026-06-10", "2026-09-11"]


def test_parse_page_name_skips_site_chrome():
    assert fred_releases.parse_page_name(CPI_HTML) == "Consumer Price Index"
    assert fred_releases.parse_page_name("<html>no title</html>") is None


def test_fetch_release_dates_windows_and_names():
    cal = fred_releases.fetch_release_dates(
        "cpi", start="2026-01-01", end="2026-06-30",
        fetch=lambda url: CPI_HTML)
    assert cal.key == "cpi" and cal.rid == 10
    assert cal.name == "Consumer Price Index"
    assert cal.dates == ("2026-04-10", "2026-05-13", "2026-06-10")


def test_fetch_refuses_mistitled_page():
    wrong = CPI_HTML.replace("Consumer Price Index", "Producer Price Index")
    with pytest.raises(ValueError, match="registry id looks wrong"):
        fred_releases.fetch_release_dates("cpi", fetch=lambda url: wrong)


def test_unknown_release_is_a_helpful_keyerror():
    with pytest.raises(KeyError, match="unknown release"):
        fred_releases.fetch_release_dates("fomc_minutes", fetch=lambda url: CPI_HTML)


def test_to_event_rows_are_store_ready():
    cal = fred_releases.fetch_release_dates(
        "cpi", start="2026-06-01", end="2026-06-30", fetch=lambda url: CPI_HTML)
    rows = fred_releases.to_event_rows(cal)
    assert len(rows) == 1
    r = rows[0]
    assert r["event_type"] == "cpi"
    assert r["release_datetime"] == "2026-06-10 00:00:00"
    assert r["actual"] is None and r["consensus"] is None
    assert r["source"] == "fred_releases"
    assert r["event_id"] == store.event_id(
        "fred_releases", "cpi", "US", "2026-06-10 00:00:00")


def test_ingest_calendar_end_to_end_idempotent(con):
    stats = fred_releases.ingest_calendar(
        con, ["cpi"], start="2026-01-01", end="2026-12-31",
        fetch=lambda url: CPI_HTML)
    assert stats["cpi"] == {"dates": 4, "new": 4}
    again = fred_releases.ingest_calendar(
        con, ["cpi"], start="2026-01-01", end="2026-12-31",
        fetch=lambda url: CPI_HTML)
    assert again["cpi"] == {"dates": 4, "new": 0}        # replay: no-op
    assert store.event_counts(con) == {"cpi": 4}
