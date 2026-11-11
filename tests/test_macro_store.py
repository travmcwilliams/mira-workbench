"""Offline tests for the point-in-time macro store + FRED/ECB adapters.

No network: fixtures + transport seams only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

_ACQUIRE = Path(__file__).resolve().parents[1] / "acquire"
if str(_ACQUIRE) not in sys.path:
    sys.path.insert(0, str(_ACQUIRE))

from macro.ecb import (  # noqa: E402
    DEFAULT_ECB_CURRENCIES,
    fetch_ecb_fx,
    series_id_for,
)
from macro.ecb import (  # noqa: E402
    observations_to_rows as ecb_obs_to_rows,
)
from macro.fred import (  # noqa: E402
    DEFAULT_FRED_SERIES,
    alfredgraph_url,
    fetch_fred_basket,
    fetch_fred_series,
    fredgraph_url,
)
from macro.fred import (  # noqa: E402
    observations_to_rows as fred_obs_to_rows,
)
from macro.store import (  # noqa: E402
    as_of,
    as_of_history,
    list_series,
    open_db,
    row_count,
    upsert_rows,
)

# --------------------------------------------------------------------------- #
# fixtures (wire-shape CSV bodies)
# --------------------------------------------------------------------------- #

FRED_CSV = """DATE,UNRATE
2024-01-01,3.7
2024-02-01,3.9
2024-03-01,.
2024-04-01,3.8
"""

# ALFRED vintage 2024-02-02: first release of Jan (3.7), no Feb yet
ALFRED_V1 = """DATE,UNRATE
2024-01-01,3.7
"""

# ALFRED vintage 2024-03-08: Jan revised to 3.8, Feb first print 3.9
ALFRED_V2 = """DATE,UNRATE
2024-01-01,3.8
2024-02-01,3.9
"""

ECB_CSV = """KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-02,1.0950
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-03,1.0920
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-04,.
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-05,1.0900
"""


@pytest.fixture
def con(tmp_path):
    db = tmp_path / "macro.duckdb"
    c = open_db(db)
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# store: upsert + idempotency
# --------------------------------------------------------------------------- #

def test_upsert_idempotent(con):
    rows = fred_obs_to_rows(
        "UNRATE",
        [("2024-01-01", 3.7), ("2024-02-01", 3.9)],
        vintage="current",
        frequency="monthly",
    )
    n1 = upsert_rows(con, rows)
    assert n1 == 2
    assert row_count(con) == 2

    # same payload again — row count stays 2
    n2 = upsert_rows(con, rows)
    assert n2 == 2
    assert row_count(con) == 2

    # value update for same vintage overwrites
    revised = fred_obs_to_rows(
        "UNRATE",
        [("2024-01-01", 3.75)],
        vintage="current",
        frequency="monthly",
    )
    upsert_rows(con, revised)
    assert row_count(con) == 2
    hist = as_of_history(con, "2099-01-01", series_id="UNRATE")
    jan = [r for r in hist if str(r["observation_date"])[:10] == "2024-01-01"][0]
    assert jan["value"] == pytest.approx(3.75)


def test_list_series_summary(con):
    upsert_rows(
        con,
        fred_obs_to_rows("DGS10", [("2024-01-02", 4.0)], vintage="current", frequency="daily"),
    )
    upsert_rows(
        con,
        fred_obs_to_rows("DGS10", [("2024-01-03", 4.1)], vintage="current", frequency="daily"),
    )
    summary = list_series(con)
    assert len(summary) == 1
    assert summary[0]["series_id"] == "DGS10"
    assert summary[0]["n_rows"] == 2
    assert summary[0]["source"] == "fred"


def test_upsert_rejects_missing_fields(con):
    with pytest.raises(ValueError, match="missing required"):
        upsert_rows(con, [{"series_id": "X", "value": 1.0}])


# --------------------------------------------------------------------------- #
# vintage / as-of semantics (the leak-safety core)
# --------------------------------------------------------------------------- #

def test_asof_hides_future_vintages(con):
    """Revised value must not leak before its ALFRED vintage date."""
    v1 = fred_obs_to_rows(
        "UNRATE",
        [("2024-01-01", 3.7)],
        vintage="2024-02-02",
        frequency="monthly",
    )
    v2 = fred_obs_to_rows(
        "UNRATE",
        [("2024-01-01", 3.8), ("2024-02-01", 3.9)],
        vintage="2024-03-08",
        frequency="monthly",
    )
    upsert_rows(con, v1 + v2)
    assert row_count(con) == 3  # two Jan vintages + one Feb

    # Before any release: empty
    assert as_of(con, "2024-01-15") == []

    # After first release, before revision: Jan = 3.7
    early = as_of(con, "2024-02-15")
    assert len(early) == 1
    assert early[0]["value"] == pytest.approx(3.7)
    assert early[0]["vintage"] == "2024-02-02"

    # After second release: Jan revised to 3.8 (latest available per series)
    late = as_of(con, "2024-03-15")
    assert len(late) == 1
    # as_of picks latest observation among latest available vintages
    # Feb 2024 obs has available 2024-03-08 and is more recent obs → 3.9
    assert late[0]["value"] == pytest.approx(3.9)
    assert str(late[0]["observation_date"])[:10] == "2024-02-01"


def test_asof_history_revision_safe(con):
    """History as-of a mid date keeps the unrevised Jan print."""
    upsert_rows(
        con,
        fred_obs_to_rows("UNRATE", [("2024-01-01", 3.7)], vintage="2024-02-02")
        + fred_obs_to_rows(
            "UNRATE",
            [("2024-01-01", 3.8), ("2024-02-01", 3.9)],
            vintage="2024-03-08",
        ),
    )
    mid = as_of_history(con, "2024-02-15", series_id="UNRATE")
    assert len(mid) == 1
    assert mid[0]["value"] == pytest.approx(3.7)

    full = as_of_history(con, "2024-03-15", series_id="UNRATE")
    by_obs = {str(r["observation_date"])[:10]: r["value"] for r in full}
    assert by_obs["2024-01-01"] == pytest.approx(3.8)  # revision now visible
    assert by_obs["2024-02-01"] == pytest.approx(3.9)


# --------------------------------------------------------------------------- #
# FRED adapter (fixture transport)
# --------------------------------------------------------------------------- #

def test_fred_current_via_transport():
    def transport(url: str) -> str:
        assert "fredgraph.csv" in url
        assert "UNRATE" in url
        return FRED_CSV

    rows = fetch_fred_series("UNRATE", transport=transport, start="2024-01-01", end="2024-04-30")
    # 3 valid values (`.` skipped)
    assert len(rows) == 3
    assert all(r["source"] == "fred" for r in rows)
    assert all(r["vintage"] == "current" for r in rows)
    # publication lag: obs 2024-01-01 → available 2024-01-02
    jan = next(r for r in rows if r["observation_date"] == "2024-01-01")
    assert jan["available_for_signal_at"].startswith("2024-01-02")


def test_fred_alfred_vintages_via_transport():
    def transport(url: str) -> str:
        if "vintage_date=2024-02-02" in url:
            return ALFRED_V1
        if "vintage_date=2024-03-08" in url:
            return ALFRED_V2
        raise AssertionError(f"unexpected url: {url}")

    rows = fetch_fred_series(
        "UNRATE",
        vintage_dates=["2024-02-02", "2024-03-08"],
        transport=transport,
    )
    assert len(rows) == 3
    # vintage stamps available_for_signal_at
    v1 = [r for r in rows if r["vintage"] == "2024-02-02"]
    assert len(v1) == 1
    assert v1[0]["available_for_signal_at"].startswith("2024-02-02")
    assert v1[0]["value"] == pytest.approx(3.7)


def test_fred_basket_default_ids():
    seen: list[str] = []

    def transport(url: str) -> str:
        # extract id= from query
        for part in url.split("?")[-1].split("&"):
            if part.startswith("id="):
                seen.append(part.split("=", 1)[1])
        return "DATE,X\n2024-01-01,1.0\n"

    rows = fetch_fred_basket(transport=transport)
    assert set(seen) == set(DEFAULT_FRED_SERIES)
    assert len(rows) == len(DEFAULT_FRED_SERIES)


def test_fred_url_builders():
    assert "fredgraph.csv" in fredgraph_url("DGS10")
    assert "id=DGS10" in fredgraph_url("DGS10")
    u = alfredgraph_url("UNRATE", "2024-02-02")
    assert "alfredgraph.csv" in u
    assert "vintage_date=2024-02-02" in u


# --------------------------------------------------------------------------- #
# ECB adapter (fixture transport)
# --------------------------------------------------------------------------- #

def test_ecb_fx_via_transport():
    def transport(url: str) -> str:
        assert "EXR/D.USD.EUR.SP00.A" in url
        assert "format=csvdata" in url
        return ECB_CSV

    rows = fetch_ecb_fx(["USD"], transport=transport)
    # 3 valid (`.` skipped)
    assert len(rows) == 3
    assert all(r["series_id"] == "ECB_USD" for r in rows)
    assert all(r["source"] == "ecb" for r in rows)
    assert all(r["frequency"] == "daily" for r in rows)
    # same-day 15:00 UTC availability
    assert rows[0]["available_for_signal_at"] == "2024-01-02T15:00:00Z"
    assert rows[0]["value"] == pytest.approx(1.0950)


def test_ecb_into_store_and_asof(con):
    rows = ecb_obs_to_rows(
        "USD",
        [("2024-01-02", 1.095), ("2024-01-03", 1.092)],
    )
    upsert_rows(con, rows)
    # morning of 2024-01-02 (before 15:00 UTC) must not see the print
    assert as_of(con, "2024-01-02T12:00:00Z", series_ids=["ECB_USD"]) == []
    after = as_of(con, "2024-01-02T16:00:00Z", series_ids=["ECB_USD"])
    assert len(after) == 1
    assert after[0]["value"] == pytest.approx(1.095)


def test_ecb_series_id_helper():
    assert series_id_for("usd") == "ECB_USD"
    assert "USD" in DEFAULT_ECB_CURRENCIES


# --------------------------------------------------------------------------- #
# end-to-end: adapters → store → as-of (still offline)
# --------------------------------------------------------------------------- #

def test_fred_vintages_end_to_end(con):
    def transport(url: str) -> str:
        if "vintage_date=2024-02-02" in url:
            return ALFRED_V1
        if "vintage_date=2024-03-08" in url:
            return ALFRED_V2
        raise AssertionError(url)

    rows = fetch_fred_series(
        "UNRATE",
        vintage_dates=["2024-02-02", "2024-03-08"],
        transport=transport,
    )
    upsert_rows(con, rows)

    mid_hist = as_of_history(con, "2024-02-20", series_id="UNRATE")
    assert len(mid_hist) == 1
    assert mid_hist[0]["value"] == pytest.approx(3.7)

    late_hist = as_of_history(con, "2024-03-20", series_id="UNRATE")
    by_obs = {str(r["observation_date"])[:10]: r["value"] for r in late_hist}
    assert by_obs["2024-01-01"] == pytest.approx(3.8)
    assert by_obs["2024-02-01"] == pytest.approx(3.9)
