"""Keyless FRED release-calendar adapter — release dates + names, no API key.

FRED's official API (``api.stlouisfed.org``) requires an ``api_key``; the public
site does not. Every release has a release-dates page

    https://fred.stlouisfed.org/release/dates?rid=<release_id>

listing its historical and scheduled dates. We fetch that HTML through an
INJECTABLE transport (``fetch`` — offline tests feed fixture bytes, the default
is a paced ``requests`` GET) and extract dates tolerantly: any ISO
``YYYY-MM-DD`` or ``Month D, YYYY`` token in the page body, validated as a real
calendar date, deduped and sorted. Tolerant-by-design means cosmetic markup
drift does not break the adapter; a structural change shows up as an obviously
wrong date count, which the ``--dry-run`` smoke CLI makes easy to eyeball:

    python acquire/events/fred_releases.py --release cpi --start 2026-01-01 --dry-run
    python acquire/events/fred_releases.py --db var/macro_events/macro_events.duckdb

Release ids are DATA (pack-style), not code — the ``RELEASES`` registry ships
the FOMC-adjacent core prints and extends by adding a line. The ids mirror
https://fred.stlouisfed.org/releases ; they were written from documentation and
NOT verified against the live site in an offline session, so spot-check the
name echoed back on the first live run (the page ``<title>`` is cross-checked
against the registry name and a mismatch is refused).

Calendar rows carry DATE precision only — ``release_datetime`` is stored at
midnight as a placeholder; intraday timestamps arrive with a values source
later, as successor rows (see ``events.store``).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Callable, Iterable, Mapping

if __package__ in (None, ""):  # direct script run: put acquire/ on sys.path first
    _ACQUIRE = str(Path(__file__).resolve().parents[1])
    if _ACQUIRE not in sys.path:
        sys.path.insert(0, _ACQUIRE)

from events import store  # noqa: E402

FetchFn = Callable[[str], str]

_URL = "https://fred.stlouisfed.org/release/dates?rid={rid}"
_UA = "mira-macro-research/0.1 (personal research)"
SOURCE = "fred_releases"


@dataclass(frozen=True)
class Release:
    rid: int
    name: str
    asset_classes: tuple[str, ...]


# Registry = data, not code. Keys double as the store's ``event_type``.
# ids per https://fred.stlouisfed.org/releases — verify on first live run.
RELEASES: dict[str, Release] = {
    "cpi": Release(10, "Consumer Price Index", ("equities", "rates", "fx")),
    "nfp": Release(50, "Employment Situation", ("equities", "rates", "fx")),
    "ppi": Release(46, "Producer Price Index", ("equities", "rates")),
    "gdp": Release(53, "Gross Domestic Product", ("equities", "rates", "fx")),
}


@dataclass(frozen=True)
class ReleaseCalendar:
    key: str                    # event_type in the store
    rid: int
    name: str                   # registry name, cross-checked vs the page title
    dates: tuple[str, ...]      # ISO, sorted, deduped
    asset_classes: tuple[str, ...]


# --------------------------------------------------------------------------- #
# transport seam
# --------------------------------------------------------------------------- #

def _default_fetch(url: str) -> str:
    import requests

    r = requests.get(url, timeout=30.0, headers={"User-Agent": _UA})
    r.raise_for_status()
    return r.text


# --------------------------------------------------------------------------- #
# tolerant parsing
# --------------------------------------------------------------------------- #

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9

_ISO_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_LONG_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"\.?\s+(\d{1,2}),\s*(20\d{2})\b")
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _valid_iso(year: int, month: int, day: int) -> str | None:
    try:
        return _date(year, month, day).isoformat()
    except ValueError:
        return None


def parse_release_dates(html: str) -> list[str]:
    """Every real calendar date found in the page, ISO, deduped, sorted."""
    found: set[str] = set()
    for y, m, d in _ISO_RE.findall(html):
        iso = _valid_iso(int(y), int(m), int(d))
        if iso:
            found.add(iso)
    for mon, d, y in _LONG_RE.findall(html):
        iso = _valid_iso(int(y), _MONTHS[mon.lower()], int(d))
        if iso:
            found.add(iso)
    return sorted(found)


def parse_page_name(html: str) -> str | None:
    """Best-effort release name from ``<title>`` (site-chrome segments skipped)."""
    m = _TITLE_RE.search(html)
    if not m:
        return None
    for part in m.group(1).split("|"):
        part = part.strip()
        low = part.lower()
        if not part or "fred" in low or "st. louis" in low or low == "release dates":
            continue
        return part
    return None


# --------------------------------------------------------------------------- #
# adapter
# --------------------------------------------------------------------------- #

def fetch_release_dates(
    release: str,
    *,
    start: str | None = None,
    end: str | None = None,
    fetch: FetchFn | None = None,
) -> ReleaseCalendar:
    """Fetch one release's calendar; ``start``/``end`` window is inclusive (ISO).

    Refuses a page whose title names a DIFFERENT release (guards against a
    stale/wrong registry id silently mislabeling events).
    """
    if release not in RELEASES:
        raise KeyError(
            f"unknown release {release!r}; registry has {sorted(RELEASES)} — "
            "add a line to RELEASES (ids at https://fred.stlouisfed.org/releases)")
    reg = RELEASES[release]
    html = (fetch or _default_fetch)(_URL.format(rid=reg.rid))
    page_name = parse_page_name(html)
    if page_name and reg.name.lower() not in page_name.lower() \
            and page_name.lower() not in reg.name.lower():
        raise ValueError(
            f"rid={reg.rid} page is titled {page_name!r}, expected {reg.name!r} — "
            "registry id looks wrong; verify at https://fred.stlouisfed.org/releases")
    dates = [d for d in parse_release_dates(html)
             if (start is None or d >= start) and (end is None or d <= end)]
    return ReleaseCalendar(release, reg.rid, reg.name, tuple(dates), reg.asset_classes)


def to_event_rows(cal: ReleaseCalendar, *, country: str = "US") -> list[dict[str, object]]:
    """Store-ready rows (calendar precision: values NULL, midnight timestamps)."""
    return [
        {
            "event_id": store.event_id(SOURCE, cal.key, country, f"{d} 00:00:00"),
            "event_type": cal.key,
            "release_datetime": f"{d} 00:00:00",
            "country": country,
            "actual": None,
            "consensus": None,
            "previous": None,
            "surprise": None,
            "source": SOURCE,
            "asset_classes": list(cal.asset_classes),
        }
        for d in cal.dates
    ]


def ingest_calendar(
    con,
    releases: Iterable[str] = tuple(RELEASES),
    *,
    start: str | None = None,
    end: str | None = None,
    fetch: FetchFn | None = None,
) -> Mapping[str, dict[str, int]]:
    """Fetch + idempotently ingest each release's calendar; per-release stats."""
    stats: dict[str, dict[str, int]] = {}
    for key in releases:
        cal = fetch_release_dates(key, start=start, end=end, fetch=fetch)
        new = store.ingest_events(con, to_event_rows(cal))
        stats[key] = {"dates": len(cal.dates), "new": new}
    return stats


# --------------------------------------------------------------------------- #
# smoke CLI (the human's live-verification path)
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Keyless FRED release-calendar ingest (macro events store)")
    ap.add_argument("--release", action="append", choices=sorted(RELEASES),
                    help="repeatable; default: all registered releases")
    ap.add_argument("--start", default=None, help="ISO date, inclusive")
    ap.add_argument("--end", default=None, help="ISO date, inclusive")
    ap.add_argument("--db", default=store.DEFAULT_DB)
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + parse + print only; touch no database")
    a = ap.parse_args(argv)

    keys = a.release or sorted(RELEASES)
    if a.dry_run:
        for key in keys:
            cal = fetch_release_dates(key, start=a.start, end=a.end)
            head = ", ".join(cal.dates[:5])
            tail = f" .. {cal.dates[-1]}" if len(cal.dates) > 5 else ""
            print(f"[{key}] rid={cal.rid} {cal.name!r}: "
                  f"{len(cal.dates)} dates  {head}{tail}")
        return 0

    con = store.open_db(a.db)
    try:
        stats = ingest_calendar(con, keys, start=a.start, end=a.end)
    finally:
        con.close()
    for key, s in stats.items():
        print(f"[{key}] {s['dates']} dates in window, {s['new']} new rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
