"""Keyless FRED / ALFRED adapter — release vintages into macro_series.

No API key. Pulls public CSV endpoints:

  * Current release: ``fred.stlouisfed.org/graph/fredgraph.csv?id=...``
  * Historical vintage: ``alfred.stlouisfed.org/graph/alfredgraph.csv?id=...&vintage_date=...``

Network I/O lives behind a ``Transport`` callable (``url -> response text``) so
unit tests inject fixture payloads and never hit the wire.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable, Sequence
from urllib.parse import urlencode

import requests

# url -> response body (text). Swap in tests.
Transport = Callable[[str], str]

FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ALFREDGRAPH_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
_UA = "mira-macro/0.1 (research; keyless fredgraph/alfredgraph)"

# Default basket — same economic coverage the stock panel expects, but owned here.
DEFAULT_FRED_SERIES: dict[str, str] = {
    "VIXCLS": "daily",
    "DGS10": "daily",
    "T10Y2Y": "daily",
    "DFF": "daily",
    "DEXUSEU": "daily",
    "UNRATE": "monthly",
    "CPIAUCSL": "monthly",
}

# When no ALFRED vintage is supplied, assume the print is usable the next calendar day.
_DEFAULT_PUBLICATION_LAG_DAYS = 1


def default_transport(url: str, *, timeout: float = 45.0) -> str:
    """Live HTTP GET — the only place requests talks to FRED/ALFRED."""
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": _UA, "Accept": "text/csv,text/plain,*/*"},
    )
    resp.raise_for_status()
    return resp.text


def fredgraph_url(series_id: str) -> str:
    return f"{FREDGRAPH_URL}?{urlencode({'id': series_id})}"


def alfredgraph_url(series_id: str, vintage_date: str) -> str:
    return f"{ALFREDGRAPH_URL}?{urlencode({'id': series_id, 'vintage_date': vintage_date})}"


def _parse_csv_body(body: str) -> list[tuple[str, float]]:
    """Parse DATE,VALUE CSV (fredgraph / single-vintage alfredgraph).

    Skips missing markers (``.``) and blank cells. Returns ``(YYYY-MM-DD, value)``.
    """
    text = (body or "").strip()
    if not text:
        return []
    reader = csv.reader(io.StringIO(text))
    rows_iter = iter(reader)
    try:
        header = next(rows_iter)
    except StopIteration:
        return []

    # Tolerate BOM / alternate header labels.
    header_l = [h.strip().lstrip("\ufeff").upper() for h in header]
    date_idx = 0
    value_idx = 1 if len(header_l) > 1 else None
    for i, name in enumerate(header_l):
        if name in ("DATE", "OBSERVATION_DATE", "TIME", "TIME_PERIOD"):
            date_idx = i
        elif name not in ("DATE", "OBSERVATION_DATE", "TIME", "TIME_PERIOD") and value_idx is None:
            value_idx = i
    if value_idx is None:
        value_idx = 1 if len(header) > 1 else 0

    out: list[tuple[str, float]] = []
    for row in rows_iter:
        if len(row) <= max(date_idx, value_idx):
            continue
        raw_date = (row[date_idx] or "").strip()
        raw_val = (row[value_idx] or "").strip()
        if not raw_date or raw_val in ("", ".", "NA", "NaN"):
            continue
        try:
            # Normalize to ISO date.
            d = datetime.strptime(raw_date[:10], "%Y-%m-%d").date()
            val = float(raw_val)
        except ValueError:
            continue
        out.append((d.isoformat(), val))
    return out


def _in_window(obs: str, start: str | None, end: str | None) -> bool:
    if start and obs < start[:10]:
        return False
    if end and obs > end[:10]:
        return False
    return True


def _avail_from_lag(obs_date: str, lag_days: int = _DEFAULT_PUBLICATION_LAG_DAYS) -> str:
    d = datetime.strptime(obs_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (d + timedelta(days=lag_days)).strftime("%Y-%m-%dT00:00:00Z")


def _avail_from_vintage(vintage_date: str) -> str:
    """ALFRED vintage date is the day that release was public → available then."""
    d = datetime.strptime(vintage_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return d.strftime("%Y-%m-%dT00:00:00Z")


def observations_to_rows(
    series_id: str,
    observations: Iterable[tuple[str, float]],
    *,
    vintage: str,
    available_for_signal_at: str | None = None,
    frequency: str | None = None,
    start: str | None = None,
    end: str | None = None,
    publication_lag_days: int = _DEFAULT_PUBLICATION_LAG_DAYS,
) -> list[dict]:
    """Normalize ``(observation_date, value)`` pairs into macro_series row dicts."""
    freq = frequency or DEFAULT_FRED_SERIES.get(series_id, "unknown")
    rows: list[dict] = []
    for obs_date, value in observations:
        if not _in_window(obs_date, start, end):
            continue
        if vintage == "current":
            avail = available_for_signal_at or _avail_from_lag(obs_date, publication_lag_days)
            release_ts = avail
            vintage_label = "current"
        else:
            avail = available_for_signal_at or _avail_from_vintage(vintage)
            release_ts = avail
            vintage_label = vintage[:10]
        rows.append(
            {
                "series_id": series_id,
                "observation_date": obs_date[:10],
                "value": value,
                "release_timestamp": release_ts,
                "available_for_signal_at": avail,
                "vintage": vintage_label,
                "source": "fred",
                "frequency": freq,
            }
        )
    return rows


def fetch_fred_series(
    series_id: str,
    *,
    vintage_dates: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    transport: Transport | None = None,
    frequency: str | None = None,
    publication_lag_days: int = _DEFAULT_PUBLICATION_LAG_DAYS,
) -> list[dict]:
    """Fetch one series; optional ALFRED vintages become ``available_for_signal_at``.

    * No ``vintage_dates`` → current fredgraph pull, vintage=``current``,
      available = observation_date + publication lag.
    * With ``vintage_dates`` → one alfredgraph pull per vintage; each vintage date
      stamps ``available_for_signal_at`` (what was known that day).
    """
    get = transport or default_transport
    all_rows: list[dict] = []

    if not vintage_dates:
        body = get(fredgraph_url(series_id))
        obs = _parse_csv_body(body)
        all_rows.extend(
            observations_to_rows(
                series_id,
                obs,
                vintage="current",
                frequency=frequency,
                start=start,
                end=end,
                publication_lag_days=publication_lag_days,
            )
        )
        return all_rows

    for vdate in vintage_dates:
        body = get(alfredgraph_url(series_id, vdate[:10]))
        obs = _parse_csv_body(body)
        all_rows.extend(
            observations_to_rows(
                series_id,
                obs,
                vintage=vdate[:10],
                frequency=frequency,
                start=start,
                end=end,
            )
        )
    return all_rows


def fetch_fred_basket(
    series_ids: Sequence[str] | None = None,
    *,
    vintage_dates: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    transport: Transport | None = None,
) -> list[dict]:
    """Fetch many series (default basket) into store-ready rows."""
    ids = list(series_ids) if series_ids else list(DEFAULT_FRED_SERIES)
    rows: list[dict] = []
    for sid in ids:
        rows.extend(
            fetch_fred_series(
                sid,
                vintage_dates=vintage_dates,
                start=start,
                end=end,
                transport=transport,
                frequency=DEFAULT_FRED_SERIES.get(sid),
            )
        )
    return rows


def month_end_vintages(start: str, end: str) -> list[str]:
    """Convenience: one vintage date per month-end in ``[start, end]`` (inclusive)."""
    d0 = datetime.strptime(start[:10], "%Y-%m-%d").date()
    d1 = datetime.strptime(end[:10], "%Y-%m-%d").date()
    out: list[str] = []
    # Walk month by month; vintage = last calendar day of each month in range.
    y, m = d0.year, d0.month
    while date(y, m, 1) <= d1:
        if m == 12:
            last = date(y, 12, 31)
            y, m = y + 1, 1
        else:
            last = date(y, m + 1, 1) - timedelta(days=1)
            m += 1
        if d0 <= last <= d1:
            out.append(last.isoformat())
    return out
