"""Keyless ECB daily FX reference-rates adapter (EUR crosses).

Pulls the public ECB Data Portal SDMX endpoint (no API key):

    https://data-api.ecb.europa.eu/service/data/EXR/D.{CCY}.EUR.SP00.A

Values are *foreign currency units per one euro* (ECB convention). Stored under
``series_id = ECB_{CCY}`` (e.g. ``ECB_USD``) with ``source=ecb``, ``frequency=daily``.

ECB publishes reference rates around 16:00 CET on the observation day; we stamp
``available_for_signal_at`` at 15:00 UTC that day (CET≈UTC+1 winter / +2 summer
is approximated — conservative enough that a morning-of signal never sees them).

Network I/O is isolated behind a ``Transport`` seam for offline tests.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Callable, Sequence
from urllib.parse import urlencode

import requests

Transport = Callable[[str], str]

ECB_DATA_URL = "https://data-api.ecb.europa.eu/service/data/EXR"
_UA = "mira-macro/0.1 (research; keyless ECB EXR)"

# Common EUR crosses used in macro panels (USD mirrors FRED DEXUSEU territory).
DEFAULT_ECB_CURRENCIES: tuple[str, ...] = (
    "USD",
    "GBP",
    "JPY",
    "CHF",
    "CNY",
)


def default_transport(url: str, *, timeout: float = 45.0) -> str:
    """Live HTTP GET — the only place requests talks to the ECB."""
    resp = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": _UA,
            "Accept": "text/csv",
        },
    )
    resp.raise_for_status()
    return resp.text


def series_id_for(currency: str) -> str:
    return f"ECB_{currency.upper()}"


def ecb_fx_url(
    currency: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> str:
    """Build the SDMX data URL for daily EUR/{currency} reference rates."""
    key = f"D.{currency.upper()}.EUR.SP00.A"
    params: dict[str, str] = {"format": "csvdata"}
    if start:
        params["startPeriod"] = start[:10]
    if end:
        params["endPeriod"] = end[:10]
    return f"{ECB_DATA_URL}/{key}?{urlencode(params)}"


def _parse_ecb_csv(body: str) -> list[tuple[str, float]]:
    """Parse ECB csvdata into ``(TIME_PERIOD, OBS_VALUE)`` pairs."""
    text = (body or "").strip()
    if not text:
        return []
    reader = csv.DictReader(io.StringIO(text))
    out: list[tuple[str, float]] = []
    for row in reader:
        # Keys vary slightly by Accept header / format flag; be defensive.
        period = (
            row.get("TIME_PERIOD")
            or row.get("time_period")
            or row.get("DATE")
            or ""
        ).strip()
        raw_val = (
            row.get("OBS_VALUE")
            or row.get("obs_value")
            or row.get("OBS_VALUE".lower())
            or ""
        ).strip()
        if not period or raw_val in ("", ".", "NA"):
            continue
        try:
            d = datetime.strptime(period[:10], "%Y-%m-%d").date()
            val = float(raw_val)
        except ValueError:
            continue
        out.append((d.isoformat(), val))
    return out


def _available_at(obs_date: str) -> str:
    """Same-day 15:00 UTC ≈ post-publication window for ECB ref rates."""
    d = datetime.strptime(obs_date[:10], "%Y-%m-%d").replace(
        hour=15, minute=0, second=0, tzinfo=timezone.utc
    )
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def observations_to_rows(
    currency: str,
    observations: Sequence[tuple[str, float]],
) -> list[dict]:
    """Normalize ECB observations into macro_series row dicts."""
    sid = series_id_for(currency)
    rows: list[dict] = []
    for obs_date, value in observations:
        avail = _available_at(obs_date)
        rows.append(
            {
                "series_id": sid,
                "observation_date": obs_date[:10],
                "value": value,
                "release_timestamp": avail,
                "available_for_signal_at": avail,
                "vintage": obs_date[:10],  # ECB daily print = its own vintage day
                "source": "ecb",
                "frequency": "daily",
            }
        )
    return rows


def fetch_ecb_fx(
    currencies: Sequence[str] | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    transport: Transport | None = None,
) -> list[dict]:
    """Fetch EUR crosses for ``currencies`` (default basket) as store-ready rows."""
    get = transport or default_transport
    ccys = list(currencies) if currencies else list(DEFAULT_ECB_CURRENCIES)
    rows: list[dict] = []
    for ccy in ccys:
        body = get(ecb_fx_url(ccy, start=start, end=end))
        obs = _parse_ecb_csv(body)
        rows.extend(observations_to_rows(ccy, obs))
    return rows
