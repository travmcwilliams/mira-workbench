"""Kalshi daily-candlestick backfill: census CSV -> kalshi.duckdb probability series.

Scope: macro EVENT categories only (Economics, Elections, Politics, World,
Companies, Science and Technology, Health, Transportation). Crypto, Financials,
Climate and Commodities are excluded as price-ladder families — logged, not silent.
Coverage is effectively 2025+ (the legacy pre-migration host needs an API key).

Writes to its own DuckDB file (kalshi.duckdb) so it can run concurrently with a
Polymarket backfill — DuckDB is single-writer per file. Resumes via fetch_log.

    python acquire/predmarkets/kalshi.py --min-contracts 100
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from store import connect, done_market_ids  # noqa: E402

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
_UA = {"User-Agent": "mira-predmarkets-research (personal research)"}
CENSUS = Path(__file__).resolve().parents[2] / "data" / "predmarkets" / "probe_kalshi.csv"
DB = Path(__file__).resolve().parents[2] / "var" / "predmarkets" / "kalshi.duckdb"

EVENT_CATEGORIES = [
    "Economics", "Elections", "Politics", "World", "Companies",
    "Science and Technology", "Health", "Transportation",
]

_SESSION = requests.Session()
_SESSION.headers.update(_UA)


def _get(url: str, params, tries: int = 5, timeout_s: float = 30.0) -> requests.Response:
    for attempt in range(tries):
        r = _SESSION.get(url, params=params, timeout=timeout_s)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
            continue
        return r
    return r


def select_markets(min_contracts: float) -> pd.DataFrame:
    df = pd.read_csv(CENSUS, low_memory=False)
    df = df[df["status"] != "empty_series"]
    n_all = len(df)
    df = df[df["category"].isin(EVENT_CATEGORIES)].copy()
    vol = pd.to_numeric(df["volume_contracts"], errors="coerce").fillna(0)
    kept = df[vol >= min_contracts].copy()
    print(f"[kalshi] {len(kept)} markets selected "
          f"(of {n_all} censused; ladder categories excluded, "
          f"volume floor {min_contracts:g} contracts)", file=sys.stderr)
    kept["market_id"] = kept["market_id"].astype(str)
    return kept


def stage_prices(con, markets: pd.DataFrame, pace_s: float = 0.15) -> None:
    done = done_market_ids(con, "kalshi", "prices")
    todo = markets[~markets["market_id"].isin(done)]
    print(f"[prices] {len(todo)} markets to fetch ({len(done)} done)", file=sys.stderr)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    for n, m in enumerate(todo.itertuples(index=False)):
        try:
            start = int(pd.Timestamp(m.created_at).timestamp())
            end = min(int(pd.Timestamp(m.end_date).timestamp()), now_ts)
        except (ValueError, TypeError):
            con.execute("INSERT INTO fetch_log VALUES ('kalshi', ?, 'prices', 'bad_dates', 0, '', current_timestamp)",
                        [m.market_id])
            continue
        r = _get(f"{KALSHI}/series/{m.series_ticker}/markets/{m.market_id}/candlesticks",
                 {"start_ts": start, "end_ts": end, "period_interval": 1440})
        if r.status_code != 200:
            con.execute("INSERT INTO fetch_log VALUES ('kalshi', ?, 'prices', 'http_error', 0, ?, current_timestamp)",
                        [m.market_id, f"status={r.status_code}"])
            time.sleep(pace_s)
            continue
        candles = r.json().get("candlesticks", [])
        rows = []
        for c in candles:
            close = (c.get("price") or {}).get("close_dollars")
            if close is None:
                continue
            ts = int(c["end_period_ts"])
            rows.append((m.market_id, ts,
                         datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
                         float(close)))
        if rows:
            con.executemany(
                "INSERT INTO prices VALUES ('kalshi', ?, ?, ?, ?, current_timestamp)", rows)
        con.execute("INSERT INTO fetch_log VALUES ('kalshi', ?, 'prices', 'ok', ?, '', current_timestamp)",
                    [m.market_id, len(rows)])
        if (n + 1) % 500 == 0:
            print(f"[prices] {n + 1}/{len(todo)}", file=sys.stderr)
        time.sleep(pace_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kalshi candlestick backfill (checkpointed)")
    ap.add_argument("--min-contracts", type=float, default=100)
    a = ap.parse_args(argv)

    con = connect(DB)
    markets = select_markets(a.min_contracts)
    done = done_market_ids(con, "kalshi", "meta")
    fresh = markets[~markets["market_id"].isin(done)]
    for m in fresh.itertuples(index=False):
        con.execute("INSERT INTO markets VALUES ('kalshi', ?, ?, ?, ?, ?, ?, NULL, NULL, current_timestamp)",
                    [m.market_id, m.question, m.category, m.end_date,
                     m.status == "finalized",
                     float(pd.to_numeric(m.volume_contracts, errors="coerce") or 0)])
        con.execute("INSERT INTO fetch_log VALUES ('kalshi', ?, 'meta', 'ok', 1, '', current_timestamp)",
                    [m.market_id])
    stage_prices(con, markets)

    stats = con.execute("""
        SELECT stage, status, count(*) FROM fetch_log
        WHERE platform='kalshi' GROUP BY 1,2 ORDER BY 1,2""").fetchall()
    n_prices = con.execute("SELECT count(*) FROM prices").fetchone()[0]
    print("[kalshi] fetch_log:", stats)
    print(f"[kalshi] price points stored: {n_prices:,}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
