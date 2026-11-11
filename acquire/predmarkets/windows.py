"""Minute-fidelity event-window fetcher (Polymarket) — on-demand, not a campaign.

Blanket minute backfill would be billions of rows for probabilities that mostly
sit still; the research value of minute data is concentrated in event windows
(CPI prints, FOMC statements, election nights). This tool takes explicit
(market, event-time) requests and pulls CLOB fidelity=1 series around them.

Self-contained: market tokens are looked up from Gamma directly (no dependency
on the campaign DuckDB, which may be write-locked by a running backfill) and
rows land in their own minute_windows.duckdb.

    python acquire/predmarkets/windows.py --market-id 240381 \
        --event-ts 2024-01-15T23:00:00Z --before-min 360 --after-min 360
    python acquire/predmarkets/windows.py --requests-csv my_windows.csv
        # columns: market_id, event_ts[, before_min, after_min]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from store import connect  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
_UA = {"User-Agent": "mira-predmarkets-research (personal research)"}
DB = Path(__file__).resolve().parents[2] / "var" / "predmarkets" / "minute_windows.duckdb"

_SESSION = requests.Session()
_SESSION.headers.update(_UA)


def _token_for(market_id: str) -> str | None:
    for closed in ("true", "false"):
        r = _SESSION.get(f"{GAMMA}/markets",
                         params={"id": market_id, "closed": closed, "limit": 1},
                         timeout=30)
        if r.status_code == 200 and r.json():
            tokens = json.loads(r.json()[0].get("clobTokenIds") or "[]")
            return tokens[0] if tokens else None
    return None


def fetch_window(con, market_id: str, event_ts: pd.Timestamp,
                 before_min: int, after_min: int) -> int:
    tok = _token_for(market_id)
    if not tok:
        con.execute("INSERT INTO fetch_log VALUES ('polymarket', ?, 'minute_window', 'no_clob_token', 0, ?, current_timestamp)",
                    [market_id, event_ts.isoformat()])
        return 0
    start = int(event_ts.timestamp()) - before_min * 60
    end = int(event_ts.timestamp()) + after_min * 60
    r = _SESSION.get(f"{CLOB}/prices-history",
                     params={"market": tok, "startTs": start, "endTs": end,
                             "fidelity": 1}, timeout=30)
    hist = r.json().get("history", []) if r.status_code == 200 else []
    rows = [(market_id, int(p["t"]),
             datetime.fromtimestamp(int(p["t"]), tz=timezone.utc).isoformat(),
             float(p["p"])) for p in hist]
    if rows:
        con.executemany(
            "INSERT INTO prices VALUES ('polymarket', ?, ?, ?, ?, current_timestamp)", rows)
    con.execute("INSERT INTO fetch_log VALUES ('polymarket', ?, 'minute_window', ?, ?, ?, current_timestamp)",
                [market_id, "ok" if r.status_code == 200 else "http_error",
                 len(rows), event_ts.isoformat()])
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Minute-fidelity event-window fetcher")
    ap.add_argument("--market-id")
    ap.add_argument("--event-ts", help="ISO timestamp of the event (UTC)")
    ap.add_argument("--before-min", type=int, default=180)
    ap.add_argument("--after-min", type=int, default=180)
    ap.add_argument("--requests-csv",
                    help="CSV with market_id,event_ts[,before_min,after_min]")
    a = ap.parse_args(argv)

    reqs: list[tuple[str, pd.Timestamp, int, int]] = []
    if a.requests_csv:
        df = pd.read_csv(a.requests_csv)
        for r in df.itertuples(index=False):
            reqs.append((str(r.market_id), pd.Timestamp(r.event_ts),
                         int(getattr(r, "before_min", a.before_min) or a.before_min),
                         int(getattr(r, "after_min", a.after_min) or a.after_min)))
    elif a.market_id and a.event_ts:
        reqs.append((a.market_id, pd.Timestamp(a.event_ts), a.before_min, a.after_min))
    else:
        ap.error("provide --market-id + --event-ts, or --requests-csv")

    con = connect(DB)
    total = 0
    for mid, ts, b, af in reqs:
        n = fetch_window(con, mid, ts, b, af)
        print(f"[window] market {mid} @ {ts.isoformat()} ±({b}m/{af}m): {n} minute points")
        total += n
        time.sleep(0.2)
    print(f"[window] stored {total} points -> {DB}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
