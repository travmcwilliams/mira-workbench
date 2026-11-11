"""Polymarket price-history backfill: census CSV -> DuckDB daily probability series.

Two checkpointed stages, driven by the probe census (probe_polymarket.csv):

  meta   — re-query Gamma in id batches for the selected volume tier to recover
           clobTokenIds (the census kept only a boolean) -> markets table.
  prices — CLOB /prices-history per market's YES token, daily fidelity, full
           range -> prices table. Pre-CLOB (2020-2021 AMM-only) markets have no
           CLOB series; they are LOGGED as no_clob_token, never silently dropped.

Both stages resume via fetch_log — rerun the module until it reports nothing left.

    python acquire/predmarkets/backfill.py --min-volume 100000 --stage both
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

from store import connect, done_market_ids  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
_UA = {"User-Agent": "mira-predmarkets-research (personal research)"}
CENSUS = Path(__file__).resolve().parents[2] / "data" / "predmarkets" / "probe_polymarket.csv"


_SESSION = requests.Session()          # keep-alive: one TLS handshake, not 15k
_SESSION.headers.update(_UA)


def _get(url: str, params, tries: int = 5, timeout_s: float = 30.0) -> requests.Response:
    for attempt in range(tries):
        r = _SESSION.get(url, params=params, timeout=timeout_s)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
            continue
        return r
    return r


def select_tier(min_volume: float) -> pd.DataFrame:
    df = pd.read_csv(CENSUS, low_memory=False).drop_duplicates(subset=["market_id"])
    vol = pd.to_numeric(df["volume_usd"], errors="coerce").fillna(0)
    tier = df[vol >= min_volume].copy()
    tier["market_id"] = tier["market_id"].astype(str)
    return tier


def stage_meta(con, tier: pd.DataFrame, pace_s: float = 0.2) -> None:
    done = done_market_ids(con, "polymarket", "meta")
    todo = [m for m in tier["market_id"] if m not in done]
    print(f"[meta] {len(todo)} markets to fetch ({len(done)} done)", file=sys.stderr)
    slugs = dict(zip(tier["market_id"], tier["tag_slug"]))
    for i in range(0, len(todo), 100):
        ids = todo[i:i + 100]
        got: dict[str, dict] = {}
        for closed in ("true", "false"):
            r = _get(f"{GAMMA}/markets",
                     [("id", x) for x in ids] + [("closed", closed), ("limit", 100)])
            if r.status_code == 200:
                for m in r.json():
                    got[str(m["id"])] = m
            time.sleep(pace_s)
        for mid in ids:
            m = got.get(mid)
            if m is None:
                con.execute(
                    "INSERT INTO fetch_log VALUES ('polymarket', ?, 'meta', 'not_found', 0, '', current_timestamp)",
                    [mid])
                continue
            tokens = json.loads(m.get("clobTokenIds") or "[]")
            con.execute(
                "INSERT INTO markets VALUES (?,?,?,?,?,?,?,?,?,current_timestamp)",
                ["polymarket", mid, m.get("question"), slugs.get(mid, ""),
                 m.get("endDateIso") or m.get("endDate"), bool(m.get("closed")),
                 float(m.get("volumeNum") or 0), tokens[0] if tokens else None,
                 m.get("outcomes")])
            con.execute(
                "INSERT INTO fetch_log VALUES ('polymarket', ?, 'meta', 'ok', 1, '', current_timestamp)",
                [mid])
        if (i // 100) % 20 == 0:
            print(f"[meta] {i + len(ids)}/{len(todo)}", file=sys.stderr)


def stage_prices(con, pace_s: float = 0.25) -> None:
    done = done_market_ids(con, "polymarket", "prices")
    rows = con.execute(
        "SELECT market_id, clob_token_yes FROM markets WHERE platform='polymarket'"
    ).fetchall()
    todo = [(mid, tok) for mid, tok in rows if mid not in done]
    print(f"[prices] {len(todo)} markets to fetch ({len(done)} done)", file=sys.stderr)
    for n, (mid, tok) in enumerate(todo):
        if not tok:
            con.execute(
                "INSERT INTO fetch_log VALUES ('polymarket', ?, 'prices', 'no_clob_token', 0, 'pre-CLOB AMM market', current_timestamp)",
                [mid])
            continue
        r = _get(f"{CLOB}/prices-history",
                 {"market": tok, "interval": "max", "fidelity": 1440})
        if r.status_code != 200:
            con.execute(
                "INSERT INTO fetch_log VALUES ('polymarket', ?, 'prices', 'http_error', 0, ?, current_timestamp)",
                [mid, f"status={r.status_code}"])
            time.sleep(pace_s)
            continue
        hist = r.json().get("history", [])
        if hist:
            rows_out = [
                (mid, int(pt["t"]),
                 datetime.fromtimestamp(int(pt["t"]), tz=timezone.utc).date().isoformat(),
                 float(pt["p"]))
                for pt in hist]
            con.executemany(
                "INSERT INTO prices VALUES ('polymarket', ?, ?, ?, ?, current_timestamp)",
                rows_out)
        con.execute(
            "INSERT INTO fetch_log VALUES ('polymarket', ?, 'prices', 'ok', ?, '', current_timestamp)",
            [mid, len(hist)])
        if (n + 1) % 200 == 0:
            print(f"[prices] {n + 1}/{len(todo)}", file=sys.stderr)
        time.sleep(pace_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Polymarket price backfill (checkpointed)")
    ap.add_argument("--min-volume", type=float, default=100_000)
    ap.add_argument("--stage", choices=["meta", "prices", "both"], default="both")
    a = ap.parse_args(argv)

    con = connect()
    tier = select_tier(a.min_volume)
    print(f"[backfill] tier >= ${a.min_volume:,.0f}: {len(tier)} markets")
    if a.stage in ("meta", "both"):
        stage_meta(con, tier)
    if a.stage in ("prices", "both"):
        stage_prices(con)

    stats = con.execute("""
        SELECT stage, status, count(*) FROM fetch_log
        WHERE platform='polymarket' GROUP BY 1,2 ORDER BY 1,2""").fetchall()
    n_prices = con.execute("SELECT count(*) FROM prices").fetchone()[0]
    print("[backfill] fetch_log:", stats)
    print(f"[backfill] price points stored: {n_prices:,}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
