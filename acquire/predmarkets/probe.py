"""Discovery probe: census both platforms BEFORE committing to a price backfill.

Enumerates every market (Polymarket: full catalog via Gamma; Kalshi: series-first,
macro-relevant categories only, so sports/parlay noise never enters) and writes
per-market rows to CSV. The printed summary answers the go/no-go questions:
how many markets per year, per category, and above candidate liquidity floors.

The CSVs double as the backfill work-list — the price stage reads them instead of
re-enumerating.

    python -m predmarkets.probe            # from acquire/, or:
    python acquire/predmarkets/probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
_UA = {"User-Agent": "mira-predmarkets-research (personal research)"}

# Polymarket is tag-scoped: the full catalog is ~millions of markets (sports),
# and Gamma's offset pagination is hard-capped at 2000, so we census only the
# macro-relevant tag trees (related_tags=true pulls in descendants). Slugs that
# don't resolve are skipped with a note.
# "crypto" is deliberately absent: that tag tree is dominated by 15-minute
# BTC/ETH up-or-down ladder markets (six figures of them) — high-frequency
# price bets, not macro events. Crypto EVENT markets (ETF approval, regulation)
# largely also carry finance/business/politics tags and enter through those.
PM_TAG_SLUGS = [
    "politics", "elections", "economy", "business", "finance",
    "geopolitics", "world", "fed", "inflation", "energy", "climate",
    "companies", "tech", "science", "recession", "tariffs",
]

# Kalshi categories worth backfilling for a macro theme layer. Sports /
# Entertainment / Mentions / Social / Exotics are deliberately absent.
KALSHI_CATEGORIES = [
    "Economics", "Financials", "Politics", "Elections", "World",
    "Commodities", "Crypto", "Climate and Weather", "Companies",
    "Science and Technology", "Health", "Transportation", "Energy",
]

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "predmarkets"


def _get(url: str, params: dict, tries: int = 5, timeout_s: float = 30.0) -> requests.Response:
    """GET with backoff on 429/5xx — both APIs are public but rate-limited."""
    for attempt in range(tries):
        r = requests.get(url, params=params, timeout=timeout_s, headers=_UA)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


_PM_OFFSET_CAP = 2000   # Gamma 422s past offset 2000 — bands must stay under this
_PM_PAGE = 100          # server silently caps limit at 100


def _pm_row(m: dict, tag_slug: str) -> dict:
    return {
        "platform": "polymarket",
        "market_id": m.get("id"),
        "tag_slug": tag_slug,
        "question": m.get("question"),
        "category": m.get("category") or "",
        "created_at": m.get("createdAt"),
        "end_date": m.get("endDateIso") or m.get("endDate"),
        "closed": m.get("closed"),
        "volume_usd": m.get("volumeNum"),
        "liquidity_usd": m.get("liquidityNum"),
        "has_clob_tokens": bool(m.get("clobTokenIds")),
    }


def _pm_pages(params: dict, tag_slug: str, pace_s: float) -> tuple[list[dict], bool]:
    """Page one query to the offset cap. Returns (rows, capped)."""
    rows: list[dict] = []
    for offset in range(0, _PM_OFFSET_CAP, _PM_PAGE):
        r = _get(f"{GAMMA}/markets", {**params, "limit": _PM_PAGE, "offset": offset})
        batch = r.json()
        rows.extend(_pm_row(m, tag_slug) for m in batch)
        time.sleep(pace_s)
        if len(batch) < _PM_PAGE:
            return rows, False
    return rows, True


def _pm_band(tag_id: int, tag_slug: str, closed: str, lo: float, hi: float,
             pace_s: float) -> list[dict]:
    """One volume band [lo, hi) of a SINGLE tag (no related_tags — Gamma 500s when
    related_tags is combined with volume_num_max). Recursively splits when the
    2000-row offset cap truncates. Ties at identical volume (mostly $0) can't be
    split — those are truncated LOUDLY rather than silently."""
    rows, capped = _pm_pages({
        "tag_id": tag_id, "closed": closed,
        "volume_num_min": lo, "volume_num_max": hi,
    }, tag_slug, pace_s)
    if not capped:
        return rows
    if hi - lo < 0.01:
        print(f"  polymarket TRUNCATED: tag_id={tag_id} ({tag_slug}) closed={closed} "
              f"vol[{lo},{hi}) has >{_PM_OFFSET_CAP} rows, unsplittable",
              file=sys.stderr)
        return rows
    mid = (lo * hi) ** 0.5 if lo > 0 else max(hi / 1000, 0.01)
    return (_pm_band(tag_id, tag_slug, closed, lo, mid, pace_s)
            + _pm_band(tag_id, tag_slug, closed, mid, hi, pace_s))


def _pm_unit(tag_id: int, tag_slug: str, closed: str, pace_s: float) -> list[dict]:
    """One (tag tree, closed) census unit.

    Fast path: the whole tree via related_tags=true with no volume filters.
    If that hits the offset cap, expand the tree one level client-side
    (/tags/{id}/related-tags) and walk each tag with volume bands; dedupe by id.
    """
    rows, capped = _pm_pages(
        {"tag_id": tag_id, "related_tags": "true", "closed": closed},
        tag_slug, pace_s)
    if not capped:
        return rows
    rel = _get(f"{GAMMA}/tags/{tag_id}/related-tags", {}).json()
    tag_ids = [tag_id] + [t["relatedTagID"] for t in rel if t.get("relatedTagID")]
    print(f"  polymarket: {tag_slug} closed={closed} capped — expanding to "
          f"{len(tag_ids)} tags with volume bands", file=sys.stderr)
    out: dict[str, dict] = {}
    for tid in tag_ids:
        for row in _pm_band(tid, tag_slug, closed, 0.0, 1e10, pace_s):
            out[str(row["market_id"])] = row
    return list(out.values())


def probe_polymarket(pace_s: float = 0.2) -> pd.DataFrame:
    """Census of the macro-relevant Polymarket tag trees.

    Checkpointed per (tag, closed) unit: finished units append to
    probe_polymarket.csv and a rerun skips them.
    """
    out_csv = OUT_DIR / "probe_polymarket.csv"
    done: set[str] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv, usecols=["tag_slug", "closed"])
        done = {f"{t}|{str(c).lower()}" for t, c in prev.itertuples(index=False)}

    for slug in PM_TAG_SLUGS:
        r = requests.get(f"{GAMMA}/tags/slug/{slug}", timeout=30, headers=_UA)
        if r.status_code != 200:
            print(f"  polymarket: tag slug '{slug}' not found — skipped",
                  file=sys.stderr)
            continue
        tag_id = int(r.json()["id"])
        for closed in ("true", "false"):
            if f"{slug}|{closed}" in done:
                continue
            try:
                rows = _pm_unit(tag_id, slug, closed, pace_s)
            except requests.HTTPError as e:
                print(f"  polymarket: {slug} closed={closed} FAILED ({e}) — "
                      f"will retry on next run", file=sys.stderr)
                continue
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            print(f"  polymarket: {slug} closed={closed}: {len(rows)} markets",
                  file=sys.stderr)

    pm = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    if not pm.empty:
        pm = pm.drop_duplicates(subset=["market_id"]).reset_index(drop=True)
    return pm


def probe_polymarket_legacy(end_id: int = 400_000, pace_s: float = 0.2) -> int:
    """Id-walk the pre-tag era (2020-2023): those markets carry no modern tags, so
    the tag census can't see them. Batches of 100 ids with closed=true (the whole
    era is resolved by now). Checkpointed via legacy_walk_state.json.
    """
    import json
    out_csv = OUT_DIR / "probe_polymarket.csv"
    state_f = OUT_DIR / "legacy_walk_state.json"
    next_id = 1
    if state_f.exists():
        next_id = json.loads(state_f.read_text())["next_id"]
    found = 0
    while next_id <= end_id:
        ids = list(range(next_id, min(next_id + 100, end_id + 1)))
        r = _get(f"{GAMMA}/markets",
                 [("id", i) for i in ids] + [("closed", "true"), ("limit", 100)])
        batch = r.json()
        if batch:
            pd.DataFrame([_pm_row(m, "legacy-idwalk") for m in batch]).to_csv(
                out_csv, mode="a", header=not out_csv.exists(), index=False)
            found += len(batch)
        next_id = ids[-1] + 1
        state_f.write_text(json.dumps({"next_id": next_id}))
        if next_id % 20_000 < 100:
            print(f"  legacy walk: id {next_id}/{end_id}, {found} markets this run",
                  file=sys.stderr)
        time.sleep(pace_s)
    print(f"  legacy walk complete to id {end_id}: {found} markets this run",
          file=sys.stderr)
    return found


def probe_kalshi(pace_s: float = 0.15) -> pd.DataFrame:
    """Series-first census of macro-relevant Kalshi categories.

    Checkpointed: rows append to probe_kalshi.csv per series, and a rerun skips
    series already probed — the census survives being killed mid-run.
    """
    series = _get(f"{KALSHI}/series", {}).json()["series"]
    keep = [s for s in series if s.get("category") in KALSHI_CATEGORIES]

    out_csv = OUT_DIR / "probe_kalshi.csv"
    done: set[str] = set()
    if out_csv.exists():
        done = set(pd.read_csv(out_csv, usecols=["series_ticker"])["series_ticker"])
        keep = [s for s in keep if s["ticker"] not in done]
    print(f"  kalshi: {len(keep)} series to probe "
          f"({len(done)} already done, macro categories only)", file=sys.stderr)

    for i, s in enumerate(keep):
        rows: list[dict] = []
        st, cat = s["ticker"], s["category"]
        cursor = None
        while True:
            params = {"series_ticker": st, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            d = _get(f"{KALSHI}/markets", params).json()
            for m in d.get("markets", []):
                rows.append({
                    "platform": "kalshi",
                    "market_id": m.get("ticker"),
                    "series_ticker": st,
                    "category": cat,
                    "question": m.get("title"),
                    "created_at": m.get("open_time"),
                    "end_date": m.get("close_time"),
                    "status": m.get("status"),
                    "result": m.get("result"),
                    "volume_contracts": m.get("volume_fp"),
                    "notional_usd": m.get("notional_value_dollars"),
                })
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
            time.sleep(pace_s)
        # flush per series so a killed run resumes cleanly; a series with zero
        # markets still gets a sentinel row so the resume skip-set sees it
        if not rows:
            rows = [{"platform": "kalshi", "market_id": "", "series_ticker": st,
                     "category": cat, "question": "", "created_at": "",
                     "end_date": "", "status": "empty_series", "result": "",
                     "volume_contracts": 0, "notional_usd": 0}]
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(),
                                  index=False)
        if (i + 1) % 250 == 0:
            print(f"  kalshi: {i + 1}/{len(keep)} series...", file=sys.stderr)
        time.sleep(pace_s)
    full = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    if not full.empty:
        full = full[full["status"] != "empty_series"].reset_index(drop=True)
    return full


def _year(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed").dt.year


def summarize(pm: pd.DataFrame, ks: pd.DataFrame) -> None:
    print("\n=== Polymarket ===")
    print(f"markets total: {len(pm)}")
    pm = pm.assign(year=_year(pm["end_date"]))
    print("by end-year:")
    print(pm["year"].value_counts().sort_index().to_string())
    vol = pd.to_numeric(pm["volume_usd"], errors="coerce").fillna(0)
    print(f"volume: total ${vol.sum():,.0f} | median ${vol.median():,.0f} "
          f"| p90 ${vol.quantile(0.9):,.0f}")
    for floor in (1_000, 10_000, 100_000):
        sub = pm[vol >= floor]
        print(f"  >= ${floor:>7,}: {len(sub):6d} markets "
              f"({sub['year'].value_counts().sort_index().to_dict()})")
    print("top categories:")
    print(pm["category"].value_counts().head(15).to_string())

    print("\n=== Kalshi (macro categories) ===")
    print(f"markets total: {len(ks)}")
    ks = ks.assign(year=_year(ks["end_date"]))
    print("by close-year:")
    print(ks["year"].value_counts().sort_index().to_string())
    print("by category:")
    print(ks["category"].value_counts().to_string())
    vol = pd.to_numeric(ks["volume_contracts"], errors="coerce").fillna(0)
    print(f"volume(contracts): median {vol.median():,.0f} | p90 {vol.quantile(0.9):,.0f}")
    for floor in (100, 1_000, 10_000):
        sub = ks[vol >= floor]
        print(f"  >= {floor:>6,} contracts: {len(sub):6d} markets "
              f"({sub['year'].value_counts().sort_index().to_dict()})")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Prediction-market discovery probe")
    ap.add_argument("--platform",
                    choices=["polymarket", "polymarket-legacy", "kalshi", "both"],
                    default="both")
    ap.add_argument("--summary-only", action="store_true",
                    help="just summarize existing probe CSVs")
    a = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        if a.platform in ("polymarket", "both"):
            print("[probe] Polymarket: macro tag trees...", file=sys.stderr)
            pm = probe_polymarket()
            print(f"[probe] polymarket census: {len(pm)} unique markets")
        if a.platform == "polymarket-legacy":
            print("[probe] Polymarket: pre-tag-era id walk...", file=sys.stderr)
            probe_polymarket_legacy()
        if a.platform in ("kalshi", "both"):
            print("[probe] Kalshi: macro categories, series-first...", file=sys.stderr)
            ks = probe_kalshi()
            print(f"[probe] kalshi census: {len(ks)} markets")

    pm_csv, ks_csv = OUT_DIR / "probe_polymarket.csv", OUT_DIR / "probe_kalshi.csv"
    pm = (pd.read_csv(pm_csv).drop_duplicates(subset=["market_id"])
          if pm_csv.exists() else pd.DataFrame())
    ks = pd.read_csv(ks_csv) if ks_csv.exists() else pd.DataFrame()
    if not ks.empty:
        ks = ks[ks["status"] != "empty_series"]
    if not pm.empty and not ks.empty:
        summarize(pm, ks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
