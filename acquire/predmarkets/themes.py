"""Theme mapping + daily exports: DuckDB probability series -> join-ready CSVs.

Maps every backfilled market to a macro theme via themes.json (data, not code;
first match wins, unmatched -> 'unmapped' for review) and writes two exports:

  pm_market_daily.csv — long table: date, theme, market_id, prob, chg
      (day-over-day). The event-study consumer. Question text and volume live
      once per market in pm_markets.csv (join on market_id), not per row.
  pm_theme_daily.csv  — per theme per day: n_markets, repricing intensity
      (volume-weighted mean |chg|), max |chg|, weighted mean prob, total weight.
      The downstream join consumer (ctx_pm_* metrics; leak-safe at T+1 downstream).

Probabilities of DIFFERENT questions are not directly comparable, so the theme
aggregate leans on CHANGE-based features (repricing intensity), not prob levels;
the weighted mean prob is exported but is the weakest column — treat accordingly.

TIMESTAMP CONVENTION (validated on FOMC/CPI/NFP days): CLOB daily buckets are
END-stamped, so the row dated D carries the close of D-1's US session, and D's
`chg` is the move DURING US day D-1. A row dated D is fully known by D 00:00 UTC
— before D's US open — so joining it to day-D returns is leak-safe as-is
(available_at = D 00:00Z). For event-day alignment, compare events on day E
against rows dated E+1.

    python acquire/predmarkets/themes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from store import connect  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "predmarkets"
THEMES_JSON = _HERE / "themes.json"


def load_rules() -> tuple[list[str], list[dict], dict[str, str]]:
    cfg = json.loads(THEMES_JSON.read_text(encoding="utf-8"))
    return cfg["exclude_patterns"], cfg["themes"], cfg.get("tag_fallbacks", {})


def assign_theme(question: str, exclude: list[str], themes: list[dict],
                 tag_fallbacks: dict[str, str] | None = None,
                 tag_slug: str = "") -> str:
    q = f" {(question or '').lower()} "
    if any(pat in q for pat in exclude):
        return "excluded_ladder"
    for t in themes:
        if any(kw in q for kw in t["any_keywords"]):
            return t["key"]
    if tag_fallbacks and tag_slug in tag_fallbacks:
        return tag_fallbacks[tag_slug]
    return "unmapped"


def build(min_points: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exclude, themes, tag_fallbacks = load_rules()
    con = connect()
    mk = con.execute(
        "SELECT market_id, question, tag_slug, volume_usd FROM markets WHERE platform='polymarket'"
    ).df()
    px = con.execute("""
        SELECT market_id, date, avg(prob) AS prob
        FROM prices WHERE platform='polymarket'
        GROUP BY market_id, date
    """).df()
    con.close()

    mk["theme"] = [assign_theme(q, exclude, themes, tag_fallbacks, str(s))
                   for q, s in zip(mk["question"], mk["tag_slug"])]

    counts = px.groupby("market_id").size()
    keep_ids = set(counts[counts >= min_points].index)
    px = px[px["market_id"].isin(keep_ids)]

    d = px.merge(mk, on="market_id", how="inner").sort_values(["market_id", "date"])
    d["chg"] = d.groupby("market_id")["prob"].diff()

    market_daily = d[["date", "theme", "market_id", "prob", "chg"]].reset_index(drop=True)

    a = d.dropna(subset=["chg"]).copy()
    a["w"] = a["volume_usd"].clip(lower=1.0)
    a["w_abs_chg"] = a["chg"].abs() * a["w"]
    a["w_prob"] = a["prob"] * a["w"]
    g = a.groupby(["date", "theme"])
    theme_daily = g.agg(
        n_markets=("market_id", "nunique"),
        sum_w=("w", "sum"),
        sum_w_abs_chg=("w_abs_chg", "sum"),
        sum_w_prob=("w_prob", "sum"),
        max_abs_chg=("chg", lambda s: s.abs().max()),
    ).reset_index()
    theme_daily["repricing"] = theme_daily["sum_w_abs_chg"] / theme_daily["sum_w"]
    theme_daily["mean_prob"] = theme_daily["sum_w_prob"] / theme_daily["sum_w"]
    theme_daily = theme_daily.drop(columns=["sum_w_abs_chg", "sum_w_prob"])

    audit = (mk.groupby("theme").agg(markets=("market_id", "count"),
                                     volume_usd=("volume_usd", "sum"))
             .sort_values("markets", ascending=False).reset_index())
    return market_daily, theme_daily, audit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Theme mapping + daily exports")
    ap.add_argument("--min-points", type=int, default=5,
                    help="drop markets with fewer daily price points than this")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="export directory (use another drive if C: is tight)")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    market_daily, theme_daily, audit = build(a.min_points)
    out_dir.mkdir(parents=True, exist_ok=True)
    market_daily.to_csv(out_dir / "pm_market_daily.csv", index=False)
    theme_daily.to_csv(out_dir / "pm_theme_daily.csv", index=False)

    con = connect()
    mk = con.execute("""SELECT market_id, question, tag_slug, volume_usd, end_date
                        FROM markets WHERE platform='polymarket'""").df()
    con.close()
    mk.to_csv(out_dir / "pm_markets.csv", index=False)

    print(f"[themes] market-days: {len(market_daily):,} | "
          f"theme-days: {len(theme_daily):,}")
    print("[themes] mapping audit (markets / lifetime volume):")
    for r in audit.itertuples(index=False):
        print(f"  {r.theme:18s} {r.markets:6d}  ${r.volume_usd:,.0f}")
    unmapped = audit[audit["theme"] == "unmapped"]
    if not unmapped.empty:
        print(f"[themes] REVIEW: {int(unmapped['markets'].iloc[0])} unmapped markets "
              f"— sample below")
        ids = market_daily.loc[market_daily["theme"] == "unmapped", "market_id"].unique()
        for q in mk[mk["market_id"].isin(ids)]["question"].head(10):
            print(f"    - {q[:90]}")
    print(f"wrote {out_dir / 'pm_market_daily.csv'}")
    print(f"wrote {out_dir / 'pm_theme_daily.csv'}")
    print(f"wrote {out_dir / 'pm_markets.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
