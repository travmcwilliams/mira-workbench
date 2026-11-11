#!/usr/bin/env python
"""
scripts/source_url_finder.py
Web-search-powered source discovery (Agent 0).

Per spec §6.4:
- NEVER use library scraping of consumer search engines from the VM (429s).
- Use dedicated search API (Serper, Bing, etc.) with header key.
- Score results, filter by region, keyword-score candidate files.
- Manual verification of top hits before registry_update.py + repair.

This is a stub that prints what a real run would do.
Set SEARCH-API-KEY and implement the HTTP call.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.secrets import get_search_api_key

REGISTRY_PATH = Path(__file__).parent.parent / "data" / "source_registry.json"


def score_result(url: str, title: str, region: str, keywords: list[str]) -> int:
    """Very rough 0-100 scorer. Real impl uses better signals."""
    score = 40
    url_l = (url or "").lower()
    title_l = (title or "").lower()
    for kw in keywords:
        if kw.lower() in url_l or kw.lower() in title_l:
            score += 15
    if region and region.lower() in url_l:
        score += 20
    return min(100, score)


def main():
    p = argparse.ArgumentParser(description="Discover candidate sources via search API (stub)")
    p.add_argument("--region", default="00")
    p.add_argument("--keywords", default="grant opportunities,funding,awards")
    p.add_argument("--limit", type=int, default=5)
    args = p.parse_args()

    key = get_search_api_key()
    if not key:
        print("[source_url_finder] No SEARCH-API-KEY. Using simulated results for scaffold.")
        results = [
            {"url": "https://example-data.local/grants/alpha.csv", "title": "Alpha Grants CSV 2025", "score": 92},
            {"url": "https://example-data.local/grants/beta/search", "title": "Beta Agency Search Portal", "score": 78},
        ]
    else:
        print("[source_url_finder] TODO: call real search API with key (Serper-style).")
        print("  Would query for region-filtered grant lists etc.")
        results = []

    print(f"\nTop candidates for region={args.region}:")
    for r in results:
        print(f"  score={r['score']:3d}  {r['title'][:50]:50s} {r['url']}")

    print("\nNext steps (manual):")
    print("  1. Verify the real page/file manually.")
    print("  2. python scripts/registry_update.py --add ...")
    print("  3. python scripts/registry_repair.py --verify-only")


if __name__ == "__main__":
    main()
