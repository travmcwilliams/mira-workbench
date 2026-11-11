#!/usr/bin/env python
"""
scripts/registry_update.py
Helper to add or update a verified source in the registry (with validation).

After manual discovery / verification:
    python scripts/registry_update.py --add --id 00004 --name "New Source" --region 00 \
        --region-full "Demo Region Zero" --url "https://..." --type has_download_link

Always run registry_repair.py --verify-only afterwards (and git status).
"""

import argparse
import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent.parent / "data" / "source_registry.json"


def load():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save(data):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--add", action="store_true")
    p.add_argument("--id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--region-full", required=True, dest="region_full")
    p.add_argument("--url", default=None)
    p.add_argument("--type", required=True, dest="scraper_type",
                   choices=["has_download_link", "has_record_link", "has_search_form", "page_only"])
    p.add_argument("--method", default="scrape", choices=["scrape", "api", "manual"])
    p.add_argument("--mapping", default=None,
                   help='Optional column mapping as JSON string, e.g. \'{"iata":"record_id","airport":"entity_name","cnt":"primary_value"}\'')
    args = p.parse_args()

    data = load()
    existing = next((s for s in data["sources"] if s["source_id"] == args.id), None)

    new_entry = {
        "source_id": args.id,
        "source_name": args.name,
        "region": args.region,
        "region_full": args.region_full,
        "source_url": args.url,
        "scraper_type": args.scraper_type,
        "data_method": args.method,
        "last_scraped": None
    }
    if args.mapping:
        try:
            new_entry["column_mapping"] = json.loads(args.mapping)
        except Exception as e:
            print(f"Warning: could not parse --mapping as JSON: {e}")

    if existing:
        print(f"Updating existing source_id={args.id}")
        for k, v in new_entry.items():
            existing[k] = v
    else:
        print(f"Adding new source_id={args.id}")
        data["sources"].append(new_entry)

    print("Registry updated. You can now use --mapping when running the ingestor for explicit column control.")

    save(data)
    print("Registry updated. NOW RUN: python scripts/registry_repair.py --verify-only")


if __name__ == "__main__":
    main()
