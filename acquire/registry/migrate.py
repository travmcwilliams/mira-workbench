"""
registry/migrate.py
Two ways to fill the registry:

  * seed_from_config(conn, cfg)   — load the curated Phase-1 spine + rules from the
    config pack (platforms, categories, sources, crawl policies, access rules).
  * migrate_json_registry(conn)   — fold Mira's EXISTING acquire/data/source_registry.json
    (the flat legacy list) into the relational registry, non-destructively.

Both are idempotent: natural-key upserts mean running them twice changes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import registry as R
from .config import default_config

# Legacy scraper_type -> crawl_method mapping for the JSON migration.
_SCRAPER_TO_METHOD = {
    "has_download_link": "html",
    "has_search_form": "headless_browser",
    "has_record_link": "html",
    "page_only": "html",
    "api": "api",
}


def seed_from_config(conn: Any, cfg: dict[str, Any] | None = None) -> dict[str, int]:
    """Load platforms, categories, inclusion rules, and the curated source spine."""
    cfg = cfg or default_config()

    for cat in cfg.get("categories", []):
        R.upsert_category(conn, name=cat["name"], description=cat.get("description"))

    platform_ids: dict[str, int] = {}
    for p in cfg.get("platforms", []):
        pid = R.upsert_platform(conn, **p)
        platform_ids[p["platform_name"]] = pid

    for rule in cfg.get("inclusion_rules", []):
        R.add_inclusion_rule(conn, **rule)

    n_sources = 0
    for s in cfg.get("sources", []):
        s = dict(s)
        categories = s.pop("categories", [])
        crawl = s.pop("crawl", {})
        access = s.pop("access", {})
        platform_name = s.pop("platform", None)
        platform_id = platform_ids.get(platform_name) if platform_name else None

        source_id = R.upsert_source(
            conn,
            source_url=s.pop("source_url"),
            canonical_name=s.pop("canonical_name"),
            source_type=s.pop("source_type", "unknown"),
            platform_id=platform_id,
            topical_focus=categories or None,
            **s,
        )
        n_sources += 1

        for name in categories:
            cid = R.upsert_category(conn, name=name)
            R.map_source_category(conn, source_id, cid)

        if crawl:
            R.set_crawl_policy(conn, source_id, **crawl)
        if access:
            R.set_access_rule(conn, source_id, **access)

    return {
        "platforms": len(cfg.get("platforms", [])),
        "categories": len(cfg.get("categories", [])),
        "inclusion_rules": len(cfg.get("inclusion_rules", [])),
        "sources": n_sources,
    }


def _legacy_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "source_registry.json"


def migrate_json_registry(conn: Any, path: str | Path | None = None) -> dict[str, int]:
    """
    Fold the legacy flat JSON registry into the relational tables. Each legacy row
    becomes a source_entity (+ a crawl_policy derived from its scraper_type/data_method).
    Rows with example-/demo hostnames are skipped so the real registry stays clean.
    """
    p = Path(path) if path else _legacy_registry_path()
    if not p.exists():
        return {"migrated": 0, "skipped": 0, "reason": "no legacy registry file"}

    with open(p, "r", encoding="utf-8") as f:
        reg = json.load(f)

    migrated = skipped = 0
    for src in reg.get("sources", []):
        url = src.get("source_url")
        name = src.get("source_name") or src.get("source_id")
        if not url or "example" in url.lower() or ".local" in url.lower():
            skipped += 1
            continue

        source_id = R.upsert_source(
            conn,
            source_url=url,
            canonical_name=name,
            source_type="legacy_dataset_source",
            region=src.get("region"),
            notes=f"migrated from source_registry.json (legacy_id={src.get('source_id')})",
        )
        method = _SCRAPER_TO_METHOD.get(src.get("scraper_type", ""), "html")
        if src.get("data_method") == "api":
            method = "api"
        R.set_crawl_policy(conn, source_id, crawl_method=method, crawl_frequency="daily",
                           terms_risk_level="unknown")
        migrated += 1

    return {"migrated": migrated, "skipped": skipped}
