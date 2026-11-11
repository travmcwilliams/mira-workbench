"""
Mira Source Registry — a governed, self-updating source-intelligence database.

Not a flat list of websites: a schema that supports seeded sources, automated
discovery, configurable inclusion thresholds, metric snapshots + composite scoring,
per-source crawl policy + legal/access rules, daily coverage tracking, and
event-cluster linkage. Reconciled onto Mira's local-first DuckDB stack.

Typical use (production wires `conn = utils.db.get_db()`):

    from registry import init_registry_schema, seed_from_config, migrate_json_registry
    init_registry_schema(conn)
    seed_from_config(conn)          # curated Phase-1 spine from the config pack
    migrate_json_registry(conn)     # fold in the legacy flat registry
"""

from __future__ import annotations

from .config import default_config, load_config
from .migrate import migrate_json_registry, seed_from_config
from .schema import drop_registry_schema, init_registry_schema
from .scoring import (
    evaluate_inclusion,
    mira_source_score,
    validate_weights,
)

__all__ = [
    "init_registry_schema",
    "drop_registry_schema",
    "seed_from_config",
    "migrate_json_registry",
    "load_config",
    "default_config",
    "mira_source_score",
    "evaluate_inclusion",
    "validate_weights",
]
