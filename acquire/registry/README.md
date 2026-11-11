# Mira Source Registry

A **governed, self-updating source-intelligence database** — not a flat list of
websites. It decides *which* parts of the internet Mira attends to, on *which* day,
for *which* topic, and at *what* confidence level.

> Mira must never assume high-traffic sources are more *truthful*, only more
> *influential*. Authority, readership, virality, citation, official status,
> historical reliability, and legal accessibility are scored **separately**.

## Why this is DuckDB, not Postgres

The reference design specified Postgres + object storage. Mira is **local-first,
embedded DuckDB** (`acquire/utils/db.py`), and the project explicitly assumes
**no cloud/DB service** (local-first design). Standing up Postgres would (a) violate
that doctrine, (b) orphan the existing `acquire/` registry + discovery engine, and
(c) add a service dependency Mira does not have. So the *schema design* — which is
sound — was **reconciled onto DuckDB**:

| Reference (Postgres) | Here (DuckDB) |
|---|---|
| `BIGSERIAL PRIMARY KEY` | `BIGINT PRIMARY KEY DEFAULT nextval('seq_…')` |
| `TIMESTAMPTZ` | `TIMESTAMP` |
| `TEXT[]` | `TEXT` holding a JSON list (`registry.encode_list`) |
| `now()` | `CURRENT_TIMESTAMP` |
| FK constraints | **code-enforced** integrity (avoids DuckDB delete/order friction) |
| `NULL` in composite keys | sentinel `account_id = 0` = "source-level" |

If a real concurrent crawler fleet ever justifies Postgres, the schema is portable —
the DDL is isolated in `schema.py` and the API takes an explicit connection.

## Layout

| File | Role |
|---|---|
| `schema.py` | DuckDB DDL for 13 tables + sequences (`init_registry_schema(conn)`) |
| `scoring.py` | Composite **Mira Source Score** (0–100) + inclusion-rule evaluation. Pure functions. |
| `registry.py` | Idempotent upsert/query API. Every fn takes an explicit `conn`. |
| `config.py` | Loads + validates the config pack. |
| `config/source_registry.yaml` | **Config-as-data**: scoring weights, norm caps, inclusion rules, curated seed. |
| `migrate.py` | `seed_from_config()` + `migrate_json_registry()` (folds in the legacy flat registry). |

## Tables (13)

`source_platform`, `source_category`, `source_entity`, `source_account`,
`source_category_map`, `crawl_policy`, `source_access_rule`, `inclusion_rule`,
`source_metric_snapshot`, `source_discovery_candidate`, `source_daily_coverage`,
`event_cluster`, `event_source_map`.

**Deferred to the content layer:** `content_item`, `content_metric_snapshot`,
`viral_signal`, `event_content_map`. Raw crawled content already flows through Mira's
`records` table and the `acquire/news/` pipeline; this module governs **sources**, not
content. `event_source_map` gives the registry-side event linkage until then.

## Scoring

```
Mira Source Score (0–100) =
  0.25·authority + 0.20·readership + 0.15·engagement + 0.15·citation
+ 0.10·freshness + 0.10·topical_relevance + 0.05·accessibility
```

Components are 0–1 (log-compressed against `norm_caps` because web traffic is heavily
concentrated — arXiv 2404.17095). Weights live in the YAML and must sum to 1.0
(validated at load). **Overrides:** `official_government`, `official_incident_feed`,
and `academic_repository` sources are included regardless of audience metrics.

## Usage

```bash
mira registry setup      # schema + curated seed + fold in legacy source_registry.json
mira registry status     # row counts per table
mira registry list       # registered sources
mira registry top        # ranked by Mira Source Score
```

```python
# Programmatic (production wires the DuckDB singleton):
from core.acquire_bridge import registry_setup, registry_top_sources_df
registry_setup()                       # idempotent
df = registry_top_sources_df(limit=25)
```

The discovery engine (`acquire/discovery/`) feeds candidates in via
`core.acquire_bridge.registry_promote_discovery_candidates(...)` →
`source_discovery_candidate` for human review → `promote_candidate()`.

## What is intentionally NOT here (honest scope)

- **No 5-year internet-wide backfill.** That is a multi-license, multi-API, ToS-governed
  *program*, not a v1. The schema (`crawl_policy`, `source_daily_coverage`) is ready for
  it; the crawlers are not seeded.
- **No social-platform ingestion.** X/TikTok/Meta APIs are closed or paid. Seeded as
  platforms + governed by `source_access_rule`, but not crawled.
- **Coverage caveat:** Mira's own finding is that news sentiment showed *no* forward-return
  edge (temporal R²≈0). More sources ≠ more signal — inclusion is justified per use-case,
  which is exactly what `inclusion_rule` enforces.
