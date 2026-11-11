# Distributed-Source Intelligence Platform (DSI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](#)
[![Status](https://img.shields.io/badge/status-scaffold-orange.svg)](#)
[![DB: DuckDB](https://img.shields.io/badge/db-DuckDB-yellow.svg)](https://duckdb.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Reference Technical Specification & Build Scaffold**

> Multi-agent platform that aggregates semi-structured records from hundreds of inconsistent web sources, normalizes them to one canonical schema, enriches/scores/tracks them over time, and serves the result as an intelligence product.

This is a recreatable, domain-neutral scaffold for any system that:

- Aggregates semi-structured records from hundreds of independent, inconsistent web sources
- Normalizes them to one stable canonical schema
- Enriches, scores, tracks over time (deadlines, lifecycle)
- Serves the result as an intelligence product (dashboards, calendars, watchlists, APIs)

The architecture is identical across domains; **only four things change** when you repoint it (see §13 below and the original spec).

> Reference spec abstracted from a working multi-agent aggregation platform (CountyIQ / Tax Lien Intelligence). Use this as the drop-in blueprint.

## Quick Start (Local Demo on Windows)

```powershell
# 1. Create venv (use the vm file on Linux target too)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install (scaffold demo works with core deps)
pip install -r requirements.txt

# 3. (Optional) set local paths if you want to override defaults
$env:APP_PROJECT_ROOT = "<repo>\acquire"
$env:APP_DB_PATH = "<repo>\acquire\db\app.duckdb"

# 4. Run the end-to-end demo (first real dataset milestone)
python scripts\run_demo_ingest.py
```

You should see:
- Debug prints of raw columns + first 3 rows (from schema_mapper + ingestor)
- Canonical rows produced
- Dedup run
- Final count > 0
- Sample of records in the unified `records` table

Then verify health:
```powershell
python scripts\registry_repair.py --verify-only
```

### Local GUI (Recommended for daily use)

```powershell
# Install streamlit if not already (it's in requirements.txt)
pip install streamlit

# Launch the fully local GUI
streamlit run gui/app.py
```

The GUI lets you:
- Browse and add sources to the registry (with repair).
- Trigger real local scrapes using the fetcher + ingestor (anti-bot strategies, PDF parsing, dedup, logging).
- Explore the DuckDB with queries.
- View run logs and the manual review queue.
- Test the fetcher on any URL.

Everything stays on your machine. No cloud accounts or external services are required for the core scraping + data flows.

## Canonical Folder Layout (exactly per spec)

```
acquire/
├── agents/
│   ├── source_ingestor.py
│   ├── document_parser.py
│   ├── event_calendar.py
│   ├── enrichment_agent.py
│   ├── cross_reference_agent.py
│   ├── lifecycle_engine.py
│   ├── holdings_agent.py
│   ├── rules_monitor.py
│   ├── schema_mapper.py
│   └── platform_scrapers/
│       └── platform_c_scraper.py
├── crawlers/
│   ├── region_scan.py
│   └── region_demo_scraper.py
├── news/                     # Wayback news-corpus campaign (see news/README.md)
│   ├── campaign.py           # plan/scrape/score/retag/export/status CLI engine
│   ├── outlets.json          # outlet registry + CDX evidence (data, not code)
│   ├── ticker_map.csv        # top-1000 SEC tickers + curated aliases
│   ├── entities.py           # headline -> ticker matching
│   ├── tone.py               # lexicon + FinBERT tone scorers
│   ├── wayback.py            # polite CDX/snapshot client
│   ├── parse.py              # RSS/Atom snapshot -> dated headlines
│   ├── market.py             # Yahoo closes + leak-safe forward returns
│   ├── store.py              # campaign DuckDB (articles + chunk checkpoint)
│   └── backfill.py           # one-shot precursor (campaign supersedes it)
├── scripts/
│   ├── source_url_finder.py
│   ├── registry_update.py
│   ├── registry_repair.py
│   ├── rebuild_db.py
│   └── run_demo_ingest.py
├── utils/
│   ├── object_storage.py
│   ├── secrets.py
│   └── db.py                 # THE singleton
├── data/
│   ├── raw/{source_id}/...
│   ├── source_registry.json  # MASTER - schema is law
│   ├── scraper_run_log.json
│   └── sources_needing_review.json
├── db/
│   └── app.duckdb
├── .env.template
├── ENVIRONMENT.md
├── SESSION_START.md
├── requirements.txt
├── requirements_vm.txt
└── README.md
```

## The 10 Agents + Discovery (spec §4)

| # | Agent | Key file |
|---|-------|----------|
| 0 | Source Discovery | scripts/source_url_finder.py |
| 1 | Source Ingestor | agents/source_ingestor.py |
| 2 | Document Parser | agents/document_parser.py |
| 3 | Event Calendar | agents/event_calendar.py |
| 4 | Platform Scrapers | agents/platform_scrapers/ |
| 5 | Enrichment + Scoring | agents/enrichment_agent.py |
| 6 | Cross-Reference | agents/cross_reference_agent.py |
| 7 | Lifecycle Engine | agents/lifecycle_engine.py |
| 8 | Holdings / Watchlist | agents/holdings_agent.py |
| 9 | Rules Monitor (LLM) | agents/rules_monitor.py |
|10 | Schema Mapper | agents/schema_mapper.py |

**Agent conventions (enforced in this scaffold):**
- Print column names + first 3 rows of any input *before* parsing.
- Print final record count after writing.
- Request DB from `utils.db.get_db()` only.
- Long-running agents append to `scraper_run_log.json`.

## Data Architecture Highlights (spec §3)

- **Single canonical table** `records` (deliberately small & stable):
  ```sql
  record_id, entity_name, location_text, primary_value, period,
  source_name, source_id, region
  ```
- **Source registry** (`data/source_registry.json`) has a rigid schema. `registry_repair.py` is the gatekeeper. Run it after every edit.
- Dedup on `(record_id, source_id, period)` after **every** ingest.
- `scraper_type` drives ingestion strategy: `has_download_link` (highest priority) → `has_record_link` → `has_search_form` → `page_only` (manual review).

## Adapting to a New Domain (§13 of the spec)

You only touch **four** things:

1. **Canonical schema** (rarely — only if your core fields truly don't fit the 8 columns). Usually keep as-is and use enrichment/extras.
2. **Schema Mapper rules** (`agents/schema_mapper.py`) — the per-source column → canonical mappings + heuristics.
3. **Scoring model** (`agents/enrichment_agent.py`) — your domain ranking / opportunity score.
4. **Lifecycle rules** (`agents/lifecycle_engine.py`) — which deadlines and derived metrics matter for your record type.

Everything else (registry, discovery via search API, ingestion paths, anti-bot Playwright, dedup, object storage sync, secrets, VM lifecycle, run logging, manual review queue, agent decoupling) is **domain-agnostic and reusable**.

## Build / Recreate Sequence (summary of spec §10)

See the full reference spec (the source of this scaffold) for the detailed 9 phases.

High-level:
- Phase 0: Lock the narrow record type (we chose "Grant / Funding Opportunities" for the demo data in this scaffold).
- Phase 1-2: Paths + first real dataset (the demo ingest does this).
- Phase 3: Multi-source registry + repair gate.
- Phase 4: All agents scaffolded (stubs for most; full for ingestor + mapper).
- Phase 5-7: Secrets, VM, bootstrap.
- Phase 8+: Discovery at scale + product.

## Operational Protocol (spec §12)

See [SESSION_START.md](SESSION_START.md) and [ENVIRONMENT.md](ENVIRONMENT.md).

Key commands on the Linux VM (after setting aliases):
- `app`
- `git pull`
- `app-verify`
- ... work ...
- `app-push`
- `git commit && git push`
- `az vm deallocate ...` (or equivalent)

## Current Status of this Scaffold

- Core paths, singleton DB, secrets, object storage (with dev fallback) implemented.
- Registry with enforced schema + repair/update/finder scripts.
- Source ingestor for `has_download_link` (CSV/XLSX/ZIP + PDF via document parser) + schema mapper (with debug prints).
- Strong local fetcher with multi-strategy anti-bot (requests_stealth + Playwright fallbacks).
- Full document parser with pdfplumber → OCR fallback + automatic review queueing.
- Rich run logging + explicit column mapper utility.
- All other agents as clear stubs with TODOs and conventions.
- Demo that runs end-to-end and proves the "first real dataset" milestone.
- **Local GUI** (`gui/app.py`) for managing sources, triggering scrapes, exploring data, and viewing logs — all 100% local.
- Full requirements split, .env.template, ENVIRONMENT, SESSION_START, this README.

## Next Steps (recommended)

1. Run the demo, inspect the loaded records.
2. Add 1-2 more real (or synthetic) sources to the registry using `registry_update.py` + repair.
3. Implement one more ingestion path (e.g. a simple `has_record_link` page scraper using requests + bs4 or playwright).
4. Wire a real search API into `source_url_finder.py` and discover one new source.
5. Specialize the scoring (enrichment_agent) and lifecycle for "grants".
6. Add a minimal Streamlit dashboard over the `records` + `calendar_events` tables.
7. Provision the cheap always-on VM + object storage + vault following ENVIRONMENT.md.
8. Move from scaffold to your real data sources.

## License / Use

Reference scaffold — free to use as the starting blueprint for any distributed-source intelligence product. Replace the demo data and the four domain points with your own.

---

*Generated from the Reference Technical Specification & Build Scaffold document.*
