"""
registry/schema.py
DuckDB DDL for the Mira Source Registry.

Reconciles the reference Postgres design (a governed source-intelligence registry)
onto Mira's local-first embedded **DuckDB** stack. Type mapping applied:

    Postgres            ->  DuckDB
    BIGSERIAL PRIMARY KEY -> BIGINT PRIMARY KEY DEFAULT nextval('seq_...')
    TIMESTAMPTZ          ->  TIMESTAMP
    TEXT[]               ->  TEXT (JSON-encoded list; see registry.encode_list)
    now()                ->  CURRENT_TIMESTAMP

Design choices for the embedded, single-connection, code-mediated store:
  * Surrogate ids come from DuckDB SEQUENCEs; natural keys carry UNIQUE constraints
    so upserts are idempotent (re-seeding / re-migrating is safe).
  * Referential integrity is enforced in code (registry.py), not via FK constraints,
    to avoid DuckDB delete/insert-ordering friction. Logical parents are documented.
  * account_id uses sentinel 0 to mean "source-level, no specific account" wherever it
    participates in a composite key (metric snapshots, coverage) — DuckDB keys are NOT NULL.

The content layer (content_item / content_metric_snapshot / event_content_map) is
intentionally deferred: raw crawled content already flows through Mira's `records`
table and the news pipeline. This module governs SOURCES, not content. event_source_map
provides the registry-side event linkage until the content layer lands.
"""

from __future__ import annotations

from typing import Any

# Table name -> DDL. Ordered: sequences first, then tables (parents before children).
_SEQUENCES = [
    "seq_platform",
    "seq_category",
    "seq_source",
    "seq_account",
    "seq_crawl_policy",
    "seq_access_rule",
    "seq_inclusion_rule",
    "seq_metric",
    "seq_candidate",
    "seq_event",
]

_TABLES: dict[str, str] = {
    "source_platform": """
        CREATE TABLE IF NOT EXISTS source_platform (
            platform_id BIGINT PRIMARY KEY DEFAULT nextval('seq_platform'),
            platform_name TEXT NOT NULL UNIQUE,
            platform_type TEXT NOT NULL,
            parent_company TEXT,
            homepage_url TEXT,
            api_available BOOLEAN DEFAULT FALSE,
            api_documentation_url TEXT,
            rss_supported BOOLEAN DEFAULT FALSE,
            robots_policy_required BOOLEAN DEFAULT TRUE,
            auth_required BOOLEAN DEFAULT FALSE,
            crawl_method_default TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "source_category": """
        CREATE TABLE IF NOT EXISTS source_category (
            category_id BIGINT PRIMARY KEY DEFAULT nextval('seq_category'),
            category_name TEXT NOT NULL UNIQUE,
            parent_category_id BIGINT,
            description TEXT
        );
    """,
    "source_entity": """
        CREATE TABLE IF NOT EXISTS source_entity (
            source_id BIGINT PRIMARY KEY DEFAULT nextval('seq_source'),
            canonical_name TEXT NOT NULL,
            display_name TEXT,
            source_type TEXT NOT NULL,
            platform_id BIGINT,
            parent_org TEXT,
            ownership_type TEXT,
            country_code TEXT,
            region TEXT,
            primary_language TEXT,
            political_orientation TEXT,
            topical_focus TEXT,                 -- JSON-encoded list
            source_url TEXT NOT NULL UNIQUE,
            rss_url TEXT,
            api_endpoint TEXT,
            sitemap_url TEXT,
            wikipedia_url TEXT,
            wikidata_id TEXT,
            media_bias_fact_check_url TEXT,
            all_sides_url TEXT,
            fact_check_status TEXT,
            paywall_status TEXT,
            official_government BOOLEAN DEFAULT FALSE,
            official_incident_feed BOOLEAN DEFAULT FALSE,
            academic_repository BOOLEAN DEFAULT FALSE,
            active BOOLEAN DEFAULT TRUE,
            start_tracking_date DATE,
            end_tracking_date DATE,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "source_account": """
        CREATE TABLE IF NOT EXISTS source_account (
            account_id BIGINT PRIMARY KEY DEFAULT nextval('seq_account'),
            source_id BIGINT,
            platform_id BIGINT,
            account_handle TEXT,
            account_url TEXT NOT NULL UNIQUE,
            account_external_id TEXT,
            verified BOOLEAN,
            follower_count BIGINT,
            subscriber_count BIGINT,
            average_views_30d BIGINT,
            average_engagement_30d DOUBLE,
            last_metric_refresh TIMESTAMP,
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "source_category_map": """
        CREATE TABLE IF NOT EXISTS source_category_map (
            source_id BIGINT,
            category_id BIGINT,
            relevance_score DOUBLE DEFAULT 1.0,
            PRIMARY KEY (source_id, category_id)
        );
    """,
    "crawl_policy": """
        CREATE TABLE IF NOT EXISTS crawl_policy (
            crawl_policy_id BIGINT PRIMARY KEY DEFAULT nextval('seq_crawl_policy'),
            source_id BIGINT NOT NULL,
            account_id BIGINT NOT NULL DEFAULT 0,   -- 0 = source-level
            crawl_enabled BOOLEAN DEFAULT TRUE,
            crawl_method TEXT NOT NULL,
            crawl_frequency TEXT NOT NULL,
            historical_backfill_enabled BOOLEAN DEFAULT TRUE,
            backfill_start_date DATE,
            backfill_end_date DATE,
            respect_robots_txt BOOLEAN DEFAULT TRUE,
            requires_api BOOLEAN DEFAULT FALSE,
            requires_auth BOOLEAN DEFAULT FALSE,
            rate_limit_per_minute INTEGER,
            max_pages_per_run INTEGER,
            max_depth INTEGER,
            javascript_required BOOLEAN DEFAULT FALSE,
            screenshot_required BOOLEAN DEFAULT FALSE,
            transcript_required BOOLEAN DEFAULT FALSE,
            media_download_allowed BOOLEAN DEFAULT FALSE,
            terms_risk_level TEXT DEFAULT 'unknown',
            priority INTEGER DEFAULT 100,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (source_id, account_id)
        );
    """,
    "source_access_rule": """
        CREATE TABLE IF NOT EXISTS source_access_rule (
            access_rule_id BIGINT PRIMARY KEY DEFAULT nextval('seq_access_rule'),
            source_id BIGINT NOT NULL UNIQUE,
            robots_txt_url TEXT,
            robots_allowed BOOLEAN,
            terms_url TEXT,
            api_required BOOLEAN DEFAULT FALSE,
            scraping_allowed BOOLEAN,
            commercial_use_allowed BOOLEAN,
            redistribution_allowed BOOLEAN,
            login_required BOOLEAN DEFAULT FALSE,
            paywall BOOLEAN DEFAULT FALSE,
            personal_data_risk TEXT DEFAULT 'unknown',
            legal_risk_level TEXT DEFAULT 'unknown',
            last_reviewed_at TIMESTAMP,
            reviewer TEXT,
            notes TEXT
        );
    """,
    "inclusion_rule": """
        CREATE TABLE IF NOT EXISTS inclusion_rule (
            rule_id BIGINT PRIMARY KEY DEFAULT nextval('seq_inclusion_rule'),
            rule_name TEXT NOT NULL,
            source_type TEXT,
            platform_type TEXT,
            category_name TEXT,
            country_code TEXT,
            min_monthly_visits BIGINT,
            min_followers BIGINT,
            min_subscribers BIGINT,
            min_avg_views BIGINT,
            min_avg_engagement DOUBLE,
            min_mentions BIGINT,
            min_citation_count BIGINT,
            min_authority_score DOUBLE,
            include_if_government BOOLEAN DEFAULT FALSE,
            include_if_academic BOOLEAN DEFAULT FALSE,
            include_if_incident_feed BOOLEAN DEFAULT FALSE,
            include_if_local_official BOOLEAN DEFAULT FALSE,
            priority INTEGER DEFAULT 100,
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "source_metric_snapshot": """
        CREATE TABLE IF NOT EXISTS source_metric_snapshot (
            metric_id BIGINT PRIMARY KEY DEFAULT nextval('seq_metric'),
            source_id BIGINT NOT NULL,
            account_id BIGINT NOT NULL DEFAULT 0,
            snapshot_date DATE NOT NULL,
            monthly_visits BIGINT,
            global_rank BIGINT,
            country_rank BIGINT,
            category_rank BIGINT,
            follower_count BIGINT,
            subscriber_count BIGINT,
            avg_views BIGINT,
            avg_likes BIGINT,
            avg_comments BIGINT,
            avg_shares BIGINT,
            avg_retweets BIGINT,
            avg_mentions BIGINT,
            engagement_rate DOUBLE,
            citation_count BIGINT,
            inbound_links BIGINT,
            source_authority_score DOUBLE,
            viral_velocity_score DOUBLE,
            credibility_score DOUBLE,
            bias_confidence_score DOUBLE,
            mira_source_score DOUBLE,
            data_source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (source_id, account_id, snapshot_date, data_source)
        );
    """,
    "source_discovery_candidate": """
        CREATE TABLE IF NOT EXISTS source_discovery_candidate (
            candidate_id BIGINT PRIMARY KEY DEFAULT nextval('seq_candidate'),
            discovered_name TEXT,
            discovered_url TEXT NOT NULL UNIQUE,
            discovered_platform TEXT,
            discovered_from TEXT,
            discovery_method TEXT,
            first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            mention_count BIGINT DEFAULT 1,
            estimated_monthly_visits BIGINT,
            estimated_followers BIGINT,
            estimated_engagement DOUBLE,
            candidate_score DOUBLE,
            review_status TEXT DEFAULT 'pending',
            promoted_source_id BIGINT,
            notes TEXT
        );
    """,
    "source_daily_coverage": """
        CREATE TABLE IF NOT EXISTS source_daily_coverage (
            source_id BIGINT NOT NULL,
            account_id BIGINT NOT NULL DEFAULT 0,
            coverage_date DATE NOT NULL,
            expected_items INTEGER,
            discovered_items INTEGER,
            retrieved_items INTEGER,
            failed_items INTEGER,
            coverage_score DOUBLE,
            backfill_complete BOOLEAN DEFAULT FALSE,
            last_attempted_at TIMESTAMP,
            notes TEXT,
            PRIMARY KEY (source_id, account_id, coverage_date)
        );
    """,
    "event_cluster": """
        CREATE TABLE IF NOT EXISTS event_cluster (
            event_id BIGINT PRIMARY KEY DEFAULT nextval('seq_event'),
            event_name TEXT,
            event_type TEXT NOT NULL,
            event_date DATE,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            country_code TEXT,
            region TEXT,
            city TEXT,
            latitude DOUBLE,
            longitude DOUBLE,
            severity_score DOUBLE,
            confidence_score DOUBLE,
            narrative_summary TEXT,
            status TEXT DEFAULT 'open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "event_source_map": """
        CREATE TABLE IF NOT EXISTS event_source_map (
            event_id BIGINT NOT NULL,
            source_id BIGINT NOT NULL,
            relevance_score DOUBLE,
            first_report BOOLEAN DEFAULT FALSE,
            primary_source BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (event_id, source_id)
        );
    """,
}

# Tables that own a surrogate-id sequence — used to keep the sequence ahead of any
# ids inserted directly (e.g. when migrating rows that pre-assign ids). Not needed
# for the normal path but documented for completeness.
TABLE_NAMES = tuple(_TABLES.keys())


def init_registry_schema(conn: Any) -> None:
    """Create all registry sequences + tables if absent. Idempotent."""
    for seq in _SEQUENCES:
        conn.execute(f"CREATE SEQUENCE IF NOT EXISTS {seq} START 1;")
    for ddl in _TABLES.values():
        conn.execute(ddl)


def drop_registry_schema(conn: Any) -> None:
    """Drop all registry tables + sequences (used by rebuild / tests)."""
    for name in reversed(list(_TABLES.keys())):
        conn.execute(f"DROP TABLE IF EXISTS {name};")
    for seq in _SEQUENCES:
        conn.execute(f"DROP SEQUENCE IF EXISTS {seq};")
