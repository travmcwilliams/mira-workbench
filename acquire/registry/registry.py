"""
registry/registry.py
The Source Registry API — idempotent upserts + queries over the DuckDB schema.

Every function takes an explicit DuckDB `conn` (from utils.db.get_db() in production,
or a throwaway connection in tests). Referential integrity is code-enforced here;
natural keys (platform_name, category_name, source_url, account_url, discovered_url)
make every write idempotent so re-seeding and re-migrating are safe.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .scoring import (
    components_from_metrics,
    evaluate_inclusion,
    mira_source_score,
)

# account_id sentinel meaning "source-level, no specific account" (DuckDB keys are NOT NULL).
SOURCE_LEVEL = 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def encode_list(value: Optional[list[str]]) -> Optional[str]:
    return json.dumps(list(value)) if value else None


def decode_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    try:
        out = json.loads(value)
        return out if isinstance(out, list) else []
    except (ValueError, TypeError):
        return []


def _scalar(conn: Any, sql: str, params: list[Any] | None = None) -> Any:
    row = conn.execute(sql, params or []).fetchone()
    return row[0] if row else None


def _touch(conn: Any, table: str, id_col: str, id_val: Any) -> None:
    conn.execute(f"UPDATE {table} SET updated_at = CURRENT_TIMESTAMP WHERE {id_col} = ?", [id_val])


# --------------------------------------------------------------------------- #
# platforms / categories
# --------------------------------------------------------------------------- #

def upsert_platform(conn: Any, *, platform_name: str, platform_type: str = "news_site",
                    **fields: Any) -> int:
    """Insert or update a platform by its unique name. Returns platform_id."""
    existing = _scalar(conn, "SELECT platform_id FROM source_platform WHERE platform_name = ?",
                       [platform_name])
    cols = {
        "parent_company", "homepage_url", "api_available", "api_documentation_url",
        "rss_supported", "robots_policy_required", "auth_required",
        "crawl_method_default", "notes",
    }
    data = {k: v for k, v in fields.items() if k in cols}
    if existing is not None:
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            conn.execute(f"UPDATE source_platform SET {sets} WHERE platform_id = ?",
                         list(data.values()) + [existing])
            _touch(conn, "source_platform", "platform_id", existing)
        return int(existing)
    data.update(platform_name=platform_name, platform_type=platform_type)
    keys = list(data.keys())
    ph = ", ".join("?" for _ in keys)
    new_id = _scalar(conn,
                     f"INSERT INTO source_platform ({', '.join(keys)}) VALUES ({ph}) RETURNING platform_id",
                     [data[k] for k in keys])
    return int(new_id)


def upsert_category(conn: Any, *, name: str, parent_category_id: Optional[int] = None,
                    description: Optional[str] = None) -> int:
    existing = _scalar(conn, "SELECT category_id FROM source_category WHERE category_name = ?", [name])
    if existing is not None:
        return int(existing)
    return int(_scalar(
        conn,
        "INSERT INTO source_category (category_name, parent_category_id, description) "
        "VALUES (?, ?, ?) RETURNING category_id",
        [name, parent_category_id, description],
    ))


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #

_SOURCE_COLS = {
    "canonical_name", "display_name", "source_type", "platform_id", "parent_org",
    "ownership_type", "country_code", "region", "primary_language",
    "political_orientation", "rss_url", "api_endpoint", "sitemap_url",
    "wikipedia_url", "wikidata_id", "media_bias_fact_check_url", "all_sides_url",
    "fact_check_status", "paywall_status", "official_government",
    "official_incident_feed", "academic_repository", "active",
    "start_tracking_date", "end_tracking_date", "notes",
}


def upsert_source(conn: Any, *, source_url: str, canonical_name: str,
                  source_type: str = "unknown",
                  topical_focus: Optional[list[str]] = None, **fields: Any) -> int:
    """Insert or update a source by its unique source_url. Returns source_id."""
    data = {k: v for k, v in fields.items() if k in _SOURCE_COLS}
    data["canonical_name"] = canonical_name
    data["source_type"] = source_type
    if topical_focus is not None:
        data["topical_focus"] = encode_list(topical_focus)

    existing = _scalar(conn, "SELECT source_id FROM source_entity WHERE source_url = ?", [source_url])
    if existing is not None:
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            conn.execute(f"UPDATE source_entity SET {sets} WHERE source_id = ?",
                         list(data.values()) + [existing])
            _touch(conn, "source_entity", "source_id", existing)
        return int(existing)

    data["source_url"] = source_url
    keys = list(data.keys())
    ph = ", ".join("?" for _ in keys)
    return int(_scalar(
        conn,
        f"INSERT INTO source_entity ({', '.join(keys)}) VALUES ({ph}) RETURNING source_id",
        [data[k] for k in keys],
    ))


def map_source_category(conn: Any, source_id: int, category_id: int,
                        relevance_score: float = 1.0) -> None:
    exists = _scalar(conn,
                     "SELECT 1 FROM source_category_map WHERE source_id = ? AND category_id = ?",
                     [source_id, category_id])
    if exists:
        conn.execute("UPDATE source_category_map SET relevance_score = ? "
                     "WHERE source_id = ? AND category_id = ?",
                     [relevance_score, source_id, category_id])
    else:
        conn.execute("INSERT INTO source_category_map (source_id, category_id, relevance_score) "
                     "VALUES (?, ?, ?)", [source_id, category_id, relevance_score])


def source_categories(conn: Any, source_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT c.category_name FROM source_category_map m "
        "JOIN source_category c ON c.category_id = m.category_id WHERE m.source_id = ? "
        "ORDER BY m.relevance_score DESC", [source_id]).fetchall()
    return [r[0] for r in rows]


# --------------------------------------------------------------------------- #
# accounts
# --------------------------------------------------------------------------- #

_ACCOUNT_COLS = {
    "source_id", "platform_id", "account_handle", "account_external_id", "verified",
    "follower_count", "subscriber_count", "average_views_30d", "average_engagement_30d",
    "last_metric_refresh", "active",
}


def add_account(conn: Any, *, account_url: str, **fields: Any) -> int:
    data = {k: v for k, v in fields.items() if k in _ACCOUNT_COLS}
    existing = _scalar(conn, "SELECT account_id FROM source_account WHERE account_url = ?", [account_url])
    if existing is not None:
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            conn.execute(f"UPDATE source_account SET {sets} WHERE account_id = ?",
                         list(data.values()) + [existing])
            _touch(conn, "source_account", "account_id", existing)
        return int(existing)
    data["account_url"] = account_url
    keys = list(data.keys())
    ph = ", ".join("?" for _ in keys)
    return int(_scalar(
        conn,
        f"INSERT INTO source_account ({', '.join(keys)}) VALUES ({ph}) RETURNING account_id",
        [data[k] for k in keys],
    ))


# --------------------------------------------------------------------------- #
# crawl policy + access rule
# --------------------------------------------------------------------------- #

_CRAWL_COLS = {
    "crawl_enabled", "crawl_method", "crawl_frequency", "historical_backfill_enabled",
    "backfill_start_date", "backfill_end_date", "respect_robots_txt", "requires_api",
    "requires_auth", "rate_limit_per_minute", "max_pages_per_run", "max_depth",
    "javascript_required", "screenshot_required", "transcript_required",
    "media_download_allowed", "terms_risk_level", "priority",
}


def set_crawl_policy(conn: Any, source_id: int, *, account_id: int = SOURCE_LEVEL,
                     crawl_method: str = "html", crawl_frequency: str = "daily",
                     **fields: Any) -> int:
    data = {k: v for k, v in fields.items() if k in _CRAWL_COLS}
    data["crawl_method"] = crawl_method
    data["crawl_frequency"] = crawl_frequency
    existing = _scalar(conn,
                       "SELECT crawl_policy_id FROM crawl_policy WHERE source_id = ? AND account_id = ?",
                       [source_id, account_id])
    if existing is not None:
        sets = ", ".join(f"{k} = ?" for k in data)
        conn.execute(f"UPDATE crawl_policy SET {sets} WHERE crawl_policy_id = ?",
                     list(data.values()) + [existing])
        _touch(conn, "crawl_policy", "crawl_policy_id", existing)
        return int(existing)
    data["source_id"] = source_id
    data["account_id"] = account_id
    keys = list(data.keys())
    ph = ", ".join("?" for _ in keys)
    return int(_scalar(
        conn,
        f"INSERT INTO crawl_policy ({', '.join(keys)}) VALUES ({ph}) RETURNING crawl_policy_id",
        [data[k] for k in keys],
    ))


_ACCESS_COLS = {
    "robots_txt_url", "robots_allowed", "terms_url", "api_required", "scraping_allowed",
    "commercial_use_allowed", "redistribution_allowed", "login_required", "paywall",
    "personal_data_risk", "legal_risk_level", "last_reviewed_at", "reviewer", "notes",
}


def set_access_rule(conn: Any, source_id: int, **fields: Any) -> int:
    data = {k: v for k, v in fields.items() if k in _ACCESS_COLS}
    existing = _scalar(conn, "SELECT access_rule_id FROM source_access_rule WHERE source_id = ?",
                       [source_id])
    if existing is not None:
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            conn.execute(f"UPDATE source_access_rule SET {sets} WHERE access_rule_id = ?",
                         list(data.values()) + [existing])
        return int(existing)
    data["source_id"] = source_id
    keys = list(data.keys())
    ph = ", ".join("?" for _ in keys)
    return int(_scalar(
        conn,
        f"INSERT INTO source_access_rule ({', '.join(keys)}) VALUES ({ph}) RETURNING access_rule_id",
        [data[k] for k in keys],
    ))


# --------------------------------------------------------------------------- #
# inclusion rules
# --------------------------------------------------------------------------- #

_RULE_COLS = {
    "source_type", "platform_type", "category_name", "country_code",
    "min_monthly_visits", "min_followers", "min_subscribers", "min_avg_views",
    "min_avg_engagement", "min_mentions", "min_citation_count", "min_authority_score",
    "include_if_government", "include_if_academic", "include_if_incident_feed",
    "include_if_local_official", "priority", "active",
}


def add_inclusion_rule(conn: Any, *, rule_name: str, **fields: Any) -> int:
    data = {k: v for k, v in fields.items() if k in _RULE_COLS}
    existing = _scalar(conn, "SELECT rule_id FROM inclusion_rule WHERE rule_name = ?", [rule_name])
    if existing is not None:
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            conn.execute(f"UPDATE inclusion_rule SET {sets} WHERE rule_id = ?",
                         list(data.values()) + [existing])
        return int(existing)
    data["rule_name"] = rule_name
    keys = list(data.keys())
    ph = ", ".join("?" for _ in keys)
    return int(_scalar(
        conn,
        f"INSERT INTO inclusion_rule ({', '.join(keys)}) VALUES ({ph}) RETURNING rule_id",
        [data[k] for k in keys],
    ))


def load_inclusion_rules(conn: Any) -> list[dict[str, Any]]:
    return conn.execute("SELECT * FROM inclusion_rule WHERE active").df().to_dict("records")


# --------------------------------------------------------------------------- #
# metrics + scoring
# --------------------------------------------------------------------------- #

_METRIC_COLS = {
    "monthly_visits", "global_rank", "country_rank", "category_rank", "follower_count",
    "subscriber_count", "avg_views", "avg_likes", "avg_comments", "avg_shares",
    "avg_retweets", "avg_mentions", "engagement_rate", "citation_count", "inbound_links",
    "source_authority_score", "viral_velocity_score", "credibility_score",
    "bias_confidence_score",
}


def add_metric_snapshot(conn: Any, source_id: int, snapshot_date: str, *,
                        account_id: int = SOURCE_LEVEL, data_source: str = "manual",
                        config: dict[str, Any] | None = None,
                        topical_relevance: float = 0.5,
                        age_days: Optional[float] = None,
                        **metrics: Any) -> float:
    """
    Record a metric snapshot and compute + persist its Mira Source Score.
    Returns the computed score. Idempotent on (source_id, account_id, date, data_source).
    """
    data = {k: v for k, v in metrics.items() if k in _METRIC_COLS}

    score = None
    if config:
        comps = components_from_metrics(
            data, config.get("norm_caps", {}),
            legal_risk_level=metrics.get("legal_risk_level", "unknown"),
            topical_relevance=topical_relevance, age_days=age_days,
        )
        score = mira_source_score(comps, config["scoring_weights"])

    conn.execute("DELETE FROM source_metric_snapshot WHERE source_id = ? AND account_id = ? "
                 "AND snapshot_date = ? AND data_source = ?",
                 [source_id, account_id, snapshot_date, data_source])
    data.update(source_id=source_id, account_id=account_id, snapshot_date=snapshot_date,
                data_source=data_source, mira_source_score=score)
    keys = list(data.keys())
    ph = ", ".join("?" for _ in keys)
    conn.execute(f"INSERT INTO source_metric_snapshot ({', '.join(keys)}) VALUES ({ph})",
                 [data[k] for k in keys])
    return score if score is not None else 0.0


def evaluate_source_inclusion(conn: Any, source_id: int, config: dict[str, Any]):
    """Fetch a source + its latest metrics and run the inclusion rules."""
    row = conn.execute("SELECT * FROM source_entity WHERE source_id = ?", [source_id]).df()
    if row.empty:
        raise ValueError(f"source_id {source_id} not found")
    src = row.iloc[0].to_dict()
    src["categories"] = source_categories(conn, source_id)

    metric = conn.execute(
        "SELECT * FROM source_metric_snapshot WHERE source_id = ? AND account_id = ? "
        "ORDER BY snapshot_date DESC LIMIT 1", [source_id, SOURCE_LEVEL]).df()
    metrics = metric.iloc[0].to_dict() if not metric.empty else {}

    rules = config.get("inclusion_rules") or load_inclusion_rules(conn)
    return evaluate_inclusion(src, metrics, rules)


# --------------------------------------------------------------------------- #
# discovery candidates
# --------------------------------------------------------------------------- #

def add_discovery_candidate(conn: Any, *, discovered_url: str, **fields: Any) -> int:
    """Upsert a candidate by url; bumps mention_count + last_seen on re-sighting."""
    cols = {
        "discovered_name", "discovered_platform", "discovered_from", "discovery_method",
        "estimated_monthly_visits", "estimated_followers", "estimated_engagement",
        "candidate_score", "review_status", "notes",
    }
    data = {k: v for k, v in fields.items() if k in cols}
    existing = _scalar(conn,
                       "SELECT candidate_id FROM source_discovery_candidate WHERE discovered_url = ?",
                       [discovered_url])
    if existing is not None:
        sets = ", ".join(f"{k} = ?" for k in data) + (", " if data else "")
        conn.execute(
            f"UPDATE source_discovery_candidate SET {sets}"
            "mention_count = mention_count + 1, last_seen_at = CURRENT_TIMESTAMP "
            "WHERE candidate_id = ?", list(data.values()) + [existing])
        return int(existing)
    data["discovered_url"] = discovered_url
    keys = list(data.keys())
    ph = ", ".join("?" for _ in keys)
    return int(_scalar(
        conn,
        f"INSERT INTO source_discovery_candidate ({', '.join(keys)}) VALUES ({ph}) RETURNING candidate_id",
        [data[k] for k in keys],
    ))


def promote_candidate(conn: Any, candidate_id: int, *, canonical_name: str,
                      source_type: str = "unknown", **source_fields: Any) -> int:
    """Promote a reviewed candidate into a real source; links the two records."""
    cand = conn.execute("SELECT discovered_url FROM source_discovery_candidate WHERE candidate_id = ?",
                        [candidate_id]).fetchone()
    if not cand:
        raise ValueError(f"candidate_id {candidate_id} not found")
    source_id = upsert_source(conn, source_url=cand[0], canonical_name=canonical_name,
                              source_type=source_type, **source_fields)
    conn.execute("UPDATE source_discovery_candidate SET review_status = 'promoted', "
                 "promoted_source_id = ? WHERE candidate_id = ?", [source_id, candidate_id])
    return source_id


# --------------------------------------------------------------------------- #
# coverage + events
# --------------------------------------------------------------------------- #

def record_coverage(conn: Any, source_id: int, coverage_date: str, *,
                    account_id: int = SOURCE_LEVEL, expected_items: Optional[int] = None,
                    discovered_items: int = 0, retrieved_items: int = 0,
                    failed_items: int = 0, backfill_complete: bool = False,
                    notes: Optional[str] = None) -> float:
    """Upsert a day's coverage row; coverage_score = retrieved / expected (0..1)."""
    denom = expected_items if expected_items else max(discovered_items, retrieved_items, 1)
    coverage_score = round(min(retrieved_items / denom, 1.0), 4) if denom else 0.0
    conn.execute("DELETE FROM source_daily_coverage WHERE source_id = ? AND account_id = ? "
                 "AND coverage_date = ?", [source_id, account_id, coverage_date])
    conn.execute(
        "INSERT INTO source_daily_coverage (source_id, account_id, coverage_date, expected_items, "
        "discovered_items, retrieved_items, failed_items, coverage_score, backfill_complete, "
        "last_attempted_at, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
        [source_id, account_id, coverage_date, expected_items, discovered_items,
         retrieved_items, failed_items, coverage_score, backfill_complete, notes])
    return coverage_score


def create_event(conn: Any, *, event_type: str, event_name: Optional[str] = None,
                 **fields: Any) -> int:
    cols = {"event_name", "event_date", "start_time", "end_time", "country_code", "region",
            "city", "latitude", "longitude", "severity_score", "confidence_score",
            "narrative_summary", "status"}
    data = {k: v for k, v in fields.items() if k in cols}
    data["event_type"] = event_type
    data["event_name"] = event_name
    keys = list(data.keys())
    ph = ", ".join("?" for _ in keys)
    return int(_scalar(
        conn, f"INSERT INTO event_cluster ({', '.join(keys)}) VALUES ({ph}) RETURNING event_id",
        [data[k] for k in keys]))


def link_event_source(conn: Any, event_id: int, source_id: int, *,
                      relevance_score: Optional[float] = None, first_report: bool = False,
                      primary_source: bool = False) -> None:
    exists = _scalar(conn, "SELECT 1 FROM event_source_map WHERE event_id = ? AND source_id = ?",
                     [event_id, source_id])
    if exists:
        conn.execute("UPDATE event_source_map SET relevance_score = ?, first_report = ?, "
                     "primary_source = ? WHERE event_id = ? AND source_id = ?",
                     [relevance_score, first_report, primary_source, event_id, source_id])
    else:
        conn.execute("INSERT INTO event_source_map (event_id, source_id, relevance_score, "
                     "first_report, primary_source) VALUES (?, ?, ?, ?, ?)",
                     [event_id, source_id, relevance_score, first_report, primary_source])


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #

def list_sources(conn: Any, *, active_only: bool = True):
    sql = "SELECT * FROM source_entity"
    if active_only:
        sql += " WHERE active"
    sql += " ORDER BY canonical_name"
    return conn.execute(sql).df()


def get_source_by_url(conn: Any, source_url: str) -> Optional[dict[str, Any]]:
    df = conn.execute("SELECT * FROM source_entity WHERE source_url = ?", [source_url]).df()
    return df.iloc[0].to_dict() if not df.empty else None


def top_sources_by_score(conn: Any, *, limit: int = 25):
    return conn.execute("""
        SELECT e.source_id, e.canonical_name, e.source_type, e.country_code,
               m.mira_source_score, m.snapshot_date
        FROM source_entity e
        LEFT JOIN (
            SELECT source_id, mira_source_score, snapshot_date,
                   ROW_NUMBER() OVER (PARTITION BY source_id ORDER BY snapshot_date DESC) AS rn
            FROM source_metric_snapshot WHERE account_id = 0
        ) m ON m.source_id = e.source_id AND m.rn = 1
        WHERE e.active
        ORDER BY m.mira_source_score DESC NULLS LAST, e.canonical_name
        LIMIT ?
    """, [limit]).df()


def sources_due_for_crawl(conn: Any, frequency: str):
    return conn.execute("""
        SELECT e.source_id, e.canonical_name, e.source_url, p.crawl_method, p.crawl_frequency,
               p.terms_risk_level
        FROM source_entity e
        JOIN crawl_policy p ON p.source_id = e.source_id AND p.account_id = 0
        WHERE e.active AND p.crawl_enabled AND p.crawl_frequency = ?
        ORDER BY p.priority, e.canonical_name
    """, [frequency]).df()


def registry_counts(conn: Any) -> dict[str, int]:
    tables = ["source_platform", "source_category", "source_entity", "source_account",
              "crawl_policy", "source_access_rule", "inclusion_rule",
              "source_metric_snapshot", "source_discovery_candidate",
              "source_daily_coverage", "event_cluster", "event_source_map"]
    return {t: int(_scalar(conn, f"SELECT COUNT(*) FROM {t}") or 0) for t in tables}
