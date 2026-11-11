"""Offline tests for the Mira Source Registry (registry/*). No network, no shared DB.

Each test gets a throwaway in-memory DuckDB connection passed explicitly to the
registry API, so nothing here touches acquire/db/app.duckdb.
"""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("yaml")

import duckdb  # noqa: E402

from core.acquire_bridge import ensure_acquire_env  # noqa: E402

ensure_acquire_env()  # put acquire/ on sys.path so `registry` is importable

from registry import (  # noqa: E402
    default_config,
    init_registry_schema,
    migrate_json_registry,
    seed_from_config,
    validate_weights,
)
from registry import registry as R  # noqa: E402
from registry.scoring import (  # noqa: E402
    InclusionResult,
    components_from_metrics,
    evaluate_inclusion,
    freshness_from_age,
    mira_source_score,
)


@pytest.fixture()
def conn():
    c = duckdb.connect(":memory:")
    init_registry_schema(c)
    yield c
    c.close()


@pytest.fixture()
def cfg():
    return default_config()


# --- schema --------------------------------------------------------------------

def test_schema_init_is_idempotent(conn):
    init_registry_schema(conn)  # second call must not raise
    counts = R.registry_counts(conn)
    assert counts["source_entity"] == 0
    assert "event_source_map" in counts


# --- weights / scoring ---------------------------------------------------------

def test_validate_weights_rejects_bad_sum():
    with pytest.raises(ValueError):
        validate_weights({"authority": 0.9, "readership": 0.9, "engagement": 0.0,
                          "citation": 0.0, "freshness": 0.0, "topical_relevance": 0.0,
                          "accessibility": 0.0})


def test_default_config_weights_valid(cfg):
    validate_weights(cfg["scoring_weights"])  # must not raise


def test_score_is_bounded_and_monotone(cfg):
    caps = cfg["norm_caps"]
    weights = cfg["scoring_weights"]
    low = components_from_metrics({"monthly_visits": 1000}, caps, legal_risk_level="high")
    high = components_from_metrics(
        {"monthly_visits": 900_000_000, "source_authority_score": 95,
         "citation_count": 500_000, "engagement_rate": 0.18},
        caps, legal_risk_level="low", topical_relevance=1.0, age_days=0,
    )
    s_low = mira_source_score(low, weights)
    s_high = mira_source_score(high, weights)
    assert 0.0 <= s_low <= 100.0
    assert 0.0 <= s_high <= 100.0
    assert s_high > s_low


def test_freshness_decays():
    assert freshness_from_age(0, 30) == 1.0
    assert freshness_from_age(30, 30) == pytest.approx(0.5)
    assert freshness_from_age(None, 30) == 0.0


# --- inclusion -----------------------------------------------------------------

def test_government_override_includes_regardless(cfg):
    src = {"official_government": True, "source_type": "government_agency", "categories": []}
    res = evaluate_inclusion(src, {}, cfg["inclusion_rules"])
    assert isinstance(res, InclusionResult)
    assert res.included is True
    assert "government" in res.reason


def test_social_account_below_threshold_excluded(cfg):
    src = {"source_type": "social_platform_account", "categories": ["social_viral"]}
    res = evaluate_inclusion(src, {"follower_count": 5_000, "avg_views": 1_000}, cfg["inclusion_rules"])
    assert res.included is False


def test_social_account_above_threshold_included(cfg):
    src = {"source_type": "social_platform_account", "categories": ["social_viral"]}
    res = evaluate_inclusion(src, {"follower_count": 250_000}, cfg["inclusion_rules"])
    assert res.included is True
    assert "social_account_reach" in (res.rule_name or "")


# --- seeding -------------------------------------------------------------------

def test_seed_from_config_populates_and_is_idempotent(conn, cfg):
    stats = seed_from_config(conn, cfg)
    assert stats["sources"] >= 5
    assert stats["platforms"] >= 5
    c1 = R.registry_counts(conn)
    assert c1["source_entity"] == stats["sources"]
    assert c1["inclusion_rule"] >= 5

    seed_from_config(conn, cfg)  # re-seed
    c2 = R.registry_counts(conn)
    assert c2["source_entity"] == c1["source_entity"]  # no duplicates
    assert c2["source_platform"] == c1["source_platform"]


def test_seeded_source_has_categories_and_policy(conn, cfg):
    seed_from_config(conn, cfg)
    src = R.get_source_by_url(conn, "https://arxiv.org")
    assert src is not None
    assert src["academic_repository"] is True
    cats = R.source_categories(conn, int(src["source_id"]))
    assert "studies_research" in cats
    due = R.sources_due_for_crawl(conn, "daily")
    assert "arXiv" in set(due["canonical_name"])


# --- upsert semantics ----------------------------------------------------------

def test_upsert_source_idempotent_and_updates(conn):
    sid1 = R.upsert_source(conn, source_url="https://ex.test/a", canonical_name="A",
                           source_type="newspaper", country_code="US")
    sid2 = R.upsert_source(conn, source_url="https://ex.test/a", canonical_name="A2")
    assert sid1 == sid2
    row = R.get_source_by_url(conn, "https://ex.test/a")
    assert row["canonical_name"] == "A2"  # update applied
    assert R.registry_counts(conn)["source_entity"] == 1


def test_topical_focus_roundtrip(conn):
    sid = R.upsert_source(conn, source_url="https://ex.test/b", canonical_name="B",
                          topical_focus=["ai", "markets"])
    row = R.get_source_by_url(conn, "https://ex.test/b")
    assert R.decode_list(row["topical_focus"]) == ["ai", "markets"]


# --- metrics + score persistence ----------------------------------------------

def test_add_metric_snapshot_persists_score(conn, cfg):
    sid = R.upsert_source(conn, source_url="https://ex.test/news", canonical_name="News",
                          source_type="newspaper")
    score = R.add_metric_snapshot(conn, sid, "2026-07-01", data_source="test", config=cfg,
                                  monthly_visits=50_000_000, source_authority_score=80,
                                  citation_count=100_000, engagement_rate=0.05,
                                  topical_relevance=0.9, age_days=1)
    assert 0.0 < score <= 100.0
    top = R.top_sources_by_score(conn, limit=5)
    assert float(top.iloc[0]["mira_source_score"]) == pytest.approx(score)


# --- discovery candidates ------------------------------------------------------

def test_candidate_upsert_bumps_mentions_then_promotes(conn):
    cid = R.add_discovery_candidate(conn, discovered_url="https://new.test/x",
                                    discovered_name="X", discovery_method="gdelt_source")
    R.add_discovery_candidate(conn, discovered_url="https://new.test/x",
                              discovery_method="gdelt_source")
    row = conn.execute("SELECT mention_count FROM source_discovery_candidate WHERE candidate_id = ?",
                       [cid]).fetchone()
    assert row[0] == 2

    sid = R.promote_candidate(conn, cid, canonical_name="X News", source_type="digital_native_news")
    assert R.get_source_by_url(conn, "https://new.test/x") is not None
    status = conn.execute("SELECT review_status, promoted_source_id FROM source_discovery_candidate "
                          "WHERE candidate_id = ?", [cid]).fetchone()
    assert status[0] == "promoted"
    assert status[1] == sid


# --- coverage + events ---------------------------------------------------------

def test_coverage_score_and_event_link(conn):
    sid = R.upsert_source(conn, source_url="https://ex.test/cov", canonical_name="Cov")
    cs = R.record_coverage(conn, sid, "2026-07-01", expected_items=100,
                           discovered_items=90, retrieved_items=80)
    assert cs == pytest.approx(0.8)

    eid = R.create_event(conn, event_type="natural_disaster", event_name="Quake",
                         severity_score=6.1)
    R.link_event_source(conn, eid, sid, relevance_score=0.9, first_report=True)
    n = conn.execute("SELECT COUNT(*) FROM event_source_map WHERE event_id = ?", [eid]).fetchone()[0]
    assert n == 1


# --- legacy migration ----------------------------------------------------------

def test_migrate_json_registry_skips_demo_and_is_idempotent(conn):
    r1 = migrate_json_registry(conn)
    assert r1["migrated"] >= 1        # at least the real airport-traffic source
    assert r1["skipped"] >= 1         # example-data.local demo rows dropped
    n1 = R.registry_counts(conn)["source_entity"]

    r2 = migrate_json_registry(conn)  # again
    n2 = R.registry_counts(conn)["source_entity"]
    assert n2 == n1                   # no duplicates on re-run

    # migrated rows carry a crawl policy
    assert R.registry_counts(conn)["crawl_policy"] >= 1
