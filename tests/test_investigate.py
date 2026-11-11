"""Tests for the investigation planner + orchestrator (core/investigate.py)."""

from __future__ import annotations

import pandas as pd

from core import investigate as inv

# --- planning (offline, deterministic) --------------------------------------- #

def test_offline_plan_is_deterministic():
    p1 = inv.plan_investigation("Investigate disc golf course difficulty", "investigate")
    p2 = inv.plan_investigation("Investigate disc golf course difficulty", "investigate")
    assert p1.to_dict() == p2.to_dict()
    assert p1.provider == "offline"


def test_offline_plan_extracts_topic_and_queries():
    p = inv.plan_investigation("experiment with county grant award amounts", "experiment")
    assert "grant" in p.topic or "county" in p.topic
    assert len(p.queries) >= 8
    assert all(isinstance(q, str) and q for q in p.queries)
    # stopwords stripped from fields
    assert "with" not in p.fields
    assert "the" not in p.fields


def test_query_spread_dedupes_and_biases_mode():
    q = inv.build_query_spread("rainfall", ["year", "mm"], "predict")
    assert len(q) == len(set(q))  # de-duped
    assert any("historical" in x for x in q)  # predict-mode bias


def test_invalid_mode_falls_back_to_explore():
    p = inv.plan_investigation("something", "banana")
    assert p.mode == "explore"


def test_parse_plan_json_tolerates_prose_wrapping():
    raw = 'Here is your plan:\n{"topic": "x", "queries": ["a"], "fields": ["f"]}\nDone.'
    obj = inv._parse_plan_json(raw)
    assert obj["topic"] == "x"


# --- orchestration (injected discover + scrape, fully offline) --------------- #

def _fake_discover(topic, queries, fields, *, prefer=None, target_n=100):
    return {
        "provider": "fake",
        "topic": topic,
        "n_raw": 25,
        "candidates": [
            {"url": "http://a.test/data.csv", "title": "A", "data_url": "http://a.test/data.csv"},
            {"url": "http://b.test/data.csv", "title": "B", "data_url": "http://b.test/data.csv"},
            {"url": "http://c.test/none", "title": "C", "reachable": False},  # skipped
        ],
    }


def _fake_scrape(candidate, fields):
    url = candidate["url"]
    if "a.test" in url:
        return pd.DataFrame({"Revenue": [1, 2], "Region": ["x", "y"]})
    if "b.test" in url:
        return pd.DataFrame({"revenue ": [3, 4], "region": ["z", "w"]})
    return pd.DataFrame()


def test_run_investigation_builds_project(tmp_path, monkeypatch):
    # Create projects under a temp cwd so the test doesn't litter the repo.
    monkeypatch.chdir(tmp_path)

    ticks: list[tuple[str, float]] = []
    result = inv.run_investigation(
        "investigate widget revenue by region",
        "investigate",
        discover_fn=_fake_discover,
        scrape_fn=_fake_scrape,
        progress_cb=lambda m, f: ticks.append((m, f)),
    )

    assert result.error is None
    assert result.n_candidates == 3
    assert result.n_scraped == 2          # the unreachable candidate is skipped
    assert result.n_with_rows == 2
    assert "revenue" in result.columns    # cross-source columns merged
    assert result.rows == 4               # unioned
    assert result.project_id is not None
    assert ticks and ticks[-1][1] == 1.0  # progress completed

    # The created project has the consolidated schema + investigation provenance.
    from core.project import Project
    proj = Project.load(result.project_id)
    assert proj.meta.get("investigation", {}).get("goal")
    assert proj.has_raw_data()


def test_run_investigation_no_data_reports_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _empty_discover(topic, queries, fields, *, prefer=None, target_n=100):
        return {"provider": "fake", "topic": topic, "n_raw": 0, "candidates": []}

    result = inv.run_investigation(
        "obscure goal", "explore", discover_fn=_empty_discover,
        scrape_fn=_fake_scrape,
    )
    assert result.project_id is None
    assert result.error is not None
    assert result.rows == 0


def test_run_investigation_no_create(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = inv.run_investigation(
        "widget revenue", "explore", create_project=False,
        discover_fn=_fake_discover, scrape_fn=_fake_scrape,
    )
    assert result.project_id is None
    assert result.error is None
    assert result.rows == 4
    assert result.report is not None


def test_summary_shape():
    result = inv.InvestigationResult(
        plan=inv.offline_plan("x", "explore"), provider="fake",
        n_candidates=0, n_scraped=0, n_with_rows=0, n_raw_search=0,
    )
    s = result.summary()
    for key in ("goal", "mode", "topic", "candidates", "project_id", "task_type"):
        assert key in s
