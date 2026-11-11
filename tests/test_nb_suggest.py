"""Tests for notebook method pills + context-aware suggestions (core/nb_suggest.py)."""

from __future__ import annotations

import pandas as pd

from core import nb_suggest as ns


def _df():
    return pd.DataFrame({
        "price": [10.0, 20.0, 30.0, 40.0],
        "region": ["a", "b", "a", "b"],
        "qty": [1, 2, 3, 4],
    })


_SCHEMA = {"price": "numeric", "region": "categorical", "qty": "numeric"}


# --- column selection + templating ------------------------------------------ #

def test_pick_columns_uses_schema_and_target():
    cols = ns.pick_columns(_df(), _SCHEMA, target="price")
    assert cols["TARGET"] == "price"
    assert cols["CAT"] == "region"
    assert cols["NUM"] in ("price", "qty")


def test_pick_columns_without_schema_infers_from_dtypes():
    cols = ns.pick_columns(_df())
    assert cols["NUM"] in ("price", "qty")
    assert cols["CAT"] == "region"


def test_fill_replaces_tokens():
    code = "df['{{NUM}}'].hist(); df['{{TARGET}}']"
    out = ns.fill(code, {"NUM": "price", "TARGET": "qty", "CAT": "region", "DATETIME": "d"})
    assert "{{" not in out
    assert "df['price']" in out and "df['qty']" in out


def test_library_for_fills_every_snippet():
    lib = ns.library_for(_df(), _SCHEMA, target="price")
    assert set(lib.keys()) == set(ns.METHOD_LIBRARY.keys())
    for items in lib.values():
        for m in items:
            assert "{{" not in m["code"]  # all tokens resolved


# --- retrieval (RAG) -------------------------------------------------------- #

def test_retrieve_ranks_by_overlap():
    hits = ns.retrieve("I want to find correlation drivers of the target", k=3)
    titles = [r.title for r in hits]
    assert any("orrelation" in t or "target" in t.lower() for t in titles)
    assert len(hits) == 3


def test_retrieve_is_deterministic():
    a = ns.retrieve("missing values null audit", k=4)
    b = ns.retrieve("missing values null audit", k=4)
    assert [r.title for r in a] == [r.title for r in b]


def test_build_context_includes_schema_and_cells():
    class _Cell:
        id = "c1"
        source = "df.describe()"

    class _Res:
        error = None
        stdout = "some output"

    ctx = ns.build_context(_SCHEMA, "price", "regression", 4, [_Cell()], {"c1": _Res()})
    assert "price" in ctx
    assert "regression" in ctx
    assert "df.describe()" in ctx
    assert "some output" in ctx


# --- suggestions ------------------------------------------------------------ #

def test_suggest_offline_returns_filled_recipes():
    out = ns.suggest(df=_df(), schema=_SCHEMA, target="price", task_type="regression",
                     cells=[], chat_model=None, n=4)
    assert 1 <= len(out) <= 4
    for item in out:
        assert item["title"] and item["code"]
        assert "{{" not in item["code"]  # tokens filled in fallback


def test_parse_suggestions_extracts_json_list():
    raw = 'Sure:\n[{"title": "Corr", "code": "df.corr()"}, {"title": "x"}]'
    items = ns._parse_suggestions(raw)
    assert len(items) == 1  # the item without code is dropped
    assert items[0]["code"] == "df.corr()"


def test_parse_suggestions_bad_input_returns_empty():
    assert ns._parse_suggestions("no json here") == []
