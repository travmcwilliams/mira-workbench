"""Offline tests for the source-discovery feature (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from core.acquire_bridge import ensure_acquire_env

ensure_acquire_env()  # put acquire/ on sys.path so `discovery` is importable

from discovery.engine import build_queries  # noqa: E402
from discovery.probe import (  # noqa: E402
    match_fields,
    probe_candidate,
    scrape_candidate_to_df,
)
from discovery.providers import (  # noqa: E402
    DuckDuckGoProvider,
    KeylessProvider,
    MojeekProvider,
    SearchBlocked,
    SearchProvider,
    SearchResult,
    SerperProvider,
    get_provider,
    multi_search,
)
from discovery.scoring import (  # noqa: E402
    classify_scraper_type,
    score_result,
    tokenize,
)

FIELDS = ["county", "state", "population", "year"]
TOPIC_TOKENS = tokenize("us county population")
FIELD_TOKENS = tokenize(" ".join(FIELDS))


# --- scoring -------------------------------------------------------------------

def test_score_rewards_data_file_and_gov():
    s_csv = score_result("https://data.gov/county/population.csv", "Population",
                         "county population by year", TOPIC_TOKENS, FIELD_TOKENS)
    s_blog = score_result("https://medium.com/some-post", "A post",
                         "opinions", TOPIC_TOKENS, FIELD_TOKENS)
    assert s_csv > s_blog
    assert s_csv >= 50


def test_score_penalizes_junk_domains():
    s = score_result("https://pinterest.com/pin/population", "pop", "", TOPIC_TOKENS, FIELD_TOKENS)
    assert s < 0


def test_classify_scraper_type():
    assert classify_scraper_type(10, "https://x.gov/data.csv") == "has_download_link"
    assert classify_scraper_type(60, "https://x.gov/page") == "has_download_link"
    assert classify_scraper_type(30, "https://x.gov/page") == "has_record_link"
    assert classify_scraper_type(5, "https://x.gov/page") == "page_only"


# --- providers -----------------------------------------------------------------

DDG_HTML = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdata.gov%2Fpop.csv&rut=z">Population CSV</a>
  <a class="result__snippet">County population dataset by year</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Ftable">Example table</a>
  <a class="result__snippet">A table of values</a>
</div>
"""


def test_ddg_parse_unwraps_redirect_links():
    results = DuckDuckGoProvider().parse(DDG_HTML, num=10)
    assert len(results) == 2
    assert results[0].url == "https://data.gov/pop.csv"
    assert results[0].title == "Population CSV"
    assert "population" in results[0].snippet.lower()
    assert results[1].url == "https://example.org/table"


def test_get_provider_prefers_serper_when_keyed(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    assert isinstance(get_provider(), SerperProvider)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    assert isinstance(get_provider(), KeylessProvider)  # keyless failover when no key
    # Explicit overrides always win.
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    assert isinstance(get_provider(prefer="duckduckgo"), DuckDuckGoProvider)
    assert isinstance(get_provider(prefer="mojeek"), MojeekProvider)


MOJEEK_HTML = """
<ul class="results-standard">
<li class="r1"><a title="u" href="https://data.gov/pop.csv" class="ob"></a>
<h2><a class="title" title="t" href="https://data.gov/pop.csv">Population CSV dataset</a></h2>
<p class="s">county state <strong>population</strong> by year</p></li>
</ul>
"""


def test_mojeek_parse():
    results = MojeekProvider().parse(MOJEEK_HTML, num=5)
    assert len(results) == 1
    assert results[0].url == "https://data.gov/pop.csv"
    assert results[0].title == "Population CSV dataset"
    assert "population" in results[0].snippet.lower()


# --- captcha / bot-check detection ---------------------------------------------

def test_captcha_pages_are_detected():
    assert MojeekProvider.is_blocked("<html><head><title>Captcha</title></head></html>")
    assert not MojeekProvider.is_blocked(MOJEEK_HTML)
    assert DuckDuckGoProvider.is_blocked(202, "<div></div>")
    assert DuckDuckGoProvider.is_blocked(200, '<div class="anomaly-modal__box">')
    assert not DuckDuckGoProvider.is_blocked(200, DDG_HTML)


class _Blocked(SearchProvider):
    name = "blocked"

    def search(self, query, num=8):
        raise SearchBlocked(self.name)


class _Empty(SearchProvider):
    name = "empty"

    def search(self, query, num=8):
        return []


class _Hit(SearchProvider):
    name = "hit"

    def search(self, query, num=8):
        return [SearchResult(url="https://example.org/data.csv")]


def test_keyless_raises_only_when_every_engine_is_blocked():
    with pytest.raises(SearchBlocked):
        KeylessProvider([_Blocked(), _Blocked()]).search("q")
    assert KeylessProvider([_Blocked(), _Hit()]).search("q")[0].url.endswith(".csv")
    assert KeylessProvider([_Blocked(), _Empty()]).search("q") == []  # genuinely empty


def test_multi_search_reports_block_instead_of_empty():
    with pytest.raises(SearchBlocked, match="captcha"):
        multi_search(_Blocked(), ["a", "b"], sleep_between=0)
    assert multi_search(_Empty(), ["a", "b"], sleep_between=0) == []


# --- probe ---------------------------------------------------------------------

PAGE_WITH_CSV = """
<html><body>
  <h1>County population</h1>
  <a href="/files/pop.csv">Download population CSV</a>
  <table>
    <tr><th>county</th><th>state</th><th>population</th></tr>
    <tr><td>Alpha</td><td>TX</td><td>100</td></tr>
  </table>
</body></html>
"""

PAGE_TABLE_ONLY = """
<html><body>
  <table>
    <tr><th>county</th><th>state</th><th>population</th></tr>
    <tr><td>Alpha</td><td>TX</td><td>100</td></tr>
    <tr><td>Beta</td><td>NY</td><td>200</td></tr>
  </table>
</body></html>
"""


def test_match_fields_token_level():
    matched = match_fields(FIELDS, "the county and state columns here")
    assert set(matched) == {"county", "state"}


def test_probe_finds_csv_link_and_fields():
    cand = {"url": "https://data.gov/county", "title": "County population", "snippet": ""}
    probe_candidate(cand, FIELDS, page_html=PAGE_WITH_CSV)
    assert cand["reachable"] is True
    assert cand["data_url"].endswith("/files/pop.csv")
    assert cand["data_format"] == "csv"
    assert cand["n_tables"] == 1
    assert set(cand["matched_fields"]) == {"county", "state", "population"}
    assert cand["field_coverage"] == 0.75


def test_probe_direct_data_file_needs_no_fetch():
    cand = {"url": "https://data.gov/pop.csv", "title": "pop", "snippet": "county state population"}
    probe_candidate(cand, FIELDS)  # no page_html, no network — direct ext path
    assert cand["data_format"] == "csv"
    assert cand["data_url"] == "https://data.gov/pop.csv"


def test_probe_table_only_uses_html_table():
    cand = {"url": "https://example.org/t", "title": "t", "snippet": ""}
    probe_candidate(cand, FIELDS, page_html=PAGE_TABLE_ONLY)
    assert cand["data_format"] == "html_table"
    assert cand["data_url"] == "https://example.org/t"
    assert cand["n_tables"] == 1


def test_scrape_html_table_to_df():
    cand = {"url": "https://example.org/t", "data_url": "https://example.org/t",
            "data_format": "html_table"}
    df = scrape_candidate_to_df(cand, FIELDS, page_html=PAGE_TABLE_ONLY)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] == 2
    assert "population" in [str(c).lower() for c in df.columns]


# --- engine --------------------------------------------------------------------

def test_build_queries_uses_topic_and_fields():
    qs = build_queries("county population", FIELDS)
    assert qs
    assert all(q.strip() for q in qs)
    assert len(qs) == len(set(qs))  # de-duped
    assert any("county population" in q for q in qs)


def test_read_excel_best_sheet_skips_cover_sheet(tmp_path):
    """Multi-sheet workbook: pick the data sheet, not the leading notes sheet."""
    from discovery.probe import _read_file_to_df

    p = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"This spreadsheet contains notes": ["by some authors", "see letter"]}
                     ).to_excel(w, sheet_name="Methodology", index=False)
        pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=60, freq="D").strftime("%Y-%m-%d"),
            "value": range(60),
        }).to_excel(w, sheet_name="Data", index=False)

    df = _read_file_to_df(p)
    assert df is not None
    assert set(df.columns) == {"date", "value"}
    assert len(df) == 60  # the daily series, not the 2-row cover sheet


def test_read_excel_promotes_buried_header(tmp_path):
    """Single sheet with preamble rows above the real header."""
    import openpyxl

    from discovery.probe import _read_excel_best_sheet

    p = tmp_path / "preamble.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Quarterly report — internal"])
    ws.append([])
    ws.append(["date", "value"])
    for i in range(20):
        ws.append([f"2021-02-{i + 1:02d}", i * 2])
    wb.save(p)

    df = _read_excel_best_sheet(p)
    assert df is not None
    cols = [str(c) for c in df.columns]
    assert "date" in cols and "value" in cols
    assert len(df) == 20


def test_probe_no_render_when_html_supplied():
    """Caller-supplied HTML with no data must not trigger a network render."""
    cand = {"url": "https://example.org/empty", "title": "x", "snippet": ""}
    probe_candidate(cand, FIELDS, page_html="<html><body><p>nothing here</p></body></html>")
    assert cand["data_url"] is None
    assert cand["n_tables"] == 0
    assert "rendered" not in cand  # render path skipped for supplied HTML


def test_sanitize_for_parquet_handles_mixed_columns(tmp_path):
    from core.acquire_bridge import _sanitize_for_parquet

    # Mixed int/str object column (e.g. messy spreadsheet) breaks pyarrow.
    df = pd.DataFrame({"state": ["TX", "NY"], "Unnamed: 3": [1, "note"], "val": [3.5, None]})
    with pytest.raises(Exception):
        df.to_parquet(tmp_path / "raw.parquet")

    clean = _sanitize_for_parquet(df)
    clean.to_parquet(tmp_path / "clean.parquet")  # must not raise
    assert (tmp_path / "clean.parquet").exists()


def test_discover_offline_with_stub_provider():
    from discovery.engine import discover

    class StubProvider:
        name = "stub"

        def search(self, query, num=8):
            from discovery.providers import SearchResult
            return [
                SearchResult("https://data.gov/pop.csv", "Population CSV", "county state population year"),
                SearchResult("https://pinterest.com/x", "junk", "junk"),
            ]

    cands = discover("county population", FIELDS, provider=StubProvider(), do_probe=False)
    assert cands[0]["url"] == "https://data.gov/pop.csv"  # ranks above junk
    assert cands[0]["scraper_type"] == "has_download_link"
