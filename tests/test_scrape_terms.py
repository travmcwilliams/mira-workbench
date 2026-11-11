"""Headless UI tests for the Scrape tab's editable/removable search terms.

Drives the real ``_render_patch_scrape`` via Streamlit's AppTest harness (no
browser, no network) and asserts the underlying ``patch_terms`` state after
add / duplicate / inline-edit / remove interactions. The remove case guards the
index-keyed-widget staleness bug (a removed term must not linger in the row that
shifts up into its place).
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit.testing.v1", reason="streamlit AppTest not available")

from streamlit.testing.v1 import AppTest  # noqa: E402


def _scrape_app() -> None:
    """Minimal Streamlit app rendering only the patch-scrape term editor."""
    from ui.sources_page import _render_patch_scrape

    _render_patch_scrape()


def _run():
    at = AppTest.from_function(_scrape_app, default_timeout=30).run()
    if "patch_terms" not in at.session_state:
        pytest.skip("no scrape patches configured; term editor did not render")
    return at


def test_initial_terms_loaded():
    at = _run()
    assert len(at.session_state["patch_terms"]) > 0


def test_add_term_appends():
    at = _run()
    start = list(at.session_state["patch_terms"])
    at.text_input(key="mira_patch_term").set_value("2025 ai infrastructure spend").run()
    at.button(key="mira_add_term").click().run()
    terms = list(at.session_state["patch_terms"])
    assert "2025 ai infrastructure spend" in terms
    assert len(terms) == len(start) + 1


def test_duplicate_add_ignored():
    at = _run()
    at.text_input(key="mira_patch_term").set_value("dup-term").run()
    at.button(key="mira_add_term").click().run()
    n = len(at.session_state["patch_terms"])
    at.text_input(key="mira_patch_term").set_value("dup-term").run()
    at.button(key="mira_add_term").click().run()
    assert len(at.session_state["patch_terms"]) == n


def test_inline_edit_updates_term():
    at = _run()
    at.text_input(key="mira_term_edit_0").set_value("edited-term-zero").run()
    assert at.session_state["patch_terms"][0] == "edited-term-zero"


def test_remove_drops_term_without_stale_value():
    at = _run()
    # Make term 0 uniquely identifiable, then remove it.
    at.text_input(key="mira_term_edit_0").set_value("to-be-removed").run()
    before = list(at.session_state["patch_terms"])
    at.button(key="mira_term_del_0").click().run()
    after = list(at.session_state["patch_terms"])
    assert len(after) == len(before) - 1
    assert "to-be-removed" not in after
    # The row that shifts up must show the *next* term, not the removed one.
    assert after == before[1:]
