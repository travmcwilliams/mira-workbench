"""Declarative navigation spec + st.navigation builder (MPA v2 router).

`NAV_SECTIONS` is plain data so tests can validate the nav without running
Streamlit (paths exist, url_paths unique, exactly one default). `url_path` is
set explicitly to match the URLs the old pages/ auto-discovery generated, so
pre-revamp bookmarks keep resolving.

Deliberately unregistered: pages/2_Discover.py — Scraping Studio's Discover
tool is a superset (live probe status, charts, provider select); the file
stays on disk and can be re-added here if ever needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]


class PageSpec(NamedTuple):
    path: str            # repo-relative page file
    title: str
    icon: str
    url_path: str | None = None   # None only for the default page (serves "/")
    default: bool = False


NAV_SECTIONS: list[tuple[str, list[PageSpec]]] = [
    ("Project", [
        PageSpec("pages/home.py", "Home", "🏠", None, True),
        PageSpec("pages/0_Results.py", "Results", "✅", "Results"),
        PageSpec("pages/3_Board.py", "Board", "🗂️", "Board"),
        PageSpec("pages/5_Walkthrough.py", "Walkthrough", "🧭", "Walkthrough"),
    ]),
    ("Data", [
        PageSpec("pages/4_Explore.py", "Explore", "🔍", "Explore"),
        PageSpec("pages/7_Notebook.py", "Notebook", "📓", "Notebook"),
    ]),
    ("Acquire", [
        PageSpec("pages/1_Sources_and_Records.py", "Sources & Records", "🗄️",
                 "Sources_and_Records"),
        PageSpec("pages/6_Scraping_Studio.py", "Scraping Studio", "🛰️",
                 "Scraping_Studio"),
        PageSpec("pages/8_Investigate.py", "Investigate", "🔎", "Investigate"),
    ]),
]


def build_navigation() -> dict[str, list["st.Page"]]:
    """Materialize NAV_SECTIONS into st.Page objects for st.navigation."""
    sections: dict[str, list[st.Page]] = {}
    for section, specs in NAV_SECTIONS:
        sections[section] = [
            st.Page(
                str(_ROOT / spec.path),
                title=spec.title,
                icon=spec.icon,
                url_path=spec.url_path,
                default=spec.default,
            )
            for spec in specs
        ]
    return sections
