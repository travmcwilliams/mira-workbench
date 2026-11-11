"""ui/nav.py — the nav spec is plain data; validate it without running Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.nav import NAV_SECTIONS  # noqa: E402

ALL_SPECS = [spec for _, specs in NAV_SECTIONS for spec in specs]


def test_every_registered_page_exists_on_disk():
    for spec in ALL_SPECS:
        assert (_ROOT / spec.path).is_file(), spec.path


def test_paths_and_url_paths_unique():
    paths = [s.path for s in ALL_SPECS]
    assert len(paths) == len(set(paths))
    urls = [s.url_path for s in ALL_SPECS if s.url_path]
    assert len(urls) == len(set(urls))


def test_exactly_one_default_and_it_is_home():
    defaults = [s for s in ALL_SPECS if s.default]
    assert len(defaults) == 1
    assert defaults[0].path == "pages/home.py"
    assert defaults[0].url_path is None  # default serves "/"


def test_discover_and_campaigns_dropped_results_present():
    paths = {s.path for s in ALL_SPECS}
    assert "pages/2_Discover.py" not in paths
    assert "pages/0_Results.py" in paths
    assert "pages/9_Campaigns.py" not in paths

