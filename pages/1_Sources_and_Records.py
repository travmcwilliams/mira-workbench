"""MIRA — source registry, scrape control, and records explorer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.sources_page import render_sources_page  # noqa: E402

render_sources_page()
