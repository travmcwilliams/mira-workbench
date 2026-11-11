"""MIRA — Scraping Studio: source discovery, URL compiling, and an AI copilot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.scraping_studio import render_scraping_studio  # noqa: E402

render_scraping_studio()
