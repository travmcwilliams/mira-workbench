"""MIRA — Start from a question: intent-driven, schema-from-the-web project creation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.investigate_page import render_investigate  # noqa: E402

render_investigate()
