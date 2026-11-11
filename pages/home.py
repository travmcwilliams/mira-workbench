"""Home — page target for the st.navigation router (body in ui/home_page.py).

A file target (not a callable) so string-based st.switch_page("pages/home.py")
works from ui/ modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.home_page import render_home  # noqa: E402

render_home()
