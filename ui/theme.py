"""Shared MIRA dark theme: global CSS injection + small layout helpers.

`.streamlit/config.toml` sets the base dark palette for native widgets; this
module layers on the card / pill / hero styling and a few widget tweaks that
config.toml can't express. Call `inject_theme()` once at the top of every page
(it's cheap and idempotent — Streamlit dedupes identical markdown per rerun).
"""

from __future__ import annotations

import copy

import plotly.io as pio
import streamlit as st

# GitHub-dark palette, shared across the app.
BG = "#0d1117"
BG_ELEV = "#161b22"
BORDER = "#21262d"
ACCENT = "#1f6feb"
ACCENT_HOVER = "#388bfd"
TEXT = "#e6edf3"
MUTED = "#9da7b3"

# Shared Plotly template so every chart matches the dark app chrome. Import
# PLOTLY_TEMPLATE in chart code and pass template=PLOTLY_TEMPLATE (importing this
# module also registers it and makes it Plotly's process-wide default).
PLOTLY_TEMPLATE = "mira_dark"


def _register_plotly_template() -> None:
    if PLOTLY_TEMPLATE in pio.templates:
        return
    tpl = copy.deepcopy(pio.templates["plotly_dark"])
    # Transparent backgrounds let the chart sit on the app's own dark surface.
    tpl.layout.paper_bgcolor = "rgba(0,0,0,0)"
    tpl.layout.plot_bgcolor = "rgba(0,0,0,0)"
    tpl.layout.font.color = TEXT
    tpl.layout.colorway = [
        ACCENT, "#3fb950", "#e3b341", "#f778ba", "#79c0ff",
        "#d29922", "#a371f7", "#ff7b72",
    ]
    pio.templates[PLOTLY_TEMPLATE] = tpl
    pio.templates.default = PLOTLY_TEMPLATE


_register_plotly_template()

_CSS = f"""
<style>
/* base bg/sidebar-bg/borders/radii now come from .streamlit/config.toml */
.stApp {{ color:{TEXT}; }}
h1,h2,h3,h4 {{ color:#f0f6fc !important; letter-spacing:.2px; }}
div.block-container {{ padding-top:2.2rem; padding-bottom:3rem; }}
/* nav section headers (st.navigation groups) */
[data-testid="stNavSectionHeader"] {{ text-transform:uppercase;
    letter-spacing:.08em; font-size:.72rem; color:{MUTED}; }}
/* elevated card */
.mira-card {{ background:{BG_ELEV}; border:1px solid {BORDER}; border-radius:12px;
             padding:14px 16px; margin-bottom:12px; }}
/* compact stat tile (Home hero row) */
.mira-stat {{ background:{BG_ELEV}; border:1px solid {BORDER}; border-radius:10px;
             padding:8px 12px; text-align:center; }}
.mira-stat .v {{ font-size:1.35rem; font-weight:700; color:#f0f6fc; }}
.mira-stat .k {{ font-size:.75rem; color:{MUTED}; text-transform:uppercase;
                letter-spacing:.06em; }}
/* branded hero banner */
.mira-hero {{ background:linear-gradient(110deg,#11203b 0%, {BG} 60%);
             border:1px solid {ACCENT}55; border-radius:14px;
             padding:16px 22px; margin-bottom:16px; }}
.mira-hero h1 {{ margin:0; font-size:1.7rem; }}
.mira-hero p {{ margin:.3rem 0 0; color:{MUTED}; font-size:.95rem; }}
/* status pills */
.pill {{ display:inline-block; background:{ACCENT}22; color:#79c0ff;
        border:1px solid {ACCENT}55; border-radius:999px; padding:2px 10px;
        margin:2px 4px 2px 0; font-size:.78rem; white-space:nowrap; }}
.pill-green {{ background:#2ea04322; color:#3fb950; border-color:#2ea04355; }}
.pill-amber {{ background:#d2992222; color:#e3b341; border-color:#d2992255; }}
.pill-gray  {{ background:#8b949e22; color:#adbac7; border-color:#8b949e55; }}
/* buttons */
.stButton>button {{ background:{ACCENT}; color:#fff; border:0; font-weight:600; }}
.stButton>button:hover {{ background:{ACCENT_HOVER}; }}
/* Streamlit colors a focused/active button's text with primaryColor — the same
   blue as our button fill — which left only the emoji visible after a click. */
.stButton>button:hover, .stButton>button:focus, .stButton>button:focus-visible,
.stButton>button:active {{ color:#fff !important; }}
.stButton>button:focus:not(:active) {{ border:0; }}
/* metrics */
div[data-testid="stMetric"] {{ background:{BG_ELEV}; border:1px solid {BORDER};
                               border-radius:10px; padding:10px 14px; }}
div[data-testid="stMetric"]:hover {{ border-color:{ACCENT}55; }}
/* tabs */
button[data-baseweb="tab"] {{ font-weight:600; }}
/* section header helper */
.mira-section {{ margin:1.4rem 0 .4rem; }}
.mira-section h3 {{ margin:0 0 .1rem; }}
.mira-section .cap {{ color:{MUTED}; font-size:.85rem; }}
.mira-section hr {{ margin:.5rem 0 0; border-color:{BORDER}; }}
/* chat input (Scraping Studio) */
div[data-testid="stChatInput"] textarea {{ background:{BG}; color:{TEXT}; }}
</style>
"""


def inject_theme() -> None:
    """Inject the shared dark CSS. Safe to call once per page render."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_hero(title: str, subtitle: str = "") -> None:
    """Render the branded gradient hero banner at the top of a page."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div class='mira-hero'><h1>{title}</h1>{sub}</div>",
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "") -> str:
    """Return an HTML pill span. `kind` in {'', 'green', 'amber', 'gray'}."""
    cls = "pill" + (f" pill-{kind}" if kind else "")
    return f"<span class='{cls}'>{text}</span>"


def section_header(title: str, caption: str = "") -> None:
    """Uniform section header: h3 + muted caption + hairline rule."""
    cap = f"<div class='cap'>{caption}</div>" if caption else ""
    st.markdown(
        f"<div class='mira-section'><h3>{title}</h3>{cap}<hr/></div>",
        unsafe_allow_html=True,
    )
