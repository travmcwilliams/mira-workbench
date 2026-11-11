"""
utils/fetcher.py
Multi-strategy web fetcher / downloader for the DSI platform.

Generalized from the RPA bypass system in the reference CountyIQ implementation.

Purpose (per spec §6.3):
- Many sources are behind bot detection (Cloudflare 403, etc.).
- Try cheap/fast strategies first (stealth requests), then progressively heavier (Playwright).
- Log attempts via RunLogger when source_id is known.
- Graceful degradation: works even if playwright not installed.

Core functions:
- fetch_content(url, source_id="") -> str | None   (HTML/text content)
- download_file(url, dest_path: Path, source_id="") -> bool

Strategies (in order):
1. requests_stealth (realistic headers + jitter)
2. playwright_headless (Chromium with anti-webdriver scripts)
3. playwright_headless_firefox (different browser fingerprint)

Usage in agents:
    from utils.fetcher import download_file, fetch_content
    from utils.run_logger import get_run_logger

    success = download_file(url, raw_path, source_id=source_id)
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests

from .run_logger import get_run_logger

# Optional heavy deps
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore

# --- Stealth configuration ---
USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

STEALTH_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

STRATEGY_ORDER = ["requests_stealth", "playwright_headless", "playwright_firefox"]

logger = get_run_logger()  # will use RunLogger when source_id provided


def _stealth_headers() -> dict:
    return {
        **STEALTH_HEADERS,
        "User-Agent": random.choice(USER_AGENTS),
    }


def _jitter_sleep(min_s: float = 0.8, max_s: float = 2.5):
    time.sleep(random.uniform(min_s, max_s))


def _is_blocked(response_text: str, status_code: int) -> bool:
    """Heuristic for common bot blocks."""
    if status_code in (403, 429, 503):
        return True
    text_lower = (response_text or "").lower()
    blocked_indicators = [
        "cloudflare", "attention required", "ray id", "access denied",
        "bot detection", "captcha", "challenge-platform", "just a moment"
    ]
    return any(ind in text_lower for ind in blocked_indicators)


# --- Strategy implementations ---

def _try_requests_stealth(url: str, is_download: bool = False) -> Optional[str | bytes]:
    """Lightweight stealth requests. Returns text or raw bytes for downloads."""
    if not url or not url.startswith("http"):
        return None
    try:
        _jitter_sleep(0.5, 1.5)
        headers = _stealth_headers()
        resp = requests.get(
            url,
            headers=headers,
            timeout=45,
            allow_redirects=True,
            verify=False,  # common for gov sites with bad certs
        )
        if _is_blocked(resp.text, resp.status_code):
            return None
        resp.raise_for_status()
        if is_download:
            return resp.content
        return resp.text
    except Exception:
        # Silent in production; caller will try next strategy
        return None


def _try_playwright_headless(url: str) -> Optional[str]:
    """Headless Chromium with basic anti-detection."""
    if not PLAYWRIGHT_AVAILABLE or not url or not url.startswith("http"):
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                user_agent=random.choice(USER_AGENTS),
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            page = context.new_page()
            # Stealth script
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                window.chrome = { runtime: {} };
            """)
            page.goto(url, wait_until="networkidle", timeout=45000)
            content = page.content()
            browser.close()
            if _is_blocked(content, 200):
                return None
            return content
    except Exception:
        return None


def _try_playwright_firefox(url: str) -> Optional[str]:
    """Firefox engine for different fingerprint."""
    if not PLAYWRIGHT_AVAILABLE or not url or not url.startswith("http"):
        return None
    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=45000)
            content = page.content()
            browser.close()
            if _is_blocked(content, 200):
                return None
            return content
    except Exception:
        return None


STRATEGY_FUNCS = {
    "requests_stealth": _try_requests_stealth,
    "playwright_headless": _try_playwright_headless,
    "playwright_firefox": _try_playwright_firefox,
}


def fetch_content(url: str, source_id: str = "", max_strategies: int = 3) -> Optional[str]:
    """
    Fetch page content using escalating strategies.
    Returns HTML/text on success, None on total failure.
    Logs via RunLogger if source_id provided.
    """
    if source_id:
        logger.log_event(source_id, "fetch_attempt", {"url": url[:120]})

    for strat in STRATEGY_ORDER[:max_strategies]:
        func = STRATEGY_FUNCS.get(strat)
        if not func:
            continue
        content = func(url)
        if content:
            if source_id:
                logger.log_event(source_id, "fetch_ok", {"strategy": strat, "length": len(content)})
            return content
        # Try next strategy on failure / block

    if source_id:
        logger.log_event(source_id, "fetch_fail", {"url": url[:120], "reason": "all_strategies_failed"})
    return None


def download_file(url: str, dest_path: Path, source_id: str = "") -> bool:
    """
    Download a file (CSV, XLSX, ZIP, PDF, etc.) using stealth strategies.
    Saves to dest_path on success.
    Returns True on success.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if source_id:
        logger.log_event(source_id, "download_attempt", {"url": url[:120], "dest": str(dest_path)})

    # 1. Try requests first (fast path for direct files)
    content = _try_requests_stealth(url, is_download=True)
    if isinstance(content, (bytes, bytearray)):
        dest_path.write_bytes(content)
        if source_id:
            logger.log_event(source_id, "download_ok", {"strategy": "requests_stealth", "bytes": len(content)})
        return True

    # 2. Fall back to Playwright for pages that require JS or have protection
    if PLAYWRIGHT_AVAILABLE:
        for strat in ["playwright_headless", "playwright_firefox"]:
            # Playwright strategies above return HTML, not binary.
            # For real binary downloads behind protection, a better approach is to
            # use page.request or handle via context. Here we keep it simple:
            # re-use the page content fetch and then look for direct file links.
            # For true binary bypass, many sources still expose a direct link once the page loads.
            html = STRATEGY_FUNCS[strat](url)
            if html and "content-type" not in str(html).lower()[:200]:  # rough heuristic
                # If we got HTML, try to find a direct download link on the page
                if BeautifulSoup:
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        href = a.get("href", "")
                        if re.search(r"\.(csv|xlsx?|zip|pdf|json)(\?|$)", href, re.I):
                            full = urljoin(url, href)
                            bin_content = _try_requests_stealth(full, is_download=True)
                            if isinstance(bin_content, (bytes, bytearray)):
                                dest_path.write_bytes(bin_content)
                                if source_id:
                                    logger.log_event(source_id, "download_ok", {"strategy": f"{strat}_link", "bytes": len(bin_content)})
                                return True
    # Failed
    if source_id:
        logger.log_event(source_id, "download_fail", {"url": url[:120]})
    return False


def fetch_with_fallback(url: str, source_id: str = "") -> Optional[str]:
    """Convenience alias for the most common use case."""
    return fetch_content(url, source_id=source_id)
