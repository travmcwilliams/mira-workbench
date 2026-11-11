"""
discovery/providers.py

Pluggable web-search backend for source discovery.

- SerperProvider  — google.serper.dev (needs SERPER_API_KEY). Ported from the
  CountyIQ scripts (scripts/discover_state_urls.py, county_url_finder.py) which
  found Serper to be the reliable backend.
- DuckDuckGoProvider — keyless scrape of html.duckduckgo.com/html/. No API key,
  fully self-contained, but can be rate-limited/blocked (CountyIQ's documented
  caveat for keyless search).

`get_provider()` picks Serper when a key is present, else falls back to DDG, so
the feature works out of the box with no key and gets more robust if a key is set.
"""

from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import requests

try:  # Honor SERPER_API_KEY from a local .env without requiring it.
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)


@dataclass
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""


class SearchProvider:
    name = "base"

    def available(self) -> bool:
        return True

    def search(self, query: str, num: int = 8) -> list[SearchResult]:  # pragma: no cover
        raise NotImplementedError


class SerperProvider(SearchProvider):
    name = "serper"
    ENDPOINT = "https://google.serper.dev/search"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")

    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, num: int = 8) -> list[SearchResult]:
        resp = requests.post(
            self.ENDPOINT,
            headers={"X-API-KEY": self.api_key or "", "Content-Type": "application/json"},
            json={"q": query, "num": num, "gl": "us"},
            timeout=15,
        )
        resp.raise_for_status()
        organic = resp.json().get("organic") or []
        out: list[SearchResult] = []
        for r in organic:
            link = (r.get("link") or "").strip()
            if link.startswith("http"):
                out.append(SearchResult(url=link, title=r.get("title") or "", snippet=r.get("snippet") or ""))
        return out


class DuckDuckGoProvider(SearchProvider):
    """Keyless: scrapes the no-JS DuckDuckGo HTML endpoint."""

    name = "duckduckgo"
    ENDPOINT = "https://html.duckduckgo.com/html/"

    # <a class="result__a" href="...">title</a>   and   result__snippet
    _LINK_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
    _SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)

    @staticmethod
    def _clean(text: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()

    @staticmethod
    def _unwrap(href: str) -> str:
        """DDG wraps results as /l/?uddg=<real-url>; unwrap to the real target."""
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            q = parse_qs(parsed.query)
            if "uddg" in q:
                return unquote(q["uddg"][0])
        return href

    def search(self, query: str, num: int = 8) -> list[SearchResult]:
        resp = requests.post(
            self.ENDPOINT,
            data={"q": query, "kl": "us-en"},
            headers={"User-Agent": _UA, "Accept": "text/html"},
            timeout=15,
        )
        resp.raise_for_status()
        return self.parse(resp.text, num=num)

    def parse(self, page_html: str, num: int = 8) -> list[SearchResult]:
        """Parse a DDG HTML results page. Split out for offline testing."""
        snippets = [self._clean(s) for s in self._SNIPPET_RE.findall(page_html)]
        results: list[SearchResult] = []
        for i, (href, title) in enumerate(self._LINK_RE.findall(page_html)):
            url = self._unwrap(href.strip())
            if not url.startswith("http"):
                continue
            results.append(SearchResult(
                url=url,
                title=self._clean(title),
                snippet=snippets[i] if i < len(snippets) else "",
            ))
            if len(results) >= num:
                break
        return results


class MojeekProvider(SearchProvider):
    """Keyless: scrapes Mojeek's independent HTML results (no anti-bot wall)."""

    name = "mojeek"
    ENDPOINT = "https://www.mojeek.com/search"

    _RESULT_RE = re.compile(
        r'<h2><a class="title"[^>]+href="([^"]+)"[^>]*>(.*?)</a></h2>'
        r'(?:.*?<p class="s">(.*?)</p>)?',
        re.S,
    )

    @staticmethod
    def _clean(text: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()

    def search(self, query: str, num: int = 8) -> list[SearchResult]:
        headers = {
            "User-Agent": _UA,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.mojeek.com/",
        }
        last_status = None
        for attempt in range(2):  # one polite retry on rate-limit
            resp = requests.get(self.ENDPOINT, params={"q": query},
                               headers=headers, timeout=15)
            last_status = resp.status_code
            if resp.status_code == 200:
                return self.parse(resp.text, num=num)
            if resp.status_code in (403, 429) and attempt == 0:
                time.sleep(2.0)
                continue
            break
        resp.raise_for_status()
        raise requests.HTTPError(f"Mojeek status {last_status}")

    def parse(self, page_html: str, num: int = 8) -> list[SearchResult]:
        out: list[SearchResult] = []
        for url, title, snippet in self._RESULT_RE.findall(page_html):
            url = url.strip()
            if not url.startswith("http"):
                continue
            out.append(SearchResult(url=url, title=self._clean(title), snippet=self._clean(snippet)))
            if len(out) >= num:
                break
        return out


class KeylessProvider(SearchProvider):
    """Tries several keyless engines in order, returning the first with results.

    Different engines block different environments (CountyIQ's documented caveat),
    so we fail over rather than depend on any single keyless source.
    """

    name = "keyless"

    def __init__(self, engines: list[SearchProvider] | None = None):
        self.engines = engines or [MojeekProvider(), DuckDuckGoProvider()]

    def search(self, query: str, num: int = 8) -> list[SearchResult]:
        for eng in self.engines:
            try:
                results = eng.search(query, num=num)
            except Exception:
                continue
            if results:
                self.name = f"keyless:{eng.name}"
                return results
        return []


def get_provider(prefer: str | None = None) -> SearchProvider:
    """Serper if keyed/requested, else a keyless engine (Mojeek -> DDG failover)."""
    explicit = {
        "serper": SerperProvider,
        "duckduckgo": DuckDuckGoProvider,
        "mojeek": MojeekProvider,
        "keyless": KeylessProvider,
    }
    if prefer in explicit:
        return explicit[prefer]()
    serper = SerperProvider()
    if serper.available():
        return serper
    return KeylessProvider()


def multi_search(
    provider: SearchProvider,
    queries: list[str],
    num: int = 8,
    sleep_between: float = 1.2,
) -> list[SearchResult]:
    """Run several queries, de-dupe by URL, polite sleep between calls."""
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for i, q in enumerate(queries):
        if i:
            time.sleep(sleep_between)
        try:
            for r in provider.search(q, num=num):
                if r.url not in seen:
                    seen.add(r.url)
                    merged.append(r)
        except Exception:
            # One bad query shouldn't kill discovery; keep going.
            continue
    return merged
