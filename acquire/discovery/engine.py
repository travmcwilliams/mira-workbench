"""
discovery/engine.py

Top-level discovery: topic + desired fields -> ranked, probed candidates ready
to show as checkboxes. Mirrors the CountyIQ discover->score->validate flow,
generalized to an arbitrary topic.
"""

from __future__ import annotations

from typing import Any, Optional

from .probe import probe_candidate
from .providers import SearchProvider, get_provider, multi_search
from .scoring import classify_scraper_type, score_result, tokenize


def build_queries(topic: str, fields: list[str]) -> list[str]:
    """A small spread of queries biased toward downloadable, official data."""
    topic = (topic or "").strip()
    field_str = " ".join(fields[:4]) if fields else ""
    queries = [
        f"{topic} dataset csv download",
        f"{topic} data download {field_str}".strip(),
        f"{topic} statistics open data",
        f"{topic} {field_str} csv OR xlsx".strip(),
    ]
    # De-dupe while preserving order; drop empties.
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def discover(
    topic: str,
    fields: Optional[list[str]] = None,
    *,
    provider: Optional[SearchProvider] = None,
    prefer: Optional[str] = None,
    max_candidates: int = 12,
    probe_top: int = 8,
    do_probe: bool = True,
) -> list[dict[str, Any]]:
    """
    Returns candidate dicts sorted best-first. Each has: url, title, snippet,
    score, scraper_type, and (when probed) data_url, data_format, n_tables,
    matched_fields, field_coverage, reachable.
    """
    fields = [f.strip() for f in (fields or []) if f and f.strip()]
    provider = provider or get_provider(prefer=prefer)

    topic_tokens = tokenize(topic)
    field_tokens = tokenize(" ".join(fields))

    results = multi_search(provider, build_queries(topic, fields), num=8)

    scored: list[dict[str, Any]] = []
    for r in results:
        s = score_result(r.url, r.title, r.snippet, topic_tokens, field_tokens)
        scored.append({
            "url": r.url,
            "title": r.title,
            "snippet": r.snippet,
            "score": s,
            "scraper_type": classify_scraper_type(s, r.url),
            "provider": provider.name,
        })

    scored.sort(key=lambda c: c["score"], reverse=True)
    candidates = scored[:max_candidates]

    if do_probe:
        for c in candidates[:probe_top]:
            try:
                probe_candidate(c, fields)
            except Exception as exc:
                c.update(reachable=False, probe_error=str(exc)[:160],
                         data_url=None, data_format=None, n_tables=0,
                         matched_fields=[], field_coverage=0.0)

    # Probed + field-matching candidates float to the top.
    candidates.sort(
        key=lambda c: (c.get("field_coverage", 0.0), c.get("reachable", False), c["score"]),
        reverse=True,
    )
    return candidates
