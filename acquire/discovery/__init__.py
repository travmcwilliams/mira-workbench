"""Source discovery: topic + fields -> ranked scrapeable candidates -> DataFrame.

Pluggable search (Serper when SERPER_API_KEY set, keyless DuckDuckGo fallback),
URL scoring/classification and field/table probing ported & generalized from the
CountyIQ url-discovery scripts.
"""

from .engine import build_queries, discover
from .probe import probe_candidate, scrape_candidate_to_df, validate_url
from .providers import (
    DuckDuckGoProvider,
    KeylessProvider,
    MojeekProvider,
    SearchResult,
    SerperProvider,
    get_provider,
)
from .scoring import classify_scraper_type, score_result

__all__ = [
    "discover",
    "build_queries",
    "probe_candidate",
    "scrape_candidate_to_df",
    "validate_url",
    "score_result",
    "classify_scraper_type",
    "audit_site",
    "audit_many",
    "url_confidence",
    "SiteAudit",
    "get_provider",
    "SearchResult",
    "SerperProvider",
    "DuckDuckGoProvider",
    "MojeekProvider",
    "KeylessProvider",
]
