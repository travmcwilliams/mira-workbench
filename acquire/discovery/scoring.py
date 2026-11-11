"""
discovery/scoring.py

Rank and classify candidate URLs returned by a search provider.

Generalized from the CountyIQ url-discovery scorers
(scripts/county_url_finder.py::score_and_rank_urls and
scripts/discover_state_urls.py::_score_url): instead of being hard-wired to
"county treasurer / tax lien", scoring is driven by the user's *topic* and the
*data fields* they asked for. The same idea carries over: reward authoritative
domains and direct data files, punish junk/aggregator domains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional
from urllib.parse import urlparse

# A downloadable tabular artifact is the strongest possible signal.
# .pdf/.docx carry tables too — most scientific supplementary tables ship as one
# or the other — so they count as data here and are parsed in probe._read_file_to_df.
DATA_EXTENSIONS = (".csv", ".tsv", ".xlsx", ".xls", ".zip", ".json", ".parquet",
                   ".pdf", ".docx")

# Words that, in a URL or snippet, suggest the page exposes data (not prose).
DATA_KEYWORDS = (
    "download", "export", "dataset", "datasets", "data-set", "spreadsheet",
    "csv", "/data", "data.", "opendata", "open-data", "catalog", "records",
    "statistics", "table", "report", "api",
)

# Domains/platforms that host open data — a strong positive signal.
DATA_PORTAL_HINTS = (
    "data.gov", "data.world", "kaggle.com", "socrata", "ckan", "opendatasoft",
    "github.com", "githubusercontent.com", "fred.stlouisfed.org", "census.gov",
    "bls.gov", "worldbank.org", "ourworldindata.org",
)

# Authoritative TLDs — official sources usually publish cleaner data.
AUTHORITATIVE_TLDS = (".gov", ".edu", ".us", ".int")

# Social / SEO-spam / shopping aggregators — almost never scrapeable data.
PENALTY_DOMAINS = (
    "pinterest", "facebook", "twitter", "x.com", "instagram", "tiktok",
    "youtube", "reddit", "quora", "linkedin", "amazon.", "ebay.", "etsy.",
    "zillow", "realtor", "trulia", "redfin",
)

SCORE_HAS_DOWNLOAD = 50  # threshold to classify as a direct-download source
SCORE_HAS_RECORDS = 25   # threshold to classify as a data/record page


@dataclass(frozen=True)
class DomainPriors:
    """Per-topic overrides for the generic domain heuristics.

    The stock priors are tuned for open-government/statistical data. Other
    domains want different ones — a scientific-literature topic, for instance,
    should reward Dryad and Zenodo and actively punish ResearchGate, which
    ranks well and cannot be fetched at all. Supplying priors keeps that
    knowledge in a config pack instead of hard-coding a second domain list here.

    ``boost`` maps a URL substring to a score delta and is applied on top of
    (not instead of) the built-in rules, so a pack can nudge without restating
    everything.
    """

    portal_hints: tuple[str, ...] = DATA_PORTAL_HINTS
    authoritative_tlds: tuple[str, ...] = AUTHORITATIVE_TLDS
    penalty_domains: tuple[str, ...] = PENALTY_DOMAINS
    boost: Mapping[str, int] = field(default_factory=dict)


DEFAULT_PRIORS = DomainPriors()


def tokenize(text: str) -> set[str]:
    """Lowercase word tokens of length >= 3 (drops noise words like 'of', 'a')."""
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) >= 3}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def score_result(
    url: str,
    title: str,
    snippet: str,
    topic_tokens: set[str],
    field_tokens: set[str],
    priors: Optional[DomainPriors] = None,
) -> int:
    """
    Heuristic relevance score for one search result. Higher = more likely to be
    a scrapeable source matching the requested topic + fields.

    ``priors`` swaps the domain heuristics for a topic-specific set; omit it and
    the stock open-data priors apply exactly as before.
    """
    priors = priors or DEFAULT_PRIORS
    url_l = (url or "").lower()
    blob = f"{url_l} {(title or '').lower()} {(snippet or '').lower()}"
    blob_tokens = tokenize(blob)
    score = 0

    # Direct data file in the URL — the best outcome.
    if any(url_l.split("?")[0].rstrip("/").endswith(ext) for ext in DATA_EXTENSIONS):
        score += 50

    # Data-ish keywords anywhere.
    if any(kw in blob for kw in DATA_KEYWORDS):
        score += 15

    # Known open-data platforms.
    if any(h in url_l for h in priors.portal_hints):
        score += 30

    # Authoritative domain.
    if any(_domain(url).endswith(tld) for tld in priors.authoritative_tlds):
        score += 20

    # Topic relevance: overlap between the user's topic and the result text.
    score += 8 * len(topic_tokens & blob_tokens)

    # Field relevance: the actual columns the user wants showing up is a great sign.
    score += 6 * len(field_tokens & blob_tokens)

    # Junk/aggregator penalty.
    if any(bad in url_l for bad in priors.penalty_domains):
        score -= 40

    # Pack-supplied nudges, applied on top of the built-in rules.
    for needle, delta in priors.boost.items():
        if needle.lower() in url_l:
            score += delta

    # Wikipedia/news: readable but rarely a clean dataset.
    if "wikipedia.org" in url_l:
        score -= 15
    if "/news" in url_l or "news." in url_l:
        score -= 10

    return score


def classify_scraper_type(score: int, url: str) -> str:
    """
    Map a score (+ direct-file check) onto MIRA's existing `scraper_type` buckets,
    so the downstream ingest path already knows how to treat the source.
    """
    url_l = (url or "").lower().split("?")[0]
    if any(url_l.rstrip("/").endswith(ext) for ext in DATA_EXTENSIONS):
        return "has_download_link"
    if score >= SCORE_HAS_DOWNLOAD:
        return "has_download_link"
    if score >= SCORE_HAS_RECORDS:
        return "has_record_link"
    return "page_only"
