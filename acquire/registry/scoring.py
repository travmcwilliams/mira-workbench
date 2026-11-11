"""
registry/scoring.py
The trust/priority layer: turn raw source metrics into a composite 0..100
"Mira Source Score", and decide inclusion via configurable rules.

Core principle (from the reference design): high traffic means *influential*, NOT
*truthful*. Authority, readership, virality, citation, official status and legal
accessibility are scored SEPARATELY and only then combined. Official government,
official incident, and indexed-academic sources bypass popularity thresholds entirely.

Pure functions — no DB, no network — so they are trivially testable and reusable by
the discovery-candidate scorer and the crawl scheduler alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

# Component keys, in the order the weights YAML enumerates them.
COMPONENT_KEYS = (
    "authority",
    "readership",
    "engagement",
    "citation",
    "freshness",
    "topical_relevance",
    "accessibility",
)

# Accessibility: legal risk -> 0..1 (easier & permitted scores higher).
_ACCESSIBILITY = {
    "low": 1.0,
    "medium": 0.6,
    "high": 0.3,
    "blocked": 0.0,
    "unknown": 0.5,
}


def validate_weights(weights: dict[str, float]) -> None:
    """Weights must cover every component and sum to ~1.0."""
    missing = [k for k in COMPONENT_KEYS if k not in weights]
    if missing:
        raise ValueError(f"scoring_weights missing components: {missing}")
    total = sum(float(weights[k]) for k in COMPONENT_KEYS)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"scoring_weights must sum to 1.0, got {total:.6f}")


def _log_norm(value: Optional[float], cap: float) -> float:
    """log10 compression of a heavy-tailed count into 0..1 against `cap`."""
    if not value or value <= 0 or cap <= 0:
        return 0.0
    v = min(float(value), float(cap))
    return math.log10(v + 1.0) / math.log10(float(cap) + 1.0)


def _linear_norm(value: Optional[float], cap: float) -> float:
    if not value or value <= 0 or cap <= 0:
        return 0.0
    return min(float(value) / float(cap), 1.0)


def freshness_from_age(age_days: Optional[float], halflife_days: float) -> float:
    """Exponential-decay freshness: 1.0 today, halves every `halflife_days`."""
    if age_days is None or age_days < 0 or halflife_days <= 0:
        return 0.0
    return 0.5 ** (float(age_days) / float(halflife_days))


def components_from_metrics(
    metrics: dict[str, Any],
    caps: dict[str, Any],
    *,
    legal_risk_level: str = "unknown",
    topical_relevance: float = 0.5,
    age_days: Optional[float] = None,
) -> dict[str, float]:
    """
    Derive the seven 0..1 component scores from a raw metric row + config caps.

    `metrics` may carry any of: source_authority_score (0..100 or 0..1), monthly_visits,
    follower_count/subscriber_count, avg_views, engagement_rate, citation_count,
    inbound_links. Missing metrics contribute 0 to their component.
    """
    caps = caps or {}

    # authority: prefer an explicit authority score (accept 0..100 or 0..1), else
    # fall back to citation-driven proxy.
    auth_raw = metrics.get("source_authority_score")
    if auth_raw is None:
        authority = _log_norm(metrics.get("citation_count"), float(caps.get("citation_count", 1e6)))
    else:
        authority = float(auth_raw) / 100.0 if float(auth_raw) > 1.0 else float(auth_raw)
        authority = max(0.0, min(authority, 1.0))

    reach = metrics.get("monthly_visits") or metrics.get("follower_count") or metrics.get("subscriber_count")
    reach_cap = float(caps.get("monthly_visits", 1e9))
    readership = _log_norm(reach, reach_cap)

    engagement = _linear_norm(metrics.get("engagement_rate"), float(caps.get("engagement_rate", 0.2)))

    citation = _log_norm(metrics.get("citation_count") or metrics.get("inbound_links"),
                         float(caps.get("citation_count", 1e6)))

    freshness = freshness_from_age(age_days, float(caps.get("freshness_halflife_days", 30)))

    accessibility = _ACCESSIBILITY.get((legal_risk_level or "unknown").lower(), 0.5)

    return {
        "authority": authority,
        "readership": readership,
        "engagement": engagement,
        "citation": citation,
        "freshness": freshness,
        "topical_relevance": max(0.0, min(float(topical_relevance), 1.0)),
        "accessibility": accessibility,
    }


def mira_source_score(components: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted sum of 0..1 components -> 0..100. Weights validated by caller/load."""
    score = sum(float(weights[k]) * max(0.0, min(float(components.get(k, 0.0)), 1.0))
                for k in COMPONENT_KEYS)
    return round(score * 100.0, 2)


# --------------------------------------------------------------------------- #
# Inclusion evaluation
# --------------------------------------------------------------------------- #

@dataclass
class InclusionResult:
    included: bool
    rule_name: Optional[str]
    reason: str


_THRESHOLDS = {
    "min_monthly_visits": "monthly_visits",
    "min_followers": "follower_count",
    "min_subscribers": "subscriber_count",
    "min_avg_views": "avg_views",
    "min_avg_engagement": "engagement_rate",
    "min_mentions": "avg_mentions",
    "min_citation_count": "citation_count",
    "min_authority_score": "source_authority_score",
}


def _rule_filters_match(rule: dict[str, Any], source: dict[str, Any]) -> bool:
    """Null filter = wildcard. A set filter must equal the source's value."""
    for field in ("source_type", "platform_type", "country_code"):
        want = rule.get(field)
        if want is not None and source.get(field) != want:
            return False
    want_cat = rule.get("category_name")
    if want_cat is not None and want_cat not in (source.get("categories") or []):
        return False
    return True


def evaluate_inclusion(
    source: dict[str, Any],
    metrics: dict[str, Any],
    rules: list[dict[str, Any]],
) -> InclusionResult:
    """
    Decide whether a source qualifies. Order:
      1. Source-level override flags (government / incident / academic) -> always in.
      2. Active rules by ascending `priority`: filters must match AND either a
         threshold is met or the rule's include_if_* flag matches the source.
    """
    # 1. Hard overrides — popularity is irrelevant for these.
    if source.get("official_government"):
        return InclusionResult(True, None, "official_government source: included regardless of reach")
    if source.get("official_incident_feed"):
        return InclusionResult(True, None, "official_incident_feed: included regardless of reach")
    if source.get("academic_repository"):
        return InclusionResult(True, None, "indexed academic repository: included regardless of reach")

    active = [r for r in rules if r.get("active", True)]
    active.sort(key=lambda r: r.get("priority", 100))

    for rule in active:
        if not _rule_filters_match(rule, source):
            continue

        # include_if_* flags on the rule matched against source flags.
        for flag, src_key in (
            ("include_if_government", "official_government"),
            ("include_if_academic", "academic_repository"),
            ("include_if_incident_feed", "official_incident_feed"),
            ("include_if_local_official", "local_official"),
        ):
            if rule.get(flag) and source.get(src_key):
                return InclusionResult(True, rule.get("rule_name"), f"{rule.get('rule_name')}: {flag}")

        # numeric thresholds — any single satisfied threshold admits the source.
        for rkey, mkey in _THRESHOLDS.items():
            want = rule.get(rkey)
            if want is None:
                continue
            have = metrics.get(mkey)
            if have is not None and float(have) >= float(want):
                return InclusionResult(True, rule.get("rule_name"),
                                       f"{rule.get('rule_name')}: {mkey}={have} >= {want}")

    return InclusionResult(False, None, "no active inclusion rule matched")
