"""
core/webschema.py — scale web discovery and consolidate scraped tables into one
proposed schema.

Two jobs the existing acquire pipeline doesn't cover:

1. ``discover_many`` — accumulate up to ~N ranked, reachability-probed candidates
   by running a *spread* of search queries (the stock ``discover`` runs a fixed
   four). It reuses the acquire discovery engine's provider/scoring/probe pieces
   and is honest about how many it actually reached (search providers rate-limit).

2. ``consolidate`` — turn a pile of heterogeneous scraped DataFrames into ONE
   frame + a proposed schema. Column names are normalized and near-duplicates are
   fuzzy-merged (rapidfuzz) so ``Revenue``, ``revenue`` and ``revenues`` collapse
   to a single column; rows are unioned and de-duplicated; then the project's own
   ``detect_schema`` / ``suggest_target`` propose the schema and target. A
   per-column provenance report says which sources contributed each field.

Nothing here creates a Project — that's the orchestrator's job (core/investigate).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from core.acquire_bridge import ensure_acquire_env

# --------------------------------------------------------------------------- #
# discovery at scale
# --------------------------------------------------------------------------- #


def discover_many(
    topic: str,
    queries: list[str],
    fields: list[str] | None = None,
    *,
    prefer: str | None = None,
    target_n: int = 100,
    probe_top: int = 40,
    do_probe: bool = True,
    per_query_num: int = 10,
) -> dict[str, Any]:
    """Run a spread of queries → ranked, probed candidates (best-first).

    Returns ``{provider, topic, candidates, n_raw, n_queries}``. ``n_raw`` is how
    many unique URLs search actually returned before capping to ``target_n`` — the
    honest ceiling, since keyless providers rate-limit well below 100.
    """
    ensure_acquire_env()
    from discovery.probe import probe_candidate
    from discovery.providers import get_provider, multi_search
    from discovery.scoring import classify_scraper_type, score_result, tokenize

    fields = [f.strip() for f in (fields or []) if f and f.strip()]
    provider = get_provider(prefer=prefer)

    topic_tokens = tokenize(topic)
    field_tokens = tokenize(" ".join(fields))

    results = multi_search(provider, queries, num=per_query_num)

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
    candidates = scored[:target_n]

    if do_probe:
        for c in candidates[:probe_top]:
            try:
                probe_candidate(c, fields)
            except Exception as exc:  # noqa: BLE001 - one bad probe shouldn't stop discovery
                c.update(reachable=False, probe_error=str(exc)[:160],
                         data_url=None, data_format=None, n_tables=0,
                         matched_fields=[], field_coverage=0.0)

    candidates.sort(
        key=lambda c: (c.get("field_coverage", 0.0), c.get("reachable", False), c["score"]),
        reverse=True,
    )
    return {
        "provider": provider.name,
        "topic": topic,
        "candidates": candidates,
        "n_raw": len(results),
        "n_queries": len(queries),
    }


# --------------------------------------------------------------------------- #
# schema consolidation
# --------------------------------------------------------------------------- #


@dataclass
class ColumnInfo:
    name: str                       # canonical (normalized) column name
    dtype: str                      # from detect_schema
    sources: int                    # how many source frames contributed it
    aliases: list[str] = field(default_factory=list)  # raw names that mapped in


@dataclass
class ConsolidationReport:
    n_rows: int
    n_sources: int                  # source frames actually combined
    columns: list[ColumnInfo]
    schema: dict[str, str]
    task_type: str
    target_column: str | None
    target_reason: str | None
    dropped_empty: int              # source frames that contributed nothing
    strategy: str = "coherent"
    n_source_frames: int = 0        # total non-empty frames available
    dropped_incoherent: int = 0     # frames excluded because they didn't fit
    clusters: list[dict[str, Any]] = field(default_factory=list)  # for UI selection


def normalize_col(name: Any) -> str:
    """Lower, trim, and snake-case a column name for cross-source matching."""
    s = str(name).strip().lower()
    out: list[str] = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    return "".join(out).strip("_") or "col"


def _norm_colset(f: pd.DataFrame) -> set[str]:
    return {normalize_col(c) for c in f.columns}


def cluster_frames(frames: list[pd.DataFrame], *, min_shared: int = 2) -> list[list[int]]:
    """Group frame indices that share a schema into coherent clusters.

    Two frames are "coherent" when their normalized column sets overlap by
    ``>= min_shared`` columns, OR they share at least one column and their
    Jaccard overlap is >= 0.5 (so two narrow frames sharing their only column
    still merge). Connected components over that relation; groups returned
    largest-first by total row count. Frames sharing nothing stay singletons.
    """
    colsets = [_norm_colset(f) for f in frames]
    n = len(frames)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            inter = len(colsets[i] & colsets[j])
            uni = len(colsets[i] | colsets[j])
            jac = inter / uni if uni else 0.0
            if inter >= min_shared or (inter >= 1 and jac >= 0.5):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = list(groups.values())
    out.sort(key=lambda g: sum(len(frames[i]) for i in g), reverse=True)
    return out


def _cluster_columns(all_cols: list[str], threshold: int) -> dict[str, str]:
    """Map each normalized column name to a canonical representative.

    Exact normalized matches always merge; near-duplicates merge only when the
    Levenshtein ratio is >= ``threshold`` (conservative: ``ratio`` not
    ``token_set_ratio``, so ``revenue`` never swallows ``total_revenue``).
    """
    reps: list[str] = []
    mapping: dict[str, str] = {}
    counts: Counter[str] = Counter(all_cols)
    # Process most-common names first so the representative is the popular spelling.
    for col in sorted(set(all_cols), key=lambda c: (-counts[c], c)):
        match = None
        for rep in reps:
            if col == rep or fuzz.ratio(col, rep) >= threshold:
                match = rep
                break
        if match is None:
            reps.append(col)
            mapping[col] = col
        else:
            mapping[col] = match
    return mapping


def consolidate(
    frames: list[pd.DataFrame],
    *,
    task_type: str = "regression",
    fuzzy_threshold: int = 90,
    strategy: str = "coherent",
    min_shared: int = 2,
    keep_indices: list[int] | None = None,
    keep_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, ConsolidationReport]:
    """Combine scraped frames into one frame + a proposed schema.

    ``strategy``:
      - ``"coherent"`` (default): keep only the largest cluster of frames that
        actually share a schema — avoids the sparse mess of unioning unrelated
        tables. Alternative clusters are reported for the UI to offer.
      - ``"union_all"``: union every frame regardless of schema overlap.
    ``keep_indices`` overrides the strategy with an explicit frame selection
    (indices into the non-empty frame list). ``keep_columns`` prunes the final
    schema to those canonical column names.
    """
    from core.data_loader import detect_schema, suggest_target

    non_empty = [f for f in frames if f is not None and not f.empty]
    dropped = len(frames) - len(non_empty)
    if not non_empty:
        empty = pd.DataFrame()
        return empty, ConsolidationReport(
            n_rows=0, n_sources=0, columns=[], schema={}, task_type=task_type,
            target_column=None, target_reason=None, dropped_empty=dropped,
            strategy=strategy, n_source_frames=0)

    # 0) decide which frames to combine.
    groups = cluster_frames(non_empty, min_shared=min_shared)
    clusters_meta = [
        {"indices": g, "n_frames": len(g),
         "rows": int(sum(len(non_empty[i]) for i in g)),
         "columns": sorted({c for i in g for c in _norm_colset(non_empty[i])})}
        for g in groups
    ]
    if keep_indices is not None:
        sel_idx = [i for i in keep_indices if 0 <= i < len(non_empty)]
    elif strategy == "coherent" and groups:
        sel_idx = list(groups[0])
    else:
        sel_idx = list(range(len(non_empty)))
    if not sel_idx:
        sel_idx = list(range(len(non_empty)))
    selected = [non_empty[i] for i in sel_idx]
    dropped_incoherent = len(non_empty) - len(selected)

    # 1) normalize each frame's columns; track which raw name came from where.
    normed: list[pd.DataFrame] = []
    raw_by_norm: dict[str, list[str]] = {}
    for f in selected:
        f = f.dropna(axis=1, how="all").copy()
        rename: dict[Any, str] = {}
        for c in f.columns:
            nc = normalize_col(c)
            rename[c] = nc
            raw_by_norm.setdefault(nc, [])
            if str(c) not in raw_by_norm[nc]:
                raw_by_norm[nc].append(str(c))
        f = f.rename(columns=rename)
        # collapse duplicate normalized columns within one frame (keep first)
        f = f.loc[:, ~f.columns.duplicated()]
        normed.append(f)

    # 2) fuzzy-cluster normalized names to a canonical representative.
    all_cols = [c for f in normed for c in f.columns]
    cluster = _cluster_columns(all_cols, fuzzy_threshold)

    remapped: list[pd.DataFrame] = []
    canon_sources: Counter[str] = Counter()
    canon_aliases: dict[str, list[str]] = {}
    for f in normed:
        f = f.rename(columns=cluster)
        f = f.loc[:, ~f.columns.duplicated()]
        remapped.append(f)
        for nc in set(f.columns):
            canon = cluster[nc] if nc in cluster else nc
            canon_sources[canon] += 1
    for nc, canon in cluster.items():
        canon_aliases.setdefault(canon, [])
        for raw in raw_by_norm.get(nc, []):
            if raw not in canon_aliases[canon]:
                canon_aliases[canon].append(raw)

    # 3) union rows, de-duplicate.
    combined = pd.concat(remapped, ignore_index=True, sort=False)
    combined = combined.drop_duplicates().reset_index(drop=True)

    # 3b) optional manual column prune (canonical names that survive).
    if keep_columns is not None:
        keep = [c for c in combined.columns if c in set(keep_columns)]
        if keep:
            combined = combined[keep]

    # 4) propose schema + target using the project's own detectors.
    schema = detect_schema(combined)
    suggestion = suggest_target(combined, schema, task_type=task_type)
    target = suggestion.get("target_column")

    columns = [
        ColumnInfo(name=col, dtype=schema.get(col, "unknown"),
                   sources=int(canon_sources.get(col, 0)),
                   aliases=canon_aliases.get(col, [col]))
        for col in combined.columns
    ]
    columns.sort(key=lambda ci: (-ci.sources, ci.name))

    report = ConsolidationReport(
        n_rows=int(len(combined)),
        n_sources=len(selected),
        columns=columns,
        schema=schema,
        task_type=suggestion.get("task_type") or task_type,
        target_column=target,
        target_reason=suggestion.get("reason"),
        dropped_empty=dropped,
        strategy=("manual" if keep_indices is not None else strategy),
        n_source_frames=len(non_empty),
        dropped_incoherent=dropped_incoherent,
        clusters=clusters_meta,
    )
    return combined, report
