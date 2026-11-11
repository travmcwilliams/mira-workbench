"""Bridge between MIRA acquire layer (DuckDB records) and the DS pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

MIRA_ROOT = Path(__file__).resolve().parents[1]
ACQUIRE_ROOT = MIRA_ROOT / "acquire"

CANONICAL_RECORD_COLUMNS = frozenset({
    "record_id",
    "entity_name",
    "location_text",
    "primary_value",
    "period",
    "source_name",
    "source_id",
    "region",
})

ALLOWED_MAPPING_KEYS = CANONICAL_RECORD_COLUMNS


def ensure_acquire_env() -> Path:
    """Put acquire packages on sys.path and set DuckDB location."""
    if str(ACQUIRE_ROOT) not in sys.path:
        sys.path.insert(0, str(ACQUIRE_ROOT))
    os.environ.setdefault("APP_PROJECT_ROOT", str(ACQUIRE_ROOT))
    os.environ.setdefault("APP_DB_PATH", str(ACQUIRE_ROOT / "db" / "app.duckdb"))
    return ACQUIRE_ROOT


def get_acquire_status() -> dict[str, Any]:
    ensure_acquire_env()
    from utils.db import DB_PATH, PROJECT_ROOT, get_record_count, init_db_schema

    init_db_schema()
    return {
        "acquire_root": str(PROJECT_ROOT),
        "db_path": str(DB_PATH),
        "record_count": get_record_count(),
    }


def load_registry_df() -> pd.DataFrame:
    ensure_acquire_env()
    from agents.source_ingestor import load_registry

    reg = load_registry()
    return pd.DataFrame(reg.get("sources", []))


def ingest_source_id(
    source_id: str,
    *,
    local_file: Path | str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run acquire ingest for one registry source. Rows land in `records`."""
    ensure_acquire_env()
    from agents.source_ingestor import ingest_by_id
    from utils.db import get_record_count, init_db_schema

    init_db_schema()
    before = get_record_count()
    path = Path(local_file) if local_file else None
    loaded = ingest_by_id(source_id, local_file=path, dry_run=dry_run)
    after = get_record_count()
    return {
        "source_id": source_id,
        "rows_loaded": loaded,
        "records_before": before,
        "records_after": after,
        "dry_run": dry_run,
    }


def pull_source_into_project(
    project: Any,
    source_id: str,
    *,
    scrape: bool = True,
    ingest_fn: Any | None = None,
) -> dict[str, Any]:
    """
    Scrape (optional) → records → project working data.
    `ingest_fn` is called as ingest_fn(project, df, source_name=...) after load.
    """
    ingest_summary: dict[str, Any] = {}
    if scrape:
        ingest_summary = ingest_source_id(source_id)

    df = load_records_df(source_id=source_id)
    if df.empty:
        raise ValueError(f"No rows in records for source_id={source_id}")

    if ingest_fn is None:
        raise ValueError("ingest_fn is required to persist into a project")

    ingest_fn(project, df, source_name=f"records:{source_id}")
    project.meta["acquire_source_id"] = source_id
    project.meta["data_origin"] = "records"
    project.save_meta()

    return {
        "source_id": source_id,
        "rows": len(df),
        "columns": list(df.columns),
        "scrape_ran": scrape,
        "ingest": ingest_summary,
    }


def load_registry() -> dict[str, Any]:
    ensure_acquire_env()
    from agents.source_ingestor import load_registry as _load

    return _load()


def save_registry(data: dict[str, Any]) -> None:
    ensure_acquire_env()
    from utils.db import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "source_registry.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def validate_registry() -> list[str]:
    ensure_acquire_env()
    from scripts.registry_repair import load_registry as _load, validate_source

    reg = _load()
    errs: list[str] = []
    for i, src in enumerate(reg.get("sources", [])):
        errs.extend(validate_source(src, i))
    return errs


def run_registry_repair(*, verify_only: bool = True) -> tuple[bool, str]:
    """Run registry_repair.py; returns (ok, stdout+stderr)."""
    ensure_acquire_env()
    script = ACQUIRE_ROOT / "scripts" / "registry_repair.py"
    args = [sys.executable, str(script)]
    if verify_only:
        args.append("--verify-only")
    proc = subprocess.run(args, capture_output=True, text=True, cwd=str(ACQUIRE_ROOT))
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()


def get_fetcher_status() -> dict[str, Any]:
    ensure_acquire_env()
    from utils.fetcher import PLAYWRIGHT_AVAILABLE

    return {"playwright_available": PLAYWRIGHT_AVAILABLE}


def test_fetch_url(url: str, *, source_id: str = "gui-test") -> str | None:
    """Fetch URL via acquire fetcher; returns content or None."""
    ensure_acquire_env()
    from utils.fetcher import fetch_content

    return fetch_content(url, source_id=source_id)


def load_scraper_run_log(*, limit: int = 50) -> list[dict[str, Any]]:
    ensure_acquire_env()
    from utils.run_logger import get_run_logger

    rl = get_run_logger()
    return list(rl._log[-limit:])


def load_review_queue() -> list[dict[str, Any]]:
    ensure_acquire_env()
    from utils.db import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "sources_needing_review.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def clear_review_queue() -> None:
    ensure_acquire_env()
    from utils.db import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "sources_needing_review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")


def records_stats_df() -> pd.DataFrame:
    """Quick aggregate stats from the records table."""
    ensure_acquire_env()
    from utils.db import get_db, init_db_schema

    init_db_schema()
    conn = get_db()
    return conn.execute("""
        SELECT
            COUNT(*) AS total_records,
            COUNT(DISTINCT source_id) AS sources,
            MAX(period) AS latest_period,
            COUNT(DISTINCT region) AS regions
        FROM records
    """).df()


def records_by_source_df() -> pd.DataFrame:
    ensure_acquire_env()
    from utils.db import get_db, init_db_schema

    init_db_schema()
    conn = get_db()
    return conn.execute("""
        SELECT source_id, source_name, COUNT(*) AS cnt,
               MIN(period) AS min_period, MAX(period) AS max_period
        FROM records
        GROUP BY source_id, source_name
        ORDER BY cnt DESC
    """).df()


def run_records_query(sql: str) -> pd.DataFrame:
    ensure_acquire_env()
    from utils.db import get_db, init_db_schema

    init_db_schema()
    conn = get_db()
    return conn.execute(sql).df()


def discover_sources(
    topic: str,
    fields: list[str] | None = None,
    *,
    prefer: str | None = None,
    max_candidates: int = 12,
    probe_top: int = 8,
    do_probe: bool = True,
) -> dict[str, Any]:
    """
    Topic + desired fields -> ranked, probed scrapeable candidates (for checkboxes).
    Returns {"provider", "topic", "fields", "candidates": [...]}.
    """
    ensure_acquire_env()
    from discovery.engine import discover
    from discovery.providers import get_provider

    fields = [f.strip() for f in (fields or []) if f and f.strip()]
    provider = get_provider(prefer=prefer)
    candidates = discover(
        topic,
        fields,
        provider=provider,
        max_candidates=max_candidates,
        probe_top=probe_top,
        do_probe=do_probe,
    )
    return {
        "provider": provider.name,
        "topic": topic,
        "fields": fields,
        "candidates": candidates,
    }


def probe_candidate_live(candidate: dict[str, Any],
                         fields: list[str] | None = None) -> dict[str, Any]:
    """Probe ONE discovery candidate in place (reachability, tables, format, coverage).

    Exposed so the UI can drive the probe loop itself and show a live status indicator —
    which site is being checked and the shape of data found — instead of one blind spinner.
    Fetch discovery candidates with ``do_probe=False``, then call this per candidate.
    """
    ensure_acquire_env()
    from discovery.probe import probe_candidate
    clean = [f.strip() for f in (fields or []) if f and f.strip()]
    try:
        probe_candidate(candidate, clean)
    except Exception as exc:  # a bad site must not abort the whole loop
        candidate.update(reachable=False, probe_error=str(exc)[:160])
    return candidate


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scraped tables (esp. messy spreadsheets with multi-row headers) often have
    object columns holding mixed int/str values, which pyarrow can't serialize.
    Stringify the non-null cells of object columns so the data can land; the
    user can clean/retype it in the pipeline.
    """
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].map(str)
    return df


def scrape_candidates_into_project(
    candidates: list[dict[str, Any]],
    *,
    topic: str = "discovered",
    fields: list[str] | None = None,
    task_type: str = "regression",
    project_name: str | None = None,
) -> dict[str, Any]:
    """
    Scrape selected candidates -> one combined DataFrame -> a new Project with
    raw/working data, detected schema, and a suggested target. Returns a summary
    dict including "project_id" (load with Project.load) for the pipeline handoff.
    """
    ensure_acquire_env()
    from discovery.probe import scrape_candidate_to_df

    fields = [f.strip() for f in (fields or []) if f and f.strip()]

    frames: list[pd.DataFrame] = []
    per_source: list[dict[str, Any]] = []
    for cand in candidates:
        label = cand.get("title") or cand.get("url", "source")
        try:
            df = scrape_candidate_to_df(cand, fields)
        except Exception as exc:
            per_source.append({"url": cand.get("url"), "rows": 0, "error": str(exc)[:160]})
            continue
        if df is None or df.empty:
            per_source.append({"url": cand.get("url"), "rows": 0, "error": "no rows parsed"})
            continue
        df = df.copy()
        df["__source"] = label[:120]
        frames.append(df)
        per_source.append({"url": cand.get("url"), "rows": int(len(df)), "cols": int(df.shape[1])})

    if not frames:
        return {"project_id": None, "rows": 0, "sources": per_source,
                "error": "No candidate produced any rows."}

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _sanitize_for_parquet(combined)

    # Build the project + run the standard ingest (schema detection, no profile).
    from core.data_loader import detect_schema, suggest_target
    from core.ingest import ingest_dataframe
    from core.project import Project

    project = Project.create(name=project_name or f"Discover: {topic}"[:60], task_type=task_type)
    ingest_dataframe(project, combined, source_name=f"discover:{topic}", auto_profile=False)

    schema = detect_schema(combined)
    suggestion = suggest_target(combined, schema, task_type=task_type)
    target = suggestion.get("target_column")
    project.meta["schema"] = schema
    project.meta["task_type"] = suggestion.get("task_type") or task_type
    if target:
        project.meta["target_column"] = target
    project.meta["discovery"] = {"topic": topic, "fields": fields,
                                 "n_sources": len(frames)}
    project.save_meta()

    return {
        "project_id": project.id,
        "rows": int(len(combined)),
        "columns": list(combined.columns),
        "target_column": target,
        "target_reason": suggestion.get("reason"),
        "task_type": project.meta["task_type"],
        "sources": per_source,
    }


# --------------------------------------------------------------------------- #
# Source Registry bridge (registry/*) — governed source-intelligence DB.
# Production wires the real DuckDB singleton; the registry API itself takes an
# explicit connection so it stays testable in isolation.
# --------------------------------------------------------------------------- #

def registry_setup(*, seed: bool = True, migrate: bool = True) -> dict[str, Any]:
    """Idempotent: create schema, optionally seed the curated spine + fold in the
    legacy JSON registry. Safe to run repeatedly."""
    ensure_acquire_env()
    from registry import init_registry_schema, migrate_json_registry, seed_from_config
    from registry import registry as R
    from utils.db import get_db

    conn = get_db()
    init_registry_schema(conn)
    out: dict[str, Any] = {"schema": "ok"}
    if seed:
        out["seed"] = seed_from_config(conn)
    if migrate:
        out["migrate"] = migrate_json_registry(conn)
    out["counts"] = R.registry_counts(conn)
    return out


def registry_status() -> dict[str, Any]:
    ensure_acquire_env()
    from registry import init_registry_schema
    from registry import registry as R
    from utils.db import get_db

    conn = get_db()
    init_registry_schema(conn)
    return R.registry_counts(conn)


def registry_sources_df(*, active_only: bool = True) -> pd.DataFrame:
    ensure_acquire_env()
    from registry import init_registry_schema
    from registry import registry as R
    from utils.db import get_db

    conn = get_db()
    init_registry_schema(conn)
    return R.list_sources(conn, active_only=active_only)


def registry_top_sources_df(*, limit: int = 25) -> pd.DataFrame:
    ensure_acquire_env()
    from registry import init_registry_schema
    from registry import registry as R
    from utils.db import get_db

    conn = get_db()
    init_registry_schema(conn)
    return R.top_sources_by_score(conn, limit=limit)


def registry_promote_discovery_candidates(
    candidates: list[dict[str, Any]],
    *,
    discovered_from: str = "discovery",
) -> int:
    """Fold discovery-engine candidates (discover_sources output) into the registry's
    candidate table for review. Returns the number recorded/bumped."""
    ensure_acquire_env()
    from registry import init_registry_schema
    from registry import registry as R
    from utils.db import get_db

    conn = get_db()
    init_registry_schema(conn)
    n = 0
    for c in candidates:
        url = c.get("data_url") or c.get("url")
        if not url:
            continue
        R.add_discovery_candidate(
            conn,
            discovered_url=url,
            discovered_name=c.get("title"),
            discovered_from=discovered_from,
            discovery_method="news_citation_graph" if c.get("reachable") else "inbound_link_graph",
            candidate_score=float(c.get("score") or 0.0),
        )
        n += 1
    return n


def load_records_df(
    *,
    source_id: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read canonical `records` table into a DataFrame for DS stages."""
    ensure_acquire_env()
    from utils.db import get_db, init_db_schema

    init_db_schema()
    conn = get_db()
    sql = "SELECT * FROM records"
    params: list[Any] = []
    if source_id:
        sql += " WHERE source_id = ?"
        params.append(source_id)
    sql += " ORDER BY source_id, record_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).df()