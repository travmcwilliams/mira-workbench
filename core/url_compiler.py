"""
core/url_compiler.py — turn free text / URLs / pasted data into a dataset.

The Scraping Studio accepts one smart input that may be a research *topic*, a blob
of *URLs*, or raw *data*. This module:

  - ``classify_input``  — guess which of the three a blob is.
  - ``extract_urls``    — pull and de-dup every URL out of a blob.
  - ``preview_url``     — fetch a URL, report status / content-type, and detect how
    much tabular data it holds (HTML tables, CSV, JSON) without committing it.
  - ``load_url_to_df``  — best-effort load of a single URL into a DataFrame.
  - ``build_project_from_urls`` — compile several URLs into one combined Project
    (raw + working data, detected schema, suggested target) ready for the pipeline.

Network and parsing are best-effort and fully guarded: a dead URL yields an error
row, never an exception. Reuses ``core.data_loader`` / ``core.ingest`` so the
output is identical in shape to the rest of MIRA's ingest paths.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)


# --- classification -----------------------------------------------------------
def extract_urls(text: str) -> list[str]:
    """All http(s) URLs in `text`, de-duplicated, trailing punctuation trimmed."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _URL_RE.findall(text or ""):
        u = m.rstrip(".,;)")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _looks_like_data(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if (t[0], t[-1]) in (("{", "}"), ("[", "]")):
        return True  # JSON-ish
    head = t.splitlines()[:5]
    if len(head) >= 2:
        # consistent delimiter across the first lines => tabular paste
        for delim in (",", "\t", "|", ";"):
            counts = [ln.count(delim) for ln in head]
            if counts[0] >= 1 and len(set(counts)) == 1:
                return True
    return False


def classify_input(text: str) -> str:
    """Return 'url' | 'data' | 'topic'."""
    if extract_urls(text):
        return "url"
    if _looks_like_data(text):
        return "data"
    return "topic"


# --- single-URL preview / load ------------------------------------------------
@dataclass
class UrlProbe:
    url: str
    ok: bool = False
    status: int | None = None
    content_type: str = ""
    kind: str = "unknown"          # html_table | csv | json | html | error
    n_tables: int = 0
    rows: int = 0
    cols: int = 0
    note: str = ""
    elapsed_ms: int = 0


def _get(url: str, timeout: int = 20) -> requests.Response:
    return requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)


def preview_url(url: str, *, timeout: int = 20) -> UrlProbe:
    """Fetch and characterize a URL without persisting anything."""
    import time

    probe = UrlProbe(url=url)
    t0 = time.time()
    try:
        resp = _get(url, timeout=timeout)
        probe.status = resp.status_code
        probe.content_type = resp.headers.get("content-type", "").split(";")[0].strip()
        probe.ok = resp.ok
        if not resp.ok:
            probe.kind = "error"
            probe.note = f"HTTP {resp.status_code}"
            return probe

        ct = probe.content_type.lower()
        text = resp.text
        if "json" in ct or url.lower().endswith(".json"):
            df = _json_to_df(text)
            probe.kind, probe.rows, probe.cols = "json", len(df), df.shape[1]
        elif "csv" in ct or url.lower().endswith((".csv", ".tsv")):
            df = pd.read_csv(io.StringIO(text))
            probe.kind, probe.rows, probe.cols = "csv", len(df), df.shape[1]
        else:
            tables = _safe_read_html(text)
            if tables:
                biggest = max(tables, key=lambda d: d.shape[0] * d.shape[1])
                probe.kind = "html_table"
                probe.n_tables = len(tables)
                probe.rows, probe.cols = biggest.shape
            else:
                probe.kind = "html"
                probe.note = "no parseable tables"
    except Exception as exc:  # noqa: BLE001
        probe.kind = "error"
        probe.note = str(exc)[:160]
    finally:
        probe.elapsed_ms = int((time.time() - t0) * 1000)
    return probe


def _safe_read_html(text: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(io.StringIO(text))
    except Exception:
        return []


def _json_to_df(text: str) -> pd.DataFrame:
    import json

    data = json.loads(text)
    if isinstance(data, list):
        return pd.json_normalize(data)
    if isinstance(data, dict):
        # find the first list-of-records value, else normalize the dict itself
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return pd.json_normalize(v)
        return pd.json_normalize(data)
    return pd.DataFrame()


def load_url_to_df(url: str, *, timeout: int = 25) -> pd.DataFrame | None:
    """Best-effort: return the largest table / record set at `url`, else None."""
    try:
        resp = _get(url, timeout=timeout)
        if not resp.ok:
            return None
        ct = resp.headers.get("content-type", "").lower()
        text = resp.text
        if "json" in ct or url.lower().endswith(".json"):
            df = _json_to_df(text)
        elif "csv" in ct or url.lower().endswith((".csv", ".tsv")):
            df = pd.read_csv(io.StringIO(text))
        else:
            tables = _safe_read_html(text)
            if not tables:
                return None
            df = max(tables, key=lambda d: d.shape[0] * d.shape[1])
        return df if df is not None and not df.empty else None
    except Exception:
        return None


def parse_pasted_data(text: str) -> pd.DataFrame | None:
    """Parse a pasted CSV/TSV/JSON blob into a DataFrame."""
    t = (text or "").strip()
    if not t:
        return None
    if (t[0], t[-1]) in (("{", "}"), ("[", "]")):
        try:
            return _json_to_df(t)
        except Exception:
            return None
    for delim in (",", "\t", "|", ";"):
        try:
            df = pd.read_csv(io.StringIO(t), sep=delim)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return None


# --- compile many URLs into a project -----------------------------------------
@dataclass
class CompileResult:
    project_id: str | None = None
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    target_column: str | None = None
    task_type: str = "regression"
    per_url: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].map(str)
    return df


def build_project_from_urls(
    urls: list[str],
    *,
    project_name: str | None = None,
    task_type: str = "regression",
    combine: bool = True,
) -> CompileResult:
    """Load each URL, combine, and create a pipeline-ready Project."""
    frames: list[pd.DataFrame] = []
    per_url: list[dict[str, Any]] = []
    for u in urls:
        df = load_url_to_df(u)
        if df is None or df.empty:
            per_url.append({"url": u, "rows": 0, "status": "no data"})
            continue
        df = df.copy()
        df["__source"] = u[:200]
        frames.append(df)
        per_url.append({"url": u, "rows": int(len(df)), "cols": int(df.shape[1]),
                        "status": "ok"})

    if not frames:
        return CompileResult(per_url=per_url, error="No URL produced any data.")

    if combine:
        combined = pd.concat(frames, ignore_index=True, sort=False)
    else:
        combined = frames[0]
    combined = _sanitize_for_parquet(combined)

    from core.data_loader import detect_schema, suggest_target
    from core.ingest import ingest_dataframe
    from core.project import Project

    project = Project.create(
        name=project_name or "URL compile", task_type=task_type
    )
    ingest_dataframe(project, combined, source_name="url_compiler", auto_profile=False)

    schema = detect_schema(combined)
    suggestion = suggest_target(combined, schema, task_type=task_type)
    project.meta["schema"] = schema
    project.meta["task_type"] = suggestion.get("task_type") or task_type
    if suggestion.get("target_column"):
        project.meta["target_column"] = suggestion["target_column"]
    project.meta["url_compile"] = {"urls": urls, "n_sources": len(frames)}
    project.save_meta()

    return CompileResult(
        project_id=project.id,
        rows=int(len(combined)),
        columns=list(combined.columns),
        target_column=suggestion.get("target_column"),
        task_type=project.meta["task_type"],
        per_url=per_url,
    )
