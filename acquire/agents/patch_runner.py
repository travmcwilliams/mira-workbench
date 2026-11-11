"""
agents/patch_runner.py

Lightweight "Scrape Patch" runner for customizable, parameterized scraping.

A "patch" is a reusable recipe supporting:
- URL templates (path style with {term}/{fmt})
- Query param style (base_url + query_params dict)
- Term formatting (slug, raw, plus-separated, etc.)
- Pagination (generate multiple pages per term)
- Extra dynamic context (start_date, end_date, etc.)
- Different scrape_strategies including "fetch_page_then_extract_links"

The runner expands (term x format x pages) into concrete tasks that the GUI
(or scripts) can feed into the existing ingestor/fetcher pipeline.

Supports the full "choose a patch → pick terms + formats + dates + pages → click scrape" flow.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, quote_plus, urlencode, urljoin, urlparse, urlunparse

# Robust root + imports (same pattern as other agents)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

PATCHES_PATH = ROOT / "data" / "scrape_patches.json"

# Optional heavy deps (BeautifulSoup is already used elsewhere in the platform)
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    BeautifulSoup = None  # type: ignore
    HAS_BS4 = False

try:
    from utils.fetcher import fetch_content
except Exception:
    fetch_content = None  # type: ignore


def load_patches() -> List[Dict[str, Any]]:
    """Load all defined scrape patches."""
    if not PATCHES_PATH.exists():
        return []
    with open(PATCHES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("patches", [])


def get_patch(patch_id: str) -> Optional[Dict[str, Any]]:
    for p in load_patches():
        if p.get("patch_id") == patch_id:
            return p
    return None


# ---------- Term formatting ----------

def _format_term(term: str, style: str = "slug") -> str:
    """Format a search term for URLs according to patch preference."""
    t = (term or "").strip()
    if not t:
        return "term"
    style = (style or "slug").lower()
    if style == "raw":
        return t
    if style == "plus":
        return re.sub(r"\s+", "+", t)
    if style == "quoted":
        return quote(t, safe="")
    if style == "quote_plus":
        return quote_plus(t)
    # default slug (good for paths)
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-+", "-", t).strip("-")
    return t or "term"


# ---------- Core URL building (supports both styles + context) ----------

def _apply_context(value: str, context: Optional[Dict[str, Any]]) -> str:
    """Substitute {start_date}, {end_date}, {max_pages} etc. into a string value."""
    if not value or not context:
        return value or ""
    ctx = {k: str(v) for k, v in (context or {}).items() if v is not None}
    # Also provide some common aliases
    if "start_date" in ctx:
        ctx.setdefault("from", ctx["start_date"])
        ctx.setdefault("issued_date.gte", ctx["start_date"])
    if "end_date" in ctx:
        ctx.setdefault("to", ctx["end_date"])
        ctx.setdefault("issued_date.lte", ctx["end_date"])
    try:
        return value.format(**ctx)
    except Exception:
        # Fallback: simple replace for keys we know
        out = value
        for k, v in ctx.items():
            out = out.replace("{" + k + "}", v)
        return out


def build_concrete_url(
    patch: Dict[str, Any],
    term: str,
    fmt: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    custom_template: Optional[str] = None,
    custom_base: Optional[str] = None,
) -> str:
    """
    Build one final URL for a (term, fmt) combination, respecting the patch's url_style.
    Supports pagination via context (e.g. context={"page": 2}).
    """
    context = context or {}
    term_style = patch.get("term_style", "slug")
    term_val = _format_term(term, term_style)

    # fmt can be uppercased for some APIs
    fmt_val = fmt.strip().lower()
    fmt_upper = fmt_val.upper()

    url_style = patch.get("url_style", "template")

    if url_style == "query":
        base = custom_base or patch.get("base_url", "")
        if not base:
            base = "https://example.com/search"
        qparams = dict(patch.get("query_params", {}))
        # Substitute placeholders in param values
        resolved = {}
        for k, v in qparams.items():
            if isinstance(v, str):
                val = v.replace("{term}", term_val).replace("{fmt}", fmt_val).replace("{TERM}", term).replace("{FMT}", fmt_upper)
                val = _apply_context(val, context)
                resolved[k] = val
            else:
                resolved[k] = v
        # Add/override page etc from context
        for extra_key in ("page", "offset", "limit", "p"):
            if extra_key in context and context[extra_key] is not None:
                resolved[extra_key] = str(context[extra_key])
        # Merge any caller extra_params
        extra = context.get("extra_query", {}) or {}
        resolved.update({k: str(v) for k, v in extra.items()})

        # Build final URL
        parsed = urlparse(base)
        existing_q = parse_qs(parsed.query)
        for k, v in resolved.items():
            existing_q[k] = [v] if not isinstance(v, list) else v
        new_query = urlencode(existing_q, doseq=True)
        url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
        # If base had no query originally and we want clean look
        if not parsed.query and "?" not in url and new_query:
            url = url + "?" + new_query if not url.endswith("?") else url + new_query
        return url

    # Default: template style (path or full template)
    template = custom_template or patch.get("url_template", "")
    if not template:
        return ""

    # Allow {base} substitution for the custom patch
    if custom_base and "{base}" in template:
        template = template.replace("{base}", custom_base.rstrip("/"))

    try:
        url = template.format(
            term=term_val,
            fmt=fmt_val,
            TERM=term,
            FMT=fmt_upper,
            **{k: str(v) for k, v in context.items() if v is not None}
        )
    except Exception:
        url = template.replace("{term}", term_val).replace("{fmt}", fmt_val)
        url = url.replace("{TERM}", term).replace("{FMT}", fmt_upper)
        for k, v in (context or {}).items():
            if v is not None:
                url = url.replace("{" + str(k) + "}", str(v))

    # Append any extra query from context
    extra = context.get("extra_query", {}) or {}
    if extra:
        p = urlparse(url)
        q = parse_qs(p.query)
        for k, v in extra.items():
            q[k] = [str(v)]
        url = urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), p.fragment))

    return url


# ---------- Pagination helper ----------

def _generate_page_contexts(patch: Dict[str, Any], base_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """If patch declares pagination, yield contexts for each page."""
    base_context = dict(base_context or {})
    pag = patch.get("pagination") or {}
    if not pag:
        return [base_context]

    ptype = pag.get("type", "page")
    param = pag.get("param", "page")
    start = int(pag.get("start", 1))
    count = int(pag.get("count", pag.get("max_pages", 1)))

    contexts = []
    for i in range(count):
        page_num = start + i
        ctx = dict(base_context)
        if ptype == "page":
            ctx[param] = page_num
            ctx["page"] = page_num
        elif ptype == "offset":
            offset = int(pag.get("offset_step", 10)) * i
            ctx[param] = offset
            ctx["offset"] = offset
        else:
            ctx[param] = page_num
        contexts.append(ctx)
    return contexts


# ---------- Main expansion (now much richer) ----------

def expand_patch(
    patch: Dict[str, Any],
    terms: List[str],
    formats: List[str],
    *,
    context: Optional[Dict[str, Any]] = None,
    custom_template: Optional[str] = None,
    custom_base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Expand a patch + terms + formats (+ optional dates, pagination, etc. in context)
    into a list of executable tasks.

    Returns richer task dicts:
    {
        "task_id",
        "source_url": final url for this combination (may include page param),
        "term", "fmt",
        "page": optional,
        "suggested_source_id", "suggested_source_name",
        "scraper_type",
        "mapping",
        "patch_id",
        "search_url": (if different from the final url, e.g. results page),
        "needs_discovery": bool,
        "strategy": the patch's scrape_strategy
    }
    """
    tasks: List[Dict[str, Any]] = []
    context = dict(context or {})
    default_scraper = patch.get("default_scraper_type", "has_download_link")
    default_mapping = patch.get("default_mapping", {}) or {}
    strategy = patch.get("scrape_strategy", "direct_download")
    term_style = patch.get("term_style", "slug")

    # Determine if this patch run should do link discovery
    needs_discovery = "extract_links" in str(strategy).lower() or "discover" in str(strategy).lower()

    page_contexts = _generate_page_contexts(patch, context)

    seen = set()
    for raw_term in terms:
        if not raw_term or not str(raw_term).strip():
            continue
        term = str(raw_term).strip()

        for fmt in formats:
            if not fmt:
                continue
            fmt = str(fmt).strip().lower()

            for page_ctx in page_contexts:
                full_ctx = dict(context)
                full_ctx.update(page_ctx)

                url = build_concrete_url(
                    patch,
                    term,
                    fmt,
                    context=full_ctx,
                    custom_template=custom_template,
                    custom_base=custom_base,
                )
                if not url:
                    continue

                page = page_ctx.get("page") or page_ctx.get("offset")
                task_id = f"{patch.get('patch_id','p')}-{ _format_term(term, term_style) }-{fmt}"
                if page:
                    task_id += f"-p{page}"

                if task_id in seen:
                    continue
                seen.add(task_id)

                suggested_id = task_id
                suggested_name = f"{patch.get('name','Patch')}: {term} ({fmt})" + (f" page {page}" if page else "")

                task = {
                    "task_id": task_id,
                    "source_url": url,
                    "term": term,
                    "fmt": fmt,
                    "page": page,
                    "suggested_source_id": suggested_id,
                    "suggested_source_name": suggested_name,
                    "scraper_type": default_scraper,
                    "mapping": default_mapping,
                    "patch_id": patch.get("patch_id"),
                    "strategy": strategy,
                    "needs_discovery": needs_discovery,
                    "search_url": url,   # for discovery mode this is the results page
                }
                tasks.append(task)

    return tasks


# ---------- Link discovery (for has_search_form / results pages) ----------

def discover_download_links(
    page_url: str,
    *,
    exts: Optional[List[str]] = None,
    source_id: str = "",
    base_url: Optional[str] = None,
) -> List[str]:
    """
    Fetch a results/search page and extract direct download links.
    Returns list of absolute URLs to files (csv, xlsx, pdf, zip, etc.).
    """
    if not fetch_content or not HAS_BS4:
        return []

    exts = exts or [".csv", ".xlsx", ".xls", ".pdf", ".zip", ".json", ".tsv"]
    exts_lower = [e.lower() if not e.startswith(".") else e.lower() for e in exts]

    html = fetch_content(page_url, source_id=source_id or "patch-discovery")
    if not html:
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []

    found: List[str] = []
    base = base_url or page_url

    # Look for <a> tags
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript"):
            continue
        href_lower = href.lower()
        # Match common download indicators
        is_download = any(ext in href_lower for ext in exts_lower) or \
                      any(kw in href_lower for kw in ["download", "export", "getfile", "attachment"])
        if not is_download:
            continue
        absolute = urljoin(base, href)
        if absolute not in found:
            found.append(absolute)

    # Also look for buttons or data attributes that might point to downloads
    for el in soup.find_all(["button", "a", "link"], attrs={"data-href": True}):
        h = el.get("data-href", "")
        if any(ext in h.lower() for ext in exts_lower):
            absolute = urljoin(base, h)
            if absolute not in found:
                found.append(absolute)

    return found


def build_temp_source(task: Dict[str, Any], region: str = "00", region_full: str = "Custom Patch") -> Dict[str, Any]:
    """Turn an expanded task into a registry-style source dict for ingest_source()."""
    return {
        "source_id": task["suggested_source_id"],
        "source_name": task["suggested_source_name"],
        "region": region,
        "region_full": region_full,
        "source_url": task["source_url"],
        "scraper_type": task.get("scraper_type", "has_download_link"),
        "data_method": "scrape",
        "last_scraped": None,
        "column_mapping": task.get("mapping") or {},
    }


# ---------- Convenience: run a patch end-to-end (used by CLI or advanced GUI) ----------

def run_patch(
    patch: Dict[str, Any],
    terms: List[str],
    formats: List[str],
    *,
    context: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    ingest_func: Optional[callable] = None,   # pass ingest_source or wrapper
    region: str = "00",
) -> List[Dict[str, Any]]:
    """
    High-level helper: expand then (optionally) execute.
    For full power with discovery, the GUI currently drives the loop so it can show live progress.
    This is useful for scripts.
    """
    from agents.source_ingestor import ingest_source as default_ingest

    ingest = ingest_func or default_ingest
    tasks = expand_patch(patch, terms, formats, context=context)
    results = []

    for task in tasks:
        temp = build_temp_source(task, region=region)
        if task.get("needs_discovery"):
            # Discover then ingest each found link
            links = discover_download_links(
                task["source_url"],
                exts=patch.get("link_exts"),
                source_id=task["patch_id"] or "discovery",
                base_url=patch.get("base_url")
            )
            for link in links:
                sub_task = dict(task)
                sub_task["source_url"] = link
                sub_task["suggested_source_id"] = task["suggested_source_id"] + "-disc"
                sub_task["suggested_source_name"] = task["suggested_source_name"] + " (discovered)"
                sub_src = build_temp_source(sub_task, region=region)
                try:
                    rows = ingest(sub_src, local_file=None, dry_run=dry_run)
                    results.append({"task": sub_task["task_id"], "url": link, "status": "ok", "rows": rows})
                except Exception as ex:
                    results.append({"task": sub_task["task_id"], "url": link, "status": "error", "error": str(ex)[:120]})
        else:
            try:
                rows = ingest(temp, local_file=None, dry_run=dry_run)
                results.append({"task": task["task_id"], "url": task["source_url"], "status": "ok", "rows": rows})
            except Exception as ex:
                results.append({"task": task["task_id"], "url": task["source_url"], "status": "error", "error": str(ex)[:120]})
    return results


if __name__ == "__main__":
    # Quick self-test / demo
    patches = load_patches()
    print(f"Loaded {len(patches)} patches")
    for p in patches:
        print(" -", p["patch_id"], p["name"])

    # Demo expansion with pagination + context
    p = get_patch("public-csv-airport-style") or get_patch("grants-keyword-direct")
    if p:
        ctx = {"start_date": "2024-01-01", "page": 1}
        tasks = expand_patch(p, ["education", "Health & Human Services"], ["csv", "xlsx"], context=ctx)
        print("\nExpanded (with sample context):")
        for t in tasks[:6]:
            print("  ", t["source_url"], "->", t["suggested_source_id"], "needs_discovery=", t.get("needs_discovery"))

    # Test query style if present
    p2 = get_patch("datagov-ckan-search")
    if p2:
        tasks = expand_patch(p2, ["grant opportunities"], ["CSV"], context={"start_date": "2025-01-01"})
        print("\nData.gov style example:")
        for t in tasks[:2]:
            print("  ", t["source_url"])
