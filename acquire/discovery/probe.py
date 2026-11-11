"""
discovery/probe.py

Given a candidate URL, fetch it and figure out *what's actually scrapeable*:
direct data-file links, HTML tables, and which of the user's requested fields
appear. This is what fills in the checkbox row ("3 tables, 2/3 fields") and what
turns a chosen candidate into a DataFrame.

validate_url() and the link-drilldown are generalized from CountyIQ
(scripts/county_url_finder.py::validate_url / find_tax_lien_data_url).
"""

from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .scoring import DATA_EXTENSIONS, tokenize

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
_EXT_RE = re.compile(r"\.(csv|tsv|xlsx?|zip|json|parquet|pdf|docx)(\?|$)", re.I)


def _default_fetch(url: str) -> Optional[str]:
    """Fetch page HTML. Uses MIRA's stealth fetcher when available."""
    try:
        from utils.fetcher import fetch_content  # acquire env on sys.path
        return fetch_content(url, source_id="discovery")
    except Exception:
        try:
            r = requests.get(url, headers={"User-Agent": _UA}, timeout=15, verify=False)
            return r.text if r.status_code == 200 else None
        except Exception:
            return None


def _render_fetch(url: str) -> Optional[str]:
    """Force a JS-rendered fetch via Playwright. Returns None if unavailable.

    Data portals often inject their download links client-side, so the static
    HTML has no .csv/.xlsx anchor even though the page clearly shows one.
    """
    try:
        from utils.fetcher import PLAYWRIGHT_AVAILABLE, STRATEGY_FUNCS
        if not PLAYWRIGHT_AVAILABLE:
            return None
        for strat in ("playwright_headless", "playwright_firefox"):
            func = STRATEGY_FUNCS.get(strat)
            if func:
                html_out = func(url)
                if html_out:
                    return html_out
    except Exception:
        return None
    return None


def _scan_page(page_html: str, base_url: str):
    """Extract (data_links, table_headers, visible_text) from a page."""
    data_links = extract_data_links(page_html, base_url)
    headers = _table_headers(page_html)
    text = BeautifulSoup(page_html, "html.parser").get_text(" ", strip=True)[:8000]
    return data_links, headers, text


def validate_url(url: str) -> str:
    """'ok' | 'redirect:{u}' | 'failed' | 'failed:{code/err}'  (CountyIQ pattern)."""
    try:
        r = requests.get(url, timeout=12, allow_redirects=True,
                         headers={"User-Agent": _UA}, verify=False)
        if r.status_code != 200:
            return f"failed:{r.status_code}"
        if len(r.text) < 300:
            return "failed"
        head = r.text[:2000].lower()
        if "page not found" in head or ">404<" in head:
            return "failed"
        if r.url.rstrip("/") != url.rstrip("/"):
            return f"redirect:{r.url}"
        return "ok"
    except Exception as e:
        return f"failed:{type(e).__name__}"


def _ext_of(url: str) -> Optional[str]:
    m = _EXT_RE.search(url or "")
    return m.group(1).lower() if m else None


def extract_data_links(page_html: str, base_url: str) -> list[dict[str, str]]:
    """All anchor hrefs that point at a tabular data file, resolved to absolute."""
    soup = BeautifulSoup(page_html, "html.parser")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        ext = _ext_of(href)
        if not ext:
            continue
        full = urljoin(base_url, href)
        if not full.startswith("http") or full in seen:
            continue
        seen.add(full)
        out.append({"url": full, "ext": ext, "text": a.get_text(strip=True)[:120]})
    return out


def _table_headers(page_html: str) -> list[list[str]]:
    soup = BeautifulSoup(page_html, "html.parser")
    headers: list[list[str]] = []
    for table in soup.find_all("table"):
        cells = [th.get_text(strip=True) for th in table.find_all("th")]
        if not cells:
            first = table.find("tr")
            if first:
                cells = [td.get_text(strip=True) for td in first.find_all(["td", "th"])]
        headers.append(cells)
    return headers


def match_fields(field_names: list[str], haystack: str) -> list[str]:
    """Which requested fields appear (token-level) in the page/headers/links text."""
    hay_tokens = tokenize(haystack)
    matched: list[str] = []
    for f in field_names:
        ftoks = tokenize(f)
        if ftoks and ftoks & hay_tokens:
            matched.append(f)
    return matched


def probe_candidate(
    candidate: dict[str, Any],
    field_names: list[str],
    *,
    fetch_fn: Optional[Callable[[str], Optional[str]]] = None,
    page_html: Optional[str] = None,
) -> dict[str, Any]:
    """
    Enrich a candidate in place with scrapeability detail. Pass `page_html` to
    probe offline (tests); otherwise the page is fetched.
    """
    url = candidate["url"]
    fetch_fn = fetch_fn or _default_fetch

    # A candidate whose URL is itself a data file needs no fetch.
    direct_ext = _ext_of(url)
    if direct_ext:
        candidate.update(
            reachable=True, data_url=url, data_format=direct_ext,
            n_tables=0, data_links=[{"url": url, "ext": direct_ext, "text": ""}],
        )
        matched = match_fields(field_names, f"{candidate.get('title','')} {candidate.get('snippet','')} {url}")
        candidate["matched_fields"] = matched
        candidate["field_coverage"] = round(len(matched) / len(field_names), 2) if field_names else 0.0
        return candidate

    provided = page_html is not None
    if page_html is None:
        page_html = fetch_fn(url)
    if not page_html:
        candidate.update(reachable=False, data_url=None, data_format=None,
                         n_tables=0, data_links=[], matched_fields=[], field_coverage=0.0)
        return candidate

    data_links, headers, soup_text = _scan_page(page_html, url)

    # JS-rendered pages: if the static HTML exposed nothing scrapeable, retry
    # with a Playwright render (only in live mode, never for caller-supplied HTML).
    if not data_links and not headers and not provided:
        rendered = _render_fetch(url)
        if rendered and rendered != page_html:
            r_links, r_headers, r_text = _scan_page(rendered, url)
            if r_links or r_headers:
                page_html, data_links, headers, soup_text = rendered, r_links, r_headers, r_text
                candidate["rendered"] = True

    haystack = " ".join(
        [soup_text]
        + [" ".join(h) for h in headers]
        + [dl["text"] + " " + dl["url"] for dl in data_links]
    )
    matched = match_fields(field_names, haystack)

    if data_links:
        # Prefer a data link whose text mentions a requested field, else the first.
        best = next(
            (dl for dl in data_links if match_fields(field_names, dl["text"] + " " + dl["url"])),
            data_links[0],
        )
        data_url, data_format = best["url"], best["ext"]
    elif headers:
        data_url, data_format = url, "html_table"
    else:
        data_url, data_format = None, None

    candidate.update(
        reachable=True,
        data_url=data_url,
        data_format=data_format,
        n_tables=len(headers),
        data_links=data_links,
        matched_fields=matched,
        field_coverage=round(len(matched) / len(field_names), 2) if field_names else 0.0,
    )
    return candidate


# --- Turning a chosen candidate into a DataFrame -------------------------------

def _clean_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully-empty rows/columns left by spreadsheet padding."""
    return df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)


def _named_cols(df: pd.DataFrame) -> int:
    return sum(1 for c in df.columns if not str(c).startswith("Unnamed"))


def _promote_header(path: Path, sheet: str) -> Optional[pd.DataFrame]:
    """When a sheet's header is buried under preamble rows, find the real header
    row (the populated, mostly-text row) and re-read from there."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    limit = min(10, len(raw))
    best_row, best_count = 0, -1
    for i in range(limit):
        row = raw.iloc[i]
        non_null = int(row.notna().sum())
        if non_null > best_count:
            best_count, best_row = non_null, i
    df = pd.read_excel(path, sheet_name=sheet, header=best_row)
    return _clean_sheet(df)


def _read_excel_best_sheet(path: Path) -> Optional[pd.DataFrame]:
    """Pick the sheet that actually holds the data, not the first one.

    Many published workbooks lead with a cover/methodology sheet; the generic
    'read sheet 0' grabs notes instead of the series. Score each sheet by
    rows x named-columns and return the winner (promoting a buried header if the
    naive read is mostly 'Unnamed').
    """
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception:
        return None
    best: Optional[pd.DataFrame] = None
    best_score = -1
    for name, raw in sheets.items():
        if raw is None or raw.empty:
            continue
        df = _clean_sheet(raw)
        if df.empty:
            continue
        # If the header looks wrong (mostly Unnamed), try to recover it.
        if _named_cols(df) <= max(1, len(df.columns) // 2):
            try:
                promoted = _promote_header(path, name)
                if promoted is not None and _named_cols(promoted) > _named_cols(df):
                    df = promoted
            except Exception:
                pass
        score = len(df) * max(_named_cols(df), 1)
        if score > best_score:
            best_score, best = score, df
    return best


def _frame_from_rows(rows: list[list[Optional[str]]]) -> Optional[pd.DataFrame]:
    """Build a DataFrame from raw extracted cells, promoting the first row to a header."""
    rows = [r for r in rows if r and any(str(c or "").strip() for c in r)]
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    padded = [list(r) + [None] * (width - len(r)) for r in rows]
    header = [str(c or "").strip().replace("\n", " ") or f"col_{i}"
              for i, c in enumerate(padded[0])]
    df = pd.DataFrame(padded[1:], columns=header)
    return _clean_sheet(df)


def read_pdf_tables(path: Path, *, max_pages: int = 40) -> list[pd.DataFrame]:
    """Every table pdfplumber can find, in page order.

    Scientific supplementary tables are overwhelmingly PDF-only; without this the
    probe reports a perfectly good source as having no data at all.
    """
    try:
        import pdfplumber
    except Exception:
        return []
    out: list[pd.DataFrame] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages[:max_pages]:
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    continue
                for raw in tables:
                    df = _frame_from_rows(raw)
                    if df is not None and not df.empty and df.shape[1] > 1:
                        out.append(df)
    except Exception:
        return out
    return out


def read_docx_tables(path: Path) -> list[pd.DataFrame]:
    """Every table in a .docx supplement, in document order."""
    try:
        import docx  # python-docx
    except Exception:
        return []
    out: list[pd.DataFrame] = []
    try:
        document = docx.Document(str(path))
    except Exception:
        return []
    for table in document.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        df = _frame_from_rows(rows)
        if df is not None and not df.empty and df.shape[1] > 1:
            out.append(df)
    return out


def _read_file_to_df(path: Path) -> Optional[pd.DataFrame]:
    """Robust file read. For workbooks, pick the best sheet rather than sheet 0."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = _read_excel_best_sheet(path)
        if df is not None and not df.empty:
            return df
    if suffix == ".pdf":
        tables = read_pdf_tables(path)
        return max(tables, key=lambda t: t.shape[0] * t.shape[1]) if tables else None
    if suffix == ".docx":
        tables = read_docx_tables(path)
        return max(tables, key=lambda t: t.shape[0] * t.shape[1]) if tables else None
    try:
        from agents.source_ingestor import detect_format_and_read
        df = detect_format_and_read(path, source_id="discovery")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    try:
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return None


def _pick_best_table(tables: list[pd.DataFrame], field_names: list[str]) -> pd.DataFrame:
    """Choose the HTML table that best matches requested fields, else the biggest."""
    if not tables:
        return pd.DataFrame()
    ftoks = tokenize(" ".join(field_names))
    best, best_key = tables[0], (-1, -1)
    for t in tables:
        overlap = len(ftoks & tokenize(" ".join(str(c) for c in t.columns))) if ftoks else 0
        key = (overlap, t.shape[0] * max(t.shape[1], 1))
        if key > best_key:
            best, best_key = t, key
    return best


def scrape_candidate_to_df(
    candidate: dict[str, Any],
    field_names: Optional[list[str]] = None,
    *,
    page_html: Optional[str] = None,
    fetch_fn: Optional[Callable[[str], Optional[str]]] = None,
) -> pd.DataFrame:
    """Download/parse a probed candidate into a DataFrame (empty on failure)."""
    field_names = field_names or []
    fmt = candidate.get("data_format")
    data_url = candidate.get("data_url")
    if not data_url:
        return pd.DataFrame()

    if fmt == "html_table":
        provided = page_html is not None
        if page_html is None:
            page_html = (fetch_fn or _default_fetch)(data_url)
        if not page_html:
            return pd.DataFrame()

        def _tables(html_in: str) -> list[pd.DataFrame]:
            try:
                return pd.read_html(io.StringIO(html_in))
            except Exception:
                return []

        tables = _tables(page_html)
        if not tables and not provided:  # maybe the table is rendered client-side
            rendered = _render_fetch(data_url)
            if rendered:
                tables = _tables(rendered)
        return _pick_best_table(tables, field_names)

    # File-backed source: download via stealth fetcher, then read.
    try:
        from utils.fetcher import download_file
    except Exception:
        download_file = None

    name = Path(urlparse(data_url).path).name or "download"
    if "." not in name:
        name += f".{fmt or 'csv'}"
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / name
        ok = False
        if download_file is not None:
            ok = download_file(data_url, dest, source_id="discovery")
        if not ok:
            try:
                r = requests.get(data_url, headers={"User-Agent": _UA}, timeout=30, verify=False)
                if r.status_code == 200:
                    dest.write_bytes(r.content)
                    ok = True
            except Exception:
                ok = False
        if not ok or not dest.exists():
            return pd.DataFrame()
        df = _read_file_to_df(dest)
        return df if df is not None else pd.DataFrame()


def _download(data_url: str, fmt: Optional[str], dest_dir: Path) -> Optional[Path]:
    """Fetch a data file into ``dest_dir``; returns the path or None."""
    try:
        from utils.fetcher import download_file
    except Exception:
        download_file = None

    name = Path(urlparse(data_url).path).name or "download"
    if "." not in name:
        name += f".{fmt or 'csv'}"
    dest = dest_dir / name
    ok = False
    if download_file is not None:
        ok = download_file(data_url, dest, source_id="discovery")
    if not ok:
        try:
            r = requests.get(data_url, headers={"User-Agent": _UA}, timeout=45,
                             verify=False)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                ok = True
        except Exception:
            ok = False
    return dest if ok and dest.exists() else None


def scrape_candidate_to_all_dfs(
    candidate: dict[str, Any],
    field_names: Optional[list[str]] = None,
    *,
    page_html: Optional[str] = None,
    fetch_fn: Optional[Callable[[str], Optional[str]]] = None,
    max_tables: int = 40,
) -> list[pd.DataFrame]:
    """Every table a candidate exposes, not just the best-matching one.

    ``scrape_candidate_to_df`` collapses a source to a single frame, which loses
    most of a scientific paper — a myology article typically carries a forelimb
    table, a hindlimb table and a muscle-mass table, and only one survives.
    ``webschema.consolidate`` already fuzzy-merges heterogeneous frames, so
    returning all of them costs nothing downstream.
    """
    field_names = field_names or []
    fmt = candidate.get("data_format")
    data_url = candidate.get("data_url")
    if not data_url:
        return []

    if fmt == "html_table":
        provided = page_html is not None
        if page_html is None:
            page_html = (fetch_fn or _default_fetch)(data_url)
        if not page_html:
            return []

        def _tables(html_in: str) -> list[pd.DataFrame]:
            try:
                return pd.read_html(io.StringIO(html_in))
            except Exception:
                return []

        tables = _tables(page_html)
        if not tables and not provided:
            rendered = _render_fetch(data_url)
            if rendered:
                tables = _tables(rendered)
        return [t for t in tables if not t.empty][:max_tables]

    with tempfile.TemporaryDirectory() as td:
        path = _download(data_url, fmt, Path(td))
        if path is None:
            return []
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return read_pdf_tables(path)[:max_tables]
        if suffix == ".docx":
            return read_docx_tables(path)[:max_tables]
        if suffix in (".xlsx", ".xls"):
            try:
                sheets = pd.read_excel(path, sheet_name=None)
            except Exception:
                sheets = {}
            out = [_clean_sheet(s) for s in sheets.values()
                   if s is not None and not s.empty]
            if out:
                return [d for d in out if not d.empty][:max_tables]
        df = _read_file_to_df(path)
        return [df] if df is not None and not df.empty else []
