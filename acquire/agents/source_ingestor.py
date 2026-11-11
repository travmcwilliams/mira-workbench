"""
agents/source_ingestor.py
Source Ingestor Agent (Agent 1).

Downloads (or reads local), detects format (CSV/XLSX/PDF/ZIP by content, not extension),
hands off to parser/mapper, loads canonical rows, triggers dedup, appends to run log.

Per spec:
- Every ingest: print columns + first 3 rows before parsing.
- Print final record count after writing.
- Use utils.get_db() only; no own connections.
- For has_download_link: highest priority, parse immediately.
- Unzip archives and recurse.
- After load: always run dedup.
- Log to data/scraper_run_log.json

This module focuses on the direct-download path for the scaffold demo.
"""

import json
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import pandas as pd

# Robust import: ensure project root is on sys.path so we can use absolute package imports
# regardless of how the script is invoked (python foo.py, python -m, PYTHONPATH, etc.)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.document_parser import extract_tables_from_pdf, queue_for_review
from agents.schema_mapper import map_to_canonical
from utils.db import get_db, get_record_count, init_db_schema, run_dedup
from utils.fetcher import download_file
from utils.object_storage import archive_raw, ensure_raw_dir
from utils.run_logger import get_run_logger

REGISTRY_PATH = Path(__file__).parent.parent / "data" / "source_registry.json"
RUN_LOG_PATH = Path(__file__).parent.parent / "data" / "scraper_run_log.json"


def load_registry() -> Dict[str, Any]:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def append_run_log(entry: Dict[str, Any]):
    """Legacy simple append (kept for compatibility). Handles both old list format and new richer dict format."""
    data = []
    if RUN_LOG_PATH.exists():
        try:
            with open(RUN_LOG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
            elif isinstance(loaded, dict):
                data = loaded.get("log", [])
            else:
                data = []
        except Exception:
            data = []
    data.append(entry)
    # Write back in the richer format so RunLogger can read it cleanly
    payload = {"runs": {}, "log": data}
    with open(RUN_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Also feed the richer logger
    try:
        rl = get_run_logger()
        rl.append_legacy_entry(entry)
    except Exception:
        pass


# queue_for_review is now imported from agents.document_parser (centralized, richer)


def detect_format_and_read(path: Path, source_id: str = "") -> Optional[pd.DataFrame]:
    """
    Content sniff (not just extension) + delegate to specialized extractors.
    Supports CSV, XLSX, ZIP (recurse), and hands PDFs to document_parser.
    Prints columns + first 3 rows for tabular data (spec convention).
    """
    suffix = path.suffix.lower()

    # --- PDF path (use the real document_parser) ---
    if suffix == ".pdf":
        print(f"[ingestor] Handing PDF to document_parser: {path.name}")
        raw_rows = extract_tables_from_pdf(path, source_id or "unknown")
        if not raw_rows:
            return None
        # Convert list of dicts to DataFrame for uniform handling by mapper
        return pd.DataFrame(raw_rows)

    # --- ZIP: recurse ---
    if suffix == ".zip":
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(path) as z:
                z.extractall(td)
                for child in sorted(Path(td).rglob("*")):
                    if child.is_file() and child.suffix.lower() in {".csv", ".xlsx", ".xls", ".txt", ".pdf"}:
                        print(f"[ingestor] ZIP: recursing into {child.name}")
                        df = detect_format_and_read(child, source_id=source_id)
                        if df is not None and len(df) > 0:
                            return df
        return None

    # --- XLSX ---
    if suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path, engine="openpyxl")
            print(f"[ingestor] Read XLSX: {len(df)} rows")
            if len(df) > 0:
                cols = list(df.columns)
                print(f"  Columns: {cols}")
                print("  First 3 rows (truncated):")
                print(df.head(3).to_string())
            return df
        except Exception as e:
            print(f"[ingestor] XLSX read failed: {e}")
            return None

    # --- CSV / text (content sniff friendly) ---
    try:
        df = pd.read_csv(path)
        print(f"[ingestor] Read tabular file: {len(df)} rows from {path.name}")
        if len(df) > 0:
            cols = list(df.columns)
            print(f"  Columns: {cols}")
            print("  First 3 rows (truncated):")
            print(df.head(3).to_string())
        return df
    except Exception as e:
        print(f"[ingestor] CSV-style read failed for {path}: {e}")
        return None


def ingest_source(source: Dict[str, Any], local_file: Optional[Path] = None, dry_run: bool = False, mapping: Optional[Dict[str, str]] = None) -> int:
    """
    Ingest one source entry from registry.
    If local_file provided, use it (demo/dev). Else attempt download from source_url.
    mapping: optional explicit column mapping {canonical: raw_col} from registry or CLI.
    Returns rows loaded (after dedup).
    """
    source_id = source["source_id"]
    source_name = source["source_name"]
    region = source["region"]
    scraper_type = source["scraper_type"]
    url = source.get("source_url")

    # Explicit mapping from registry takes precedence unless overridden by caller (CLI/GUI)
    if mapping is None:
        mapping = source.get("column_mapping") or {}

    started = datetime.now(timezone.utc).isoformat()
    rows_in = 0
    rows_loaded = 0
    status = "ok"
    error = None

    rl = get_run_logger()
    run_id = rl.start_run(source_id, source_name)

    print(f"\n[ingestor] === Starting ingest source_id={source_id} name={source_name} type={scraper_type} ===")

    if scraper_type != "has_download_link":
        print(f"[ingestor] NOTE: {scraper_type} not fully wired in scaffold ingestor (see platform_scrapers/ and form automator).")
        if not local_file:
            status = "skipped"
            rl.finish_run(run_id, rows_in=0, rows_loaded=0, errors=0, status=status)
            append_run_log({"source_id": source_id, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                            "rows_in": 0, "rows_loaded": 0, "status": status, "error": "non-download type, no local_file override"})
            return 0

    raw_path = local_file
    if raw_path is None and url:
        print(f"[ingestor] Downloading via fetcher: {url} ...")
        rl.log_event(run_id, "download_attempt", {"url": url})
        raw_dir = ensure_raw_dir(source_id)
        candidate_name = Path(urlparse(url).path).name or f"download_{source_id}"
        if "." not in candidate_name:
            candidate_name += ".bin"
        raw_path = raw_dir / candidate_name
        if download_file(url, raw_path, source_id=source_id):
            archive_raw(source_id, raw_path)
            rl.log_event(run_id, "download_ok", {"path": str(raw_path)})
            print(f"[ingestor] Downloaded to {raw_path}")
        else:
            error = "download failed (all strategies)"
            status = "error"
            rl.log_event(run_id, "download_fail", {"url": url})
            print(f"[ingestor] {error}")

    if raw_path is None or not raw_path.exists():
        error = error or "no raw file available"
        status = "error"
        rl.finish_run(run_id, rows_in=0, rows_loaded=0, errors=1, status=status)
        append_run_log({"source_id": source_id, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                        "rows_in": 0, "rows_loaded": 0, "status": status, "error": error})
        queue_for_review(source_id, None, "ingest: no file")
        return 0

    # Read (now passes source_id so PDF path works)
    df = detect_format_and_read(raw_path, source_id=source_id)
    if df is None or len(df) == 0:
        if raw_path.suffix.lower() == ".pdf":
            # document_parser already queued it if OCR/low confidence
            status = "queued_or_empty"
        else:
            status = "error"
            error = "unreadable or empty format"
        rl.finish_run(run_id, rows_in=0, rows_loaded=0, errors=1 if status == "error" else 0, status=status)
        append_run_log({"source_id": source_id, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                        "rows_in": 0, "rows_loaded": 0, "status": status, "error": error})
        return 0

    rows_in = len(df)
    print(f"[ingestor] Read {rows_in} raw rows from {raw_path.name}")
    rl.log_event(run_id, "extract_ok", {"rows_in": rows_in, "file": str(raw_path)})

    # Map (explicit mapping from source/CLI if present)
    try:
        canonical = map_to_canonical(df, source_id, source_name, region, mapping=mapping)
        rows_loaded = len(canonical)
    except Exception as e:
        error = f"mapping failed: {e}"
        status = "error"
        rl.log_event(run_id, "parse_fail", {"error": str(e)})
        rl.finish_run(run_id, rows_in=rows_in, rows_loaded=0, errors=1, status=status)
        append_run_log({"source_id": source_id, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                        "rows_in": rows_in, "rows_loaded": 0, "status": status, "error": error})
        queue_for_review(source_id, None, "mapping error")
        return 0

    if dry_run:
        print("[ingestor] dry_run=True, not writing to DB.")
        rl.finish_run(run_id, rows_in=rows_in, rows_loaded=rows_loaded, errors=0, status="dry_run")
        return rows_loaded

    # Load
    conn = get_db()
    init_db_schema()

    conn.register("canon", canonical)
    conn.execute("""
        INSERT INTO records (record_id, entity_name, location_text, primary_value, period, source_name, source_id, region)
        SELECT record_id, entity_name, location_text, primary_value, period, source_name, source_id, region FROM canon
    """)
    conn.unregister("canon")

    # Always dedup after ingest
    deleted = run_dedup()
    final_count = get_record_count()
    print(f"[ingestor] Loaded. Dedup removed {deleted} dups. Total records now: {final_count}")

    rl.log_event(run_id, "records_loaded", {"rows_loaded": rows_loaded, "dedup_removed": deleted})
    if deleted > 0:
        rl.log_event(run_id, "duplicate_removed", {"count": deleted})

    finished = datetime.now(timezone.utc).isoformat()
    rl.finish_run(run_id, rows_in=rows_in, rows_loaded=rows_loaded, errors=0, status=status)

    append_run_log({
        "source_id": source_id,
        "started_at": started,
        "finished_at": finished,
        "rows_in": rows_in,
        "rows_loaded": rows_loaded,
        "status": status,
        "error": error
    })

    # Update last_scraped in registry (best effort)
    try:
        reg = load_registry()
        for s in reg["sources"]:
            if s["source_id"] == source_id:
                s["last_scraped"] = finished
                break
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception:
        pass

    return rows_loaded


def ingest_by_id(source_id: str, local_file: Optional[Path] = None, dry_run: bool = False, mapping: Optional[Dict[str, str]] = None) -> int:
    reg = load_registry()
    src = next((s for s in reg["sources"] if s["source_id"] == source_id), None)
    if not src:
        raise ValueError(f"source_id {source_id} not in registry")
    return ingest_source(src, local_file=local_file, dry_run=dry_run, mapping=mapping)


if __name__ == "__main__":
    # Simple CLI for manual test
    import argparse
    import json as _json
    p = argparse.ArgumentParser(description="Ingest a source (web or local file). Supports explicit column mapping.")
    p.add_argument("--source-id", required=True, help="source_id from registry")
    p.add_argument("--local-file", default=None, help="Use a local file instead of downloading")
    p.add_argument("--dry-run", action="store_true", help="Parse and map but do not write to DB")
    p.add_argument("--mapping", default=None,
                   help='Optional explicit column mapping as JSON string, e.g. \'{"iata":"record_id","airport":"entity_name","cnt":"primary_value"}\' '
                        '(overrides or supplements registry column_mapping)')
    args = p.parse_args()
    lf = Path(args.local_file) if args.local_file else None
    mapping = _json.loads(args.mapping) if args.mapping else None
    n = ingest_by_id(args.source_id, local_file=lf, dry_run=args.dry_run, mapping=mapping)
    print(f"\n[ingestor] Done. Rows prepared/loaded for {args.source_id}: {n}")
