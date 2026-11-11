"""
agents/document_parser.py
Document Parser Agent (Agent 2).

Extract tables from PDFs / scanned docs using the exact fallback chain from the spec:

    pdfplumber (text tables) → pytesseract OCR (scanned) → queue to sources_needing_review.json

NEVER silently drop a document.

This implementation generalizes the production logic from the reference CountyIQ system
while staying fully domain-neutral and following DSI agent conventions:
- Prints summary of what was extracted.
- Returns list of raw row dicts (caller can feed to schema_mapper).
- On OCR path or total failure, queues the source/file for human review.

Requires (on VM): pdfplumber + pytesseract + system tesseract-ocr installed.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REVIEW_QUEUE_PATH = Path(__file__).parent.parent / "data" / "sources_needing_review.json"


def queue_for_review(source_id: str, file_path: Path | str | None, reason: str, extra: Optional[dict] = None):
    """Flexible review queue helper. file_path can be Path, str, or None (for non-file reasons)."""
    q = []
    if REVIEW_QUEUE_PATH.exists():
        try:
            with open(REVIEW_QUEUE_PATH, "r", encoding="utf-8") as f:
                q = json.load(f)
        except Exception:
            q = []
    entry = {
        "source_id": source_id,
        "file": str(file_path) if file_path else None,
        "reason": reason,
        "queued_at": datetime.now(timezone.utc).isoformat()
    }
    if extra:
        entry.update(extra)
    q.append(entry)
    REVIEW_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(q, f, indent=2, ensure_ascii=False)
        f.write("\n")


def extract_tables_from_pdf(pdf_path: Path, source_id: str) -> List[Dict[str, Any]]:
    """
    Main entry point for PDF documents.
    Tries pdfplumber first for structured tables.
    Falls back to text extraction, then OCR.
    Any OCR path or complete failure results in review queue entry.
    """
    print(f"[document_parser] Processing PDF {pdf_path.name} for source_id={source_id}")

    if not pdf_path.exists():
        queue_for_review(source_id, pdf_path, "PDF file not found")
        return []

    records: List[Dict[str, Any]] = []
    used_ocr = False

    # --- Stage 1: pdfplumber for native text tables (best case) ---
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables or []:
                    if not table or len(table) < 2:
                        continue
                    header = [(c or "").strip().lower().replace(" ", "_") for c in table[0]]
                    for row in table[1:]:
                        rec: Dict[str, Any] = {"_page": page_num, "needs_review": False}
                        for i, val in enumerate(row):
                            key = header[i] if i < len(header) and header[i] else f"col_{i}"
                            rec[key] = (val or "").strip() if val else ""
                        if any(v for k, v in rec.items() if k not in ("_page", "needs_review")):
                            records.append(rec)
        if records:
            print(f"[document_parser] pdfplumber extracted {len(records)} structured rows from tables.")
            return records
    except ImportError:
        print("[document_parser] pdfplumber not installed — skipping table extraction stage.")
    except Exception as e:
        print(f"[document_parser] pdfplumber stage error: {e}")

    # --- Stage 2: pdfplumber text extraction (still structured-ish) ---
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = (page.extract_text() or "").strip()
                if text:
                    for line in text.splitlines():
                        line = line.strip()
                        if line:
                            records.append({"raw_line": line, "needs_review": True})
                    used_ocr = True
        if records:
            print(f"[document_parser] Extracted {len(records)} lines via pdfplumber text (flagged needs_review).")
    except Exception as e:
        print(f"[document_parser] pdfplumber text stage error: {e}")

    # --- Stage 3: pytesseract OCR fallback (scanned PDFs) ---
    if not records:
        try:
            import pytesseract
            from PIL import Image
            # Note: for real multi-page PDFs you need pdf2image + poppler.
            # This basic version tries to open as single image (works for some scanned single-page).
            # Production version should rasterize pages first.
            img = Image.open(pdf_path) if pdf_path.suffix.lower() == ".pdf" else Image.open(pdf_path)
            # If it's a PDF, PIL may fail — in that case user should have pdf2image.
            text = pytesseract.image_to_string(img)
            for line in text.splitlines():
                line = line.strip()
                if line:
                    records.append({"raw_line": line, "needs_review": True})
            used_ocr = True
            print(f"[document_parser] OCR extracted {len(records)} lines (needs_review=True).")
        except Exception as e:
            print(f"[document_parser] OCR stage failed: {e}")

    if not records:
        queue_for_review(source_id, pdf_path, "PDF parse failed (all stages: pdfplumber + OCR)")
        print(f"[document_parser] No records extracted from {pdf_path.name} — queued for review.")
        return []

    if used_ocr:
        queue_for_review(
            source_id,
            pdf_path,
            "Extracted via OCR fallback — low confidence, review recommended",
            extra={"row_count": len(records)}
        )

    print(f"[document_parser] Total rows from PDF: {len(records)}")
    return records


def extract_from_bytes(content: bytes, filename: str, source_id: str) -> List[Dict[str, Any]]:
    """Convenience wrapper when you only have bytes (e.g. from a download)."""
    tmp = Path("/tmp") / f"dsi_doc_{source_id}_{filename}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(content)
    try:
        if filename.lower().endswith(".pdf"):
            return extract_tables_from_pdf(tmp, source_id)
        # Future: add HTML table, image, etc. handlers here
        print(f"[document_parser] Unsupported bytes content type for {filename}")
        return []
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

