"""
core/dataset_analysis.py

Preprocessing- and analysis-oriented facts about a project's dataset, beyond the
bare schema. Reuses core/preanalysis (the same stats the UI workbench shows) and
adds:

  - target analysis (class balance for classification, distribution for regression)
  - data-quality flags (constant / all-missing / high-missing / high-cardinality /
    id-like columns, duplicate rows)
  - leakage signals (a feature near-perfectly correlated with, or a duplicate of,
    the target) — mira's known foot-gun
  - a per-column preprocessing recommendation rollup (drop / impute / encode / log)

These facts feed the per-task context briefs and the sprint planner so
preprocessing decisions are data-aware. Results are cached per project keyed on a
cheap data fingerprint and recomputed when the working data changes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.data_loader import detect_schema
from core.preanalysis import (
    column_summary,
    missing_matrix_summary,
    recommend_encoding,
    recommend_missing_strategy,
)
from core.project import Project

# Above this row count we sample for the per-column distribution stats (exact
# missing/duplicate counts are still computed on the full frame — they're cheap).
_SAMPLE_CAP = 200_000
_HIGH_MISSING_PCT = 50.0
_HIGH_CARDINALITY = 50
_LEAKAGE_CORR = 0.98


def data_fingerprint(project: Project) -> str | None:
    """Cheap hash of the working data file (mtime + size). None if absent."""
    wp = project.root / "data" / "working.parquet"
    if not wp.exists():
        return None
    try:
        st = wp.stat()
        return hashlib.sha1(f"{int(st.st_mtime)}:{st.st_size}".encode()).hexdigest()[:16]
    except OSError:
        return None


def _num(v: Any) -> Any:
    """Coerce numpy scalars to JSON-native; pass through None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    return v


# --- Sub-analyses ------------------------------------------------------------

def _target_analysis(df: pd.DataFrame, target: str | None,
                     task_type: str | None, schema: dict[str, str]) -> dict[str, Any]:
    if not target or target not in df.columns:
        return {"name": target, "kind": None, "note": "target not in data"}

    s = df[target].dropna()
    if s.empty:
        return {"name": target, "kind": None, "note": "target all-null"}

    kind = task_type
    if kind not in ("classification", "regression"):
        stype = schema.get(target)
        kind = "classification" if stype in ("categorical", "id") else "regression"

    if kind == "classification":
        counts = s.value_counts()
        n = int(counts.sum())
        majority = int(counts.iloc[0])
        minority = int(counts.iloc[-1])
        return {
            "name": target, "kind": "classification",
            "n_classes": int(counts.size),
            "majority_pct": round(100.0 * majority / n, 2),
            "minority_pct": round(100.0 * minority / n, 2),
            "imbalance_ratio": round(majority / minority, 2) if minority else None,
            "class_counts": {str(k): int(v) for k, v in counts.head(12).items()},
        }

    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty:
        return {"name": target, "kind": "regression", "note": "non-numeric target"}
    return {
        "name": target, "kind": "regression",
        "mean": _num(clean.mean()), "std": _num(clean.std()),
        "min": _num(clean.min()), "max": _num(clean.max()),
        "skew": _num(clean.skew()),
        "zeros_pct": round(100.0 * float((clean == 0).mean()), 2),
        "n_unique": int(clean.nunique()),
    }


def _quality_flags(summary: pd.DataFrame, miss: dict[str, Any], n_rows: int,
                   schema: dict[str, str], target: str | None) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    for _, r in summary.iterrows():
        col = r["column"]
        if col == target:
            continue
        stype = r["schema_type"]
        if r["unique"] <= 1:
            flags.append({"column": col, "issue": "constant", "detail": "single value"})
        if r["missing_pct"] >= 100.0:
            flags.append({"column": col, "issue": "all_missing", "detail": "100% null"})
        elif r["missing_pct"] >= _HIGH_MISSING_PCT:
            flags.append({"column": col, "issue": "high_missing",
                          "detail": f"{r['missing_pct']}% null"})
        if stype == "categorical" and r["unique"] > _HIGH_CARDINALITY:
            flags.append({"column": col, "issue": "high_cardinality",
                          "detail": f"{r['unique']} levels"})
        if stype == "id":
            flags.append({"column": col, "issue": "id_like",
                          "detail": "near-unique; likely identifier"})
    if miss.get("duplicate_rows", 0) > 0:
        flags.append({"column": "*", "issue": "duplicate_rows",
                      "detail": f"{miss['duplicate_rows']} duplicate rows"})
    return flags


def _leakage_signals(df: pd.DataFrame, target: str | None,
                     schema: dict[str, str]) -> list[dict[str, Any]]:
    if not target or target not in df.columns:
        return []
    signals: list[dict[str, Any]] = []
    tgt = df[target]
    numeric_tgt = pd.to_numeric(tgt, errors="coerce")
    has_numeric_tgt = numeric_tgt.notna().any() and schema.get(target) != "categorical"

    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        # exact duplicate of the target
        try:
            if s.equals(tgt):
                signals.append({"column": col, "reason": "duplicate of target",
                                "metric": 1.0})
                continue
        except Exception:
            pass
        if has_numeric_tgt and pd.api.types.is_numeric_dtype(s):
            try:
                corr = float(pd.to_numeric(s, errors="coerce").corr(numeric_tgt))
            except Exception:
                corr = float("nan")
            if corr == corr and abs(corr) >= _LEAKAGE_CORR:
                signals.append({"column": col, "reason": "near-perfect correlation",
                                "metric": round(corr, 4)})
    return signals


def _recommendations(summary: pd.DataFrame, schema: dict[str, str],
                     df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    drop: list[dict[str, str]] = []
    impute: dict[str, str] = {}
    encode: dict[str, str] = {}
    log_transform: list[str] = []

    for _, r in summary.iterrows():
        col = r["column"]
        if col == target:
            continue
        stype = r["schema_type"]
        # drop candidates
        if r["unique"] <= 1:
            drop.append({"column": col, "reason": "constant"})
            continue
        if r["missing_pct"] >= 100.0:
            drop.append({"column": col, "reason": "all-missing"})
            continue
        if stype == "id":
            drop.append({"column": col, "reason": "identifier"})
            continue
        # impute / encode
        if r["missing_pct"] > 0:
            impute[col] = recommend_missing_strategy(stype, r["missing_pct"])
        enc = recommend_encoding(stype, int(r["unique"]))
        if enc != "none":
            encode[col] = enc
        # log transform for skewed positive numerics
        if stype == "numeric":
            try:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if not series.empty and (series >= 0).all() and abs(series.skew()) > 2:
                    log_transform.append(col)
            except Exception:
                pass

    return {"drop": drop, "impute": impute, "encode": encode,
            "log_transform": log_transform}


# --- Top-level analysis ------------------------------------------------------

def analyze_dataframe(df: pd.DataFrame, *, schema: dict[str, str] | None = None,
                      target: str | None = None, task_type: str | None = None
                      ) -> dict[str, Any]:
    """Full structured analysis of a dataframe (JSON-serializable)."""
    if df is None or df.empty:
        return {"available": False, "rows": 0, "cols": 0}

    schema = schema or detect_schema(df)
    n_rows = len(df)
    miss = missing_matrix_summary(df)

    sampled = n_rows > _SAMPLE_CAP
    work = df.head(_SAMPLE_CAP) if sampled else df

    summary_df = column_summary(work, schema)
    columns = [
        {k: _num(v) for k, v in row.items()}
        for row in summary_df.to_dict(orient="records")
    ]

    return {
        "available": True,
        "rows": n_rows,
        "cols": int(df.shape[1]),
        "sampled": sampled,
        "sample_rows": int(len(work)) if sampled else None,
        "missing": miss,
        "target": _target_analysis(work, target, task_type, schema),
        "columns": columns,
        "quality_flags": _quality_flags(summary_df, miss, n_rows, schema, target),
        "leakage": _leakage_signals(work, target, schema),
        "recommendations": _recommendations(summary_df, schema, work, target),
    }


def compact_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """Bounded form for embedding in context briefs / planner (AX2: signal only)."""
    if not analysis.get("available"):
        return {"available": False}
    rec = analysis.get("recommendations", {})
    flags = analysis.get("quality_flags", [])
    return {
        "available": True,
        "rows": analysis["rows"],
        "cols": analysis["cols"],
        "sampled": analysis.get("sampled", False),
        "missing_cell_pct": analysis["missing"].get("missing_cell_pct"),
        "duplicate_rows": analysis["missing"].get("duplicate_rows"),
        "target": analysis.get("target"),
        "issues": [f"{f['column']}: {f['issue']} ({f['detail']})" for f in flags[:12]],
        "leakage": [f"{s['column']} ({s['reason']}, {s['metric']})"
                    for s in analysis.get("leakage", [])],
        "recommend": {
            "drop": [d["column"] for d in rec.get("drop", [])],
            "impute": rec.get("impute", {}),
            "encode": rec.get("encode", {}),
            "log_transform": rec.get("log_transform", []),
        },
    }


# --- Project-level cache -----------------------------------------------------

def analyze_project(project: Project, *, force: bool = False) -> dict[str, Any]:
    """Analyze a project's working data, cached by data fingerprint.

    Cache: artifacts/dataset_analysis.json + meta['dataset_analysis'] pointer.
    """
    fp = data_fingerprint(project)
    if fp is None:
        return {"available": False, "note": "no working data"}

    cache_path = project.root / "artifacts" / "dataset_analysis.json"
    meta_ptr = project.meta.get("dataset_analysis", {})
    if not force and meta_ptr.get("fingerprint") == fp and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass

    try:
        df = project.get_working_data()
    except Exception as exc:
        return {"available": False, "note": f"load failed: {exc}"}

    analysis = analyze_dataframe(
        df,
        schema=project.meta.get("schema") or None,
        target=project.meta.get("target_column"),
        task_type=project.meta.get("task_type"),
    )
    analysis["fingerprint"] = fp
    analysis["generated_at"] = datetime.datetime.now(UTC).isoformat() + "Z"

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(analysis, indent=2))
        project.meta["dataset_analysis"] = {
            "fingerprint": fp, "generated_at": analysis["generated_at"],
            "rows": analysis.get("rows"), "cols": analysis.get("cols"),
        }
        project.save_meta()
    except OSError:
        pass
    return analysis


# --- Render ------------------------------------------------------------------

def render_markdown(analysis: dict[str, Any]) -> str:
    if not analysis.get("available"):
        return "_No dataset analysis available (no working data)._"
    lines = ["## Dataset analysis",
             f"- {analysis['rows']:,} rows × {analysis['cols']} cols"
             + ("  (sampled)" if analysis.get("sampled") else ""),
             f"- missing cells: {analysis['missing'].get('missing_cell_pct')}%  ·  "
             f"duplicate rows: {analysis['missing'].get('duplicate_rows')}"]
    tgt = analysis.get("target") or {}
    if tgt.get("kind") == "classification":
        lines.append(f"- target `{tgt.get('name')}`: {tgt.get('n_classes')} classes, "
                     f"imbalance {tgt.get('imbalance_ratio')}× "
                     f"(minority {tgt.get('minority_pct')}%)")
    elif tgt.get("kind") == "regression":
        lines.append(f"- target `{tgt.get('name')}`: mean {tgt.get('mean')}, "
                     f"std {tgt.get('std')}, skew {tgt.get('skew')}")
    if analysis.get("leakage"):
        lines.append("- ⚠️ **possible leakage**: "
                     + "; ".join(f"{s['column']} ({s['reason']})"
                                 for s in analysis["leakage"]))
    flags = analysis.get("quality_flags", [])
    if flags:
        lines.append("- quality flags: "
                     + "; ".join(f"{f['column']}:{f['issue']}" for f in flags[:10]))
    rec = analysis.get("recommendations", {})
    if rec.get("drop"):
        lines.append("- recommend drop: "
                     + ", ".join(d["column"] for d in rec["drop"]))
    if rec.get("impute"):
        lines.append("- recommend impute: "
                     + ", ".join(f"{c}→{s}" for c, s in rec["impute"].items()))
    if rec.get("encode"):
        lines.append("- recommend encode: "
                     + ", ".join(f"{c}→{e}" for c, e in rec["encode"].items()))
    return "\n".join(lines)
