"""
stages/profile.py

ydata-profiling wrapper.

Generates a rich HTML profiling report and saves it under the project's artifacts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.pipeline import StageResult
from core.project import Project


def default_settings(project: Project) -> dict[str, Any]:
    """Default settings for profiling stage."""
    return {
        "minimal_threshold": 100,  # switch to minimal=True if more columns than this
        "title": None,  # defaults to project name
    }


def _ensure_ydata_import_compat() -> None:
    """
    Let ydata-profiling import on a modern SciPy.

    ydata-profiling <= 4.10 does ``from scipy.fft import _pocketfft``, a private
    module SciPy removed in 1.18. It calls exactly one symbol from it, ``fft``,
    which is the public ``scipy.fft.fft``. Aliasing the private path keeps the
    stage working; the alternative (upgrading to 4.18) pins matplotlib <= 3.10
    and drags pandas and NumPy backwards for every project sharing this
    interpreter, to fix one optional stage.
    """
    import sys
    import types

    try:
        import scipy.fft as scipy_fft
    except ImportError:  # no SciPy at all -- let the real import fail below
        return

    if "scipy.fft._pocketfft" in sys.modules:
        return
    try:
        if getattr(scipy_fft, "_pocketfft", None) is not None:
            return
    except Exception:  # pragma: no cover - deprecation shims can raise
        return

    shim = types.ModuleType("scipy.fft._pocketfft")
    shim.fft = scipy_fft.fft
    shim.ifft = scipy_fft.ifft
    sys.modules["scipy.fft._pocketfft"] = shim
    scipy_fft._pocketfft = shim


def _ensure_numpy_alias_compat() -> None:
    """
    Restore ``np.NaN``, which NumPy removed in 2.0 and ydata-profiling <= 4.10
    still references while summarising text columns. It is a plain alias for
    ``np.nan``; without it the stage cannot profile any string column.
    """
    import numpy as np

    if not hasattr(np, "NaN"):
        np.NaN = np.nan


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    """
    Generate ydata-profiling report.
    - Uses minimal mode for wide datasets to keep it fast.
    - Saves HTML to artifacts/profile_report.html
    """
    if df is None or len(df) == 0:
        return StageResult(
            status="error", error="No data available for profiling. Upload data first."
        )

    _ensure_ydata_import_compat()
    _ensure_numpy_alias_compat()
    try:
        from ydata_profiling import ProfileReport
    except ImportError as e:
        import importlib.util

        if importlib.util.find_spec("ydata_profiling") is None:
            reason = "ydata-profiling not installed (pip install -e \".[automl]\")"
        else:
            reason = (f"ydata-profiling failed to import against the current "
                      f"SciPy/pandas/NumPy versions: {e}")
        # It's an optional extra, so a plain install must still get through the
        # pipeline: write a basic pandas profile instead of failing the stage.
        return _basic_profile(project, df, settings, reason)

    minimal = len(df.columns) > settings.get("minimal_threshold", 100)
    title = settings.get("title") or project.meta.get("name", "Dataset Profile")
    explorative = bool(settings.get("explorative", True)) and not minimal

    out_path = project.artifact_path("profile_report.html")

    def _render(use_minimal: bool, use_explorative: bool) -> None:
        ProfileReport(
            df,
            minimal=use_minimal,
            title=title,
            explorative=use_explorative,
        ).to_file(out_path)

    degraded_reason: str | None = None
    try:
        _render(minimal, explorative)
    except Exception as e:  # noqa: BLE001 - any failure should degrade, not lose the stage
        if minimal:
            raise
        # The full profile exercises code paths (chi-square over value_counts,
        # correlations, interactions) that ydata-profiling 4.10 cannot run against
        # Arrow-backed pandas 3 arrays. Minimal mode avoids them and still produces
        # a real report, which beats failing the stage or pinning the whole
        # environment back to pandas 2 for every other project on this interpreter.
        root = e
        seen = 0
        while (root.__cause__ or root.__context__) and seen < 10:
            root = root.__cause__ or root.__context__
            seen += 1
        degraded_reason = f"{type(root).__name__}: {root}"
        minimal, explorative = True, False
        _render(True, False)

    summary = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "minimal": bool(minimal),
        "degraded_to_minimal": degraded_reason is not None,
        "report_path": str(out_path.relative_to(project.root))
        if out_path.is_relative_to(project.root)
        else str(out_path),
    }
    if degraded_reason:
        summary["degraded_reason"] = degraded_reason

    return StageResult(
        status="success",
        output_df=df,  # profile does not transform data
        artifacts=[out_path],
        summary=summary,
    )


def _basic_profile(project: Project, df: pd.DataFrame, settings: dict[str, Any],
                   reason: str) -> StageResult:
    """Dependency-free profile: per-column type, missingness, cardinality, stats."""
    import html

    title = settings.get("title") or project.meta.get("name", "Dataset Profile")
    n = len(df)
    overview = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing %": (df.isna().mean() * 100).round(1),
        "distinct": df.nunique(dropna=True),
        "top value": [
            (str(s.mode(dropna=True).iloc[0])[:40] if s.notna().any() else "")
            for _, s in df.items()
        ],
    })
    numeric = df.select_dtypes("number")
    stats_html = (
        numeric.describe().T.round(3).to_html(classes="t")
        if not numeric.empty else "<p>No numeric columns.</p>"
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(str(title))}</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#1f2328}}
.t{{border-collapse:collapse;font-size:13px;margin-bottom:24px}}
.t th,.t td{{border:1px solid #d0d7de;padding:4px 8px;text-align:right}}
.t th{{background:#f6f8fa}} .note{{color:#59636e;font-size:13px}}
</style></head><body>
<h1>{html.escape(str(title))}</h1>
<p>{n:,} rows &middot; {len(df.columns)} columns &middot;
{int(df.duplicated().sum()):,} duplicate rows</p>
<p class="note">Basic profile ({html.escape(reason)}).</p>
<h2>Columns</h2>{overview.to_html(classes="t")}
<h2>Numeric summary</h2>{stats_html}
</body></html>"""
    out_path = project.artifact_path("profile_report.html")
    out_path.write_text(page, encoding="utf-8")
    return StageResult(
        status="success",
        output_df=df,
        artifacts=[out_path],
        summary={
            "rows": int(n),
            "cols": int(len(df.columns)),
            "basic_profile": True,
            "basic_profile_reason": reason,
            "report_path": str(out_path.relative_to(project.root))
            if out_path.is_relative_to(project.root)
            else str(out_path),
        },
    )


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    import streamlit as st

    settings["minimal_threshold"] = st.slider(
        "Minimal mode if columns exceed",
        min_value=20,
        max_value=500,
        value=int(settings.get("minimal_threshold", 100)),
        help="Wide datasets use faster minimal profiling.",
    )
    settings["title"] = st.text_input(
        "Report title",
        value=settings.get("title") or project.meta.get("name", "Dataset Profile"),
    )
    settings["explorative"] = st.checkbox(
        "Explorative mode (when not minimal)",
        value=bool(settings.get("explorative", True)),
    )
    return settings
