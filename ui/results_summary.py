"""Results summary — "what happened" for a finished project, from disk.

The live-run renderers (ui/stage_results.py) only fire while a pipeline is
running; this module reads the same artifacts back from the project folder so
a finished project's outcome is viewable any time without re-running.
`assemble_results` is pure (no Streamlit) and never raises on missing pieces —
a half-finished project just yields fewer sections.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ui.components import image_gallery
from ui.theme import PLOTLY_TEMPLATE


def assemble_results(project_root: Path) -> dict[str, Any]:
    """Collect a finished project's results from its folder. Never raises."""
    root = Path(project_root)
    out: dict[str, Any] = {
        "meta": {}, "metrics": None, "leaderboard": None, "best_model": None,
        "eval_images": [], "history": [], "completed_stages": [],
    }

    try:
        out["meta"] = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        out["completed_stages"] = list(out["meta"].get("completed_stages") or [])
    except Exception:
        pass

    metrics_path = root / "artifacts" / "eval" / "metrics.json"
    try:
        out["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        pass

    lb_path = root / "artifacts" / "leaderboard.csv"
    try:
        out["leaderboard"] = pd.read_csv(lb_path).to_dict("records")
    except Exception:
        pass

    # Best model: model-stage run summary first, else leaderboard row 0.
    try:
        from core.stage_history import get_last_stage_run

        run = get_last_stage_run(root, "model")
        if run and isinstance(run.get("summary"), dict):
            out["best_model"] = run["summary"].get("best_model")
    except Exception:
        pass
    if not out["best_model"] and out["leaderboard"]:
        first = out["leaderboard"][0]
        out["best_model"] = first.get("model") or first.get("Model") or None

    eval_dir = root / "artifacts" / "eval"
    if eval_dir.is_dir():
        out["eval_images"] = sorted(str(p) for p in eval_dir.glob("*.png"))

    try:
        from core.stage_history import get_stage_history

        if (root / "state.db").exists():
            out["history"] = get_stage_history(root, limit=20)
    except Exception:
        pass

    return out


_SCALAR_ORDER = ("r2", "accuracy", "f1", "mae", "rmse", "auc")


def render_results_summary(project) -> None:
    results = assemble_results(project.root)
    meta = results["meta"] or project.meta
    stages = results["completed_stages"]

    st.title(f"Results — {meta.get('name', project.id)}")
    st.caption(
        f"ID: {project.id}  •  Task: {meta.get('task_type', '—')}  •  "
        f"Target: {meta.get('target_column') or '—'}  •  "
        f"Stages complete: {len(stages)}"
    )

    if not stages:
        st.info("No pipeline stages have completed yet. Run the pipeline from "
                "**Home** or **Walkthrough**, then come back.")
        st.page_link("pages/home.py", label="Go to Home", icon="🏠")
        return

    # --- headline metrics ---------------------------------------------------
    metrics = results["metrics"] or {}
    scalars = [(k, v) for k, v in metrics.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    scalars.sort(key=lambda kv: (_SCALAR_ORDER.index(kv[0])
                                 if kv[0] in _SCALAR_ORDER else 99, kv[0]))
    if results["best_model"] or scalars:
        cols = st.columns(min(4, max(1, len(scalars))) + 1)
        cols[0].metric("Best model", str(results["best_model"] or "—"))
        for col, (k, v) in zip(cols[1:], scalars[:4]):
            col.metric(k, f"{v:.4g}")
        if len(scalars) > 4 or any(not isinstance(v, (int, float))
                                   for v in metrics.values()):
            with st.expander("All metrics"):
                st.json(metrics)
    else:
        st.info("No evaluation metrics found (the `evaluate` stage hasn't run).")

    # --- leaderboard ----------------------------------------------------------
    if results["leaderboard"]:
        st.subheader("Model leaderboard")
        lb = pd.DataFrame(results["leaderboard"])
        st.dataframe(lb, use_container_width=True, hide_index=True)
        numeric_cols = [c for c in lb.columns if pd.api.types.is_numeric_dtype(lb[c])]
        name_col = next((c for c in lb.columns if c.lower() == "model"), None)
        if name_col and numeric_cols:
            try:
                import plotly.express as px

                fig = px.bar(lb, x=numeric_cols[0], y=name_col, orientation="h",
                             template=PLOTLY_TEMPLATE, height=80 + 40 * len(lb))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    # --- evaluation plots ------------------------------------------------------
    if results["eval_images"]:
        st.subheader("Evaluation plots")
        image_gallery([Path(p) for p in results["eval_images"]], max_cols=2)

    # --- stage timeline ---------------------------------------------------------
    if results["history"]:
        st.subheader("Stage timeline")
        hist = pd.DataFrame(results["history"])
        keep = [c for c in ("stage", "status", "run_at", "duration_ms") if c in hist.columns]
        st.dataframe(hist[keep] if keep else hist,
                     use_container_width=True, hide_index=True)
