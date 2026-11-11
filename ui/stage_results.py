"""
ui/stage_results.py — per-stage live result renderer for MIRA's guided run.

As the pipeline runs (the Walkthrough "Save & run"), each stage's ``StageResult``
is rendered inline, the moment it completes: a one-line takeaway, key metrics, the
standard graphs a data scientist / ML engineer needs to judge a dataset's
readiness for preprocessing, plus any PNG / HTML / CSV artifacts the stage wrote.

The data-understanding stages (acquire / profile / eda) get a full **EDA
dashboard** — shape, dtype mix, missingness, distributions, cardinality,
correlations, target balance, outliers, quality flags, leakage signals, and a
concrete preprocessing-recommendation rollup. Transform stages (clean, feature
eng / select, prep) report what changed and why. Model / evaluate / explain show
the leaderboard, metrics, and diagnostic plots.

Every chart is wrapped in ``_guard`` so a single failure degrades to a caption
instead of blanking the page.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.dataset_analysis import analyze_dataframe
from core.pipeline import StageResult
from core.preanalysis import column_summary, correlation_preview, missing_matrix_summary
from core.project import Project

# Friendly stage labels (kept local so this module has no import cycle with the
# walkthrough page).
_STAGE_LABEL: dict[str, str] = {
    "business": "Frame the question",
    "acquire": "Acquire",
    "profile": "Profile",
    "eda": "Explore (EDA)",
    "clean": "Clean",
    "feature_eng": "Feature engineering",
    "feature_select": "Feature selection",
    "data_prep": "Prepare (split & scale)",
    "model": "Model (AutoML)",
    "evaluate": "Evaluate",
    "explain": "Explain (SHAP)",
    "deploy": "Export",
}

# Stages that benefit from the full data-understanding dashboard.
_EDA_STAGES = {"acquire", "profile", "eda"}

_PLOT_SAMPLE = 20_000  # cap rows fed to plotting for responsiveness


# --- small helpers ------------------------------------------------------------
@contextlib.contextmanager
def _guard(label: str):
    """Render a caption instead of crashing the page if a chart blows up."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 — defensive UI guard
        st.caption(f"_(couldn't render {label}: {exc})_")


def _fmt_ms(ms: int | None) -> str:
    if ms is None:
        return ""
    if ms < 1000:
        return f"{ms} ms"
    return f"{ms / 1000:.1f} s"


def _plot_frame(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > _PLOT_SAMPLE:
        return df.sample(_PLOT_SAMPLE, random_state=0)
    return df


def _schema(project: Project, df: pd.DataFrame) -> dict[str, str]:
    schema = project.meta.get("schema") or {}
    if schema:
        return schema
    try:
        from core.data_loader import detect_schema

        return detect_schema(df)
    except Exception:
        return {}


def snapshot_stats(project: Project) -> dict[str, Any] | None:
    """Cheap before-stage snapshot used to show what a transform changed.

    Returns None when there's no working data yet (e.g. before acquire).
    """
    try:
        if not project.has_raw_data():
            return None
        df = project.get_working_data()
    except Exception:
        return None
    miss = missing_matrix_summary(df)
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_cells": miss["missing_cells"],
        "missing_cell_pct": miss["missing_cell_pct"],
        "duplicate_rows": miss["duplicate_rows"],
    }


# --- artifact rendering -------------------------------------------------------
def _render_artifacts(artifacts: list[Path], key: str) -> None:
    imgs = [p for p in artifacts if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    htmls = [p for p in artifacts if p.suffix.lower() in (".html", ".htm")]
    csvs = [p for p in artifacts if p.suffix.lower() == ".csv"]
    jsons = [p for p in artifacts if p.suffix.lower() == ".json"]
    others = [
        p for p in artifacts
        if p.suffix.lower() in (".pkl", ".parquet", ".joblib", ".pickle")
    ]

    if imgs:
        st.markdown("**Charts**")
        cols = st.columns(min(2, len(imgs)))
        for i, p in enumerate(imgs):
            if p.exists():
                with cols[i % len(cols)]:
                    with _guard(p.name):
                        st.image(str(p), caption=p.name, use_container_width=True)

    for p in htmls:
        if p.exists():
            with _guard(p.name):
                st.markdown(f"**{p.name}** — interactive report")
                try:
                    data = p.read_bytes()
                    st.download_button(
                        f"⬇ Open {p.name}", data, file_name=p.name,
                        mime="text/html", key=f"{key}_dl_{p.stem}",
                    )
                except Exception:
                    st.caption(str(p))

    for p in csvs:
        if p.exists():
            with _guard(p.name), st.expander(f"📄 {p.name}", expanded=False):
                try:
                    st.dataframe(pd.read_csv(p).head(50), use_container_width=True)
                except Exception:
                    st.caption(str(p))

    for p in jsons:
        if p.exists():
            with _guard(p.name), st.expander(f"📄 {p.name}", expanded=False):
                try:
                    st.json(json.loads(p.read_text()))
                except Exception:
                    st.caption(str(p))

    if others:
        st.caption("Saved: " + ", ".join(p.name for p in others))


# --- the EDA dashboard (the "ready for preprocessing" view) -------------------
def _findings_from_analysis(a: dict[str, Any]) -> list[str]:
    """Short, decision-oriented bullets distilled from the structured analysis."""
    out: list[str] = []
    miss = a.get("missing", {})
    if miss.get("missing_cell_pct"):
        out.append(
            f"{miss['missing_cell_pct']}% of cells missing across "
            f"{miss.get('columns_with_missing', 0)} column(s)."
        )
    if miss.get("duplicate_rows"):
        out.append(f"{miss['duplicate_rows']:,} duplicate rows present.")

    tgt = a.get("target") or {}
    if tgt.get("kind") == "classification" and tgt.get("imbalance_ratio"):
        out.append(
            f"Target `{tgt.get('name')}`: {tgt.get('n_classes')} classes, "
            f"{tgt.get('imbalance_ratio')}× imbalance "
            f"(minority {tgt.get('minority_pct')}%)."
        )
    elif tgt.get("kind") == "regression" and tgt.get("skew") is not None:
        out.append(
            f"Target `{tgt.get('name')}`: skew {tgt.get('skew')}, "
            f"mean {tgt.get('mean')}, std {tgt.get('std')}."
        )

    flags = a.get("quality_flags", [])
    if flags:
        kinds: dict[str, int] = {}
        for f in flags:
            kinds[f["issue"]] = kinds.get(f["issue"], 0) + 1
        out.append("Quality flags: "
                   + ", ".join(f"{v}× {k}" for k, v in kinds.items()) + ".")

    rec = a.get("recommendations", {})
    bits = []
    if rec.get("drop"):
        bits.append(f"drop {len(rec['drop'])}")
    if rec.get("impute"):
        bits.append(f"impute {len(rec['impute'])}")
    if rec.get("encode"):
        bits.append(f"encode {len(rec['encode'])}")
    if rec.get("log_transform"):
        bits.append(f"log-transform {len(rec['log_transform'])}")
    if bits:
        out.append("Preprocessing to-do: " + ", ".join(bits) + " column(s).")
    return out


def render_eda_dashboard(
    project: Project, df: pd.DataFrame, *, key: str
) -> None:
    """Standard exploratory graphs + findings to prep a dataset for preprocessing."""
    if df is None or df.empty:
        st.info("No data to analyze.")
        return

    schema = _schema(project, df)
    target = project.meta.get("target_column")
    task = project.meta.get("task_type")
    pdf = _plot_frame(df)

    analysis = {}
    with _guard("dataset analysis"):
        analysis = analyze_dataframe(df, schema=schema, target=target, task_type=task)

    # ----- headline metrics -----
    miss = analysis.get("missing") or missing_matrix_summary(df)
    mem_mb = float(df.memory_usage(deep=True).sum()) / 1e6
    m = st.columns(5)
    m[0].metric("Rows", f"{len(df):,}")
    m[1].metric("Columns", f"{df.shape[1]}")
    m[2].metric("Missing cells", f"{miss.get('missing_cell_pct', 0)}%")
    m[3].metric("Duplicate rows", f"{miss.get('duplicate_rows', 0):,}")
    m[4].metric("Memory", f"{mem_mb:.1f} MB")

    # ----- short findings -----
    findings = _findings_from_analysis(analysis) if analysis.get("available") else []
    if findings:
        st.markdown("**Findings**")
        st.markdown("\n".join(f"- {f}" for f in findings))

    leakage = analysis.get("leakage", [])
    if leakage:
        st.warning(
            "⚠️ **Possible leakage** — review before modeling: "
            + "; ".join(f"`{s['column']}` ({s['reason']}, {s['metric']})"
                        for s in leakage)
        )

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ----- target analysis -----
    tgt = analysis.get("target") or {}
    if target and target in df.columns and tgt.get("kind"):
        st.markdown(f"**Target — `{target}` ({tgt['kind']})**")
        with _guard("target chart"):
            if tgt["kind"] == "classification":
                counts = df[target].value_counts().head(20)
                fig = px.bar(
                    x=counts.index.astype(str), y=counts.values,
                    labels={"x": target, "y": "count"},
                    title=f"Class balance — {tgt.get('imbalance_ratio', '?')}× imbalance",
                )
                fig.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"{key}_tgt")
            else:
                fig = px.histogram(
                    _plot_frame(df), x=target, nbins=40, marginal="box",
                    title=f"Target distribution (skew {tgt.get('skew')})",
                )
                fig.update_layout(height=340)
                st.plotly_chart(fig, use_container_width=True, key=f"{key}_tgt")

    # ----- missingness by column -----
    with _guard("missingness chart"):
        summ = column_summary(df, schema)
        miss_cols = summ[summ["missing_pct"] > 0].sort_values("missing_pct", ascending=False)
        if not miss_cols.empty:
            st.markdown("**Missingness by column**")
            top = miss_cols.head(25)
            fig = px.bar(
                top, x="missing_pct", y="column", orientation="h",
                labels={"missing_pct": "% missing", "column": ""},
                color="missing_pct", color_continuous_scale="Reds",
            )
            fig.update_layout(height=max(260, 22 * len(top)), yaxis={"autorange": "reversed"},
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, key=f"{key}_miss")

    # ----- numeric distributions -----
    feat_num = [c for c in num_cols if c != target]
    if feat_num:
        st.markdown("**Numeric distributions**")
        show = feat_num[:6]
        cols = st.columns(3)
        for i, c in enumerate(show):
            with cols[i % 3], _guard(f"hist {c}"):
                s = pd.to_numeric(pdf[c], errors="coerce").dropna()
                skew = s.skew() if len(s) > 2 else 0.0
                fig = px.histogram(s, nbins=30, title=f"{c}  (skew {skew:.2f})")
                fig.update_layout(height=240, showlegend=False, margin=dict(t=40, b=10),
                                  xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True, key=f"{key}_h_{i}")
        if len(feat_num) > len(show):
            st.caption(f"…and {len(feat_num) - len(show)} more numeric columns.")

    # ----- categorical cardinality -----
    if cat_cols:
        with _guard("cardinality chart"):
            card = (
                pd.DataFrame({"column": cat_cols,
                              "unique": [int(df[c].nunique(dropna=True)) for c in cat_cols]})
                .sort_values("unique", ascending=False).head(20)
            )
            st.markdown("**Categorical cardinality** (high = consider label-encoding / dropping)")
            fig = px.bar(card, x="unique", y="column", orientation="h",
                         labels={"unique": "distinct values", "column": ""})
            fig.update_layout(height=max(240, 22 * len(card)),
                              yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True, key=f"{key}_card")

    # ----- correlation heatmap -----
    with _guard("correlation heatmap"):
        corr = correlation_preview(df, target=target)
        if not corr.empty:
            st.markdown("**Correlation (numeric)**")
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                            zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True, key=f"{key}_corr")

    # ----- outliers (IQR) -----
    if feat_num:
        with _guard("outlier chart"):
            rows = []
            for c in feat_num:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(s) < 4:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                if n_out:
                    rows.append({"column": c, "outlier_pct": round(100.0 * n_out / len(s), 2)})
            if rows:
                odf = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).head(20)
                st.markdown("**Outliers** (IQR rule, % of rows beyond 1.5×IQR)")
                fig = px.bar(odf, x="outlier_pct", y="column", orientation="h",
                             labels={"outlier_pct": "% outliers", "column": ""},
                             color="outlier_pct", color_continuous_scale="Oranges")
                fig.update_layout(height=max(240, 22 * len(odf)),
                                  yaxis={"autorange": "reversed"}, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True, key=f"{key}_out")

    # ----- quality flags + recommendations -----
    flags = analysis.get("quality_flags", [])
    rec = analysis.get("recommendations", {})
    if flags or any(rec.get(k) for k in ("drop", "impute", "encode", "log_transform")):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Data-quality flags**")
            if flags:
                st.dataframe(pd.DataFrame(flags), use_container_width=True,
                             hide_index=True, height=min(280, 40 + 35 * len(flags)))
            else:
                st.success("No quality flags.")
        with c2:
            st.markdown("**Preprocessing recommendations**")
            if rec.get("drop"):
                st.markdown("- **Drop:** " + ", ".join(f"`{d['column']}` ({d['reason']})"
                                                       for d in rec["drop"][:12]))
            if rec.get("impute"):
                st.markdown("- **Impute:** " + ", ".join(f"`{c}`→{s}"
                                                        for c, s in list(rec["impute"].items())[:12]))
            if rec.get("encode"):
                st.markdown("- **Encode:** " + ", ".join(f"`{c}`→{e}"
                                                        for c, e in list(rec["encode"].items())[:12]))
            if rec.get("log_transform"):
                st.markdown("- **Log-transform:** "
                            + ", ".join(f"`{c}`" for c in rec["log_transform"][:12]))
            if not any(rec.get(k) for k in ("drop", "impute", "encode", "log_transform")):
                st.success("No preprocessing actions recommended.")

    # ----- column statistics table -----
    with _guard("column table"), st.expander("Column statistics (full table)", expanded=False):
        st.dataframe(
            column_summary(df, schema), use_container_width=True, hide_index=True,
            column_config={"missing_pct": st.column_config.NumberColumn(format="%.1f%%")},
        )


# --- transform-stage deltas ---------------------------------------------------
def _delta_metrics(before: dict[str, Any] | None, after_df: pd.DataFrame) -> None:
    after = missing_matrix_summary(after_df)
    cols = st.columns(4)
    if before:
        cols[0].metric("Rows", f"{len(after_df):,}",
                       delta=f"{len(after_df) - before['rows']:+,}")
        cols[1].metric("Columns", f"{after_df.shape[1]}",
                       delta=f"{after_df.shape[1] - before['cols']:+d}")
        cols[2].metric("Missing cells", f"{after['missing_cell_pct']}%",
                       delta=f"{after['missing_cell_pct'] - before['missing_cell_pct']:+.2f}%",
                       delta_color="inverse")
        cols[3].metric("Duplicate rows", f"{after['duplicate_rows']:,}",
                       delta=f"{after['duplicate_rows'] - before['duplicate_rows']:+,}",
                       delta_color="inverse")
    else:
        cols[0].metric("Rows", f"{len(after_df):,}")
        cols[1].metric("Columns", f"{after_df.shape[1]}")
        cols[2].metric("Missing cells", f"{after['missing_cell_pct']}%")
        cols[3].metric("Duplicate rows", f"{after['duplicate_rows']:,}")


def _render_clean(project, result, before, key) -> None:
    s = result.summary
    st.markdown(
        f"Imputed **{s.get('cells_imputed', 0):,}** cells · "
        f"dropped **{s.get('duplicates_dropped', 0):,}** duplicates · "
        f"removed **{s.get('rows_dropped_outliers', 0):,}** outlier rows."
    )
    with _guard("clean deltas"):
        _delta_metrics(before, project.get_working_data())


def _render_feature_eng(project, result, before, key) -> None:
    s = result.summary
    st.markdown(
        f"Added **{s.get('new_columns', 0)}** column(s) · "
        f"saved **{s.get('encoders_saved', 0)}** encoder(s) · "
        f"dropped **{len(s.get('dropped_columns', []))}**."
    )
    with _guard("fe deltas"):
        df = project.get_working_data()
        _delta_metrics(before, df)
        if before:
            new = [c for c in df.columns if c not in set(before["columns"])]
            if new:
                st.caption("New columns: " + ", ".join(f"`{c}`" for c in new[:25])
                           + (" …" if len(new) > 25 else ""))
        dropped_details = s.get("dropped_column_reasons", [])
        if dropped_details:
            st.caption(
                "Dropped columns: "
                + ", ".join(f"`{item['column']}` ({item['reason']})" for item in dropped_details)
            )


def _render_feature_select(project, result, before, key) -> None:
    s = result.summary
    orig, sel = s.get("original_features", 0), s.get("selected_features", 0)
    st.markdown(f"Kept **{sel}** of **{orig}** features (dropped **{s.get('dropped', 0)}**).")
    with _guard("fs chart"):
        fig = px.bar(
            x=["selected", "dropped"], y=[sel, max(0, orig - sel)],
            labels={"x": "", "y": "features"}, color=["selected", "dropped"],
            color_discrete_map={"selected": "#5b9bd5", "dropped": "#e0a3a3"},
        )
        fig.update_layout(height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_fs")
    if s.get("selected_columns"):
        st.caption("Selected: " + ", ".join(f"`{c}`" for c in s["selected_columns"]))


def _render_data_prep(project, result, before, key) -> None:
    s = result.summary
    c = st.columns(4)
    c[0].metric("Train rows", f"{s.get('n_train', 0):,}")
    c[1].metric("Test rows", f"{s.get('n_test', 0):,}")
    c[2].metric("Split", str(s.get("split_strategy", "?")))
    c[3].metric("Scaler", str(s.get("scaler", "?")))
    if s.get("split_strategy") == "random":
        st.info("Random split — for time-series targets prefer **temporal** to avoid "
                "look-ahead leakage.")
    if s.get("note"):
        st.caption(f"Note: {s['note']}")


def _render_model(project, result, before, key) -> None:
    s = result.summary
    st.markdown(
        f"Backend **{s.get('backend', '?')}** · compared **{s.get('n_models', 0)}** models · "
        f"best: **{s.get('best_model', '?')}**."
    )
    lb_path = project.artifact_path("leaderboard.csv")
    if lb_path.exists():
        with _guard("leaderboard"):
            lb = pd.read_csv(lb_path)
            st.dataframe(lb, use_container_width=True, hide_index=True, height=260)
            # try to chart the first numeric score column
            num = lb.select_dtypes(include=[np.number])
            label_col = "Model" if "Model" in lb.columns else lb.columns[0]
            if not num.empty and label_col in lb.columns:
                score = num.columns[0]
                fig = px.bar(lb.head(10), x=score, y=label_col, orientation="h",
                             title=f"Model comparison — {score}")
                fig.update_layout(height=320, yaxis={"autorange": "reversed"})
                st.plotly_chart(fig, use_container_width=True, key=f"{key}_lb")


def _render_evaluate(project, result, before, key) -> None:
    s = result.summary
    st.markdown(f"Scored **{s.get('best_model', s.get('backend', ''))}** for "
                f"*{s.get('task', '')}* on target `{s.get('target', '')}`.")
    metrics_path = project.artifact_path("eval", "metrics.json")
    if metrics_path.exists():
        with _guard("metrics"):
            metrics = json.loads(metrics_path.read_text())
            scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            if scalar:
                cols = st.columns(min(4, len(scalar)))
                for i, (k, v) in enumerate(list(scalar.items())[:4]):
                    cols[i].metric(k, f"{v:.4f}" if isinstance(v, float) else v)
            with st.expander("All metrics", expanded=False):
                st.json(metrics)


# --- dispatch -----------------------------------------------------------------
_TRANSFORM_RENDERERS = {
    "clean": _render_clean,
    "feature_eng": _render_feature_eng,
    "feature_select": _render_feature_select,
    "data_prep": _render_data_prep,
    "model": _render_model,
    "evaluate": _render_evaluate,
}


def render_stage_result(
    project: Project,
    stage: str,
    result: StageResult,
    *,
    before: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    key: str = "",
) -> None:
    """Render one stage's outcome inline (called as each stage finishes)."""
    label = _STAGE_LABEL.get(stage, stage)
    key = key or f"sr_{stage}"

    head = f"### {'✅' if result.status == 'success' else '❌'} {label}"
    dur = _fmt_ms(duration_ms)
    st.markdown(head + (f"  ·  _{dur}_" if dur else ""))

    if result.status != "success":
        st.error(result.error or "Stage failed.")
        return

    if stage in _EDA_STAGES:
        with _guard("EDA dashboard"):
            render_eda_dashboard(project, project.get_working_data(), key=key)
    elif stage in _TRANSFORM_RENDERERS:
        _TRANSFORM_RENDERERS[stage](project, result, before, key)
    elif stage == "business":
        st.caption("Question framed — task & target set.")
    elif result.summary:
        st.json(result.summary)

    if result.artifacts:
        _render_artifacts(result.artifacts, key)
