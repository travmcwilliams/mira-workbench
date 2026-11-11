"""Interactive pre-analysis workbench: inspect data and configure pipeline stages."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import Pipeline
from core.preanalysis import (
    ENCODING_OPTIONS,
    FS_METHODS,
    MISSING_STRATEGIES,
    OUTLIER_ACTIONS,
    OUTLIER_METHODS,
    column_summary,
    correlation_preview,
    missing_matrix_summary,
    recommend_encoding,
    recommend_missing_strategy,
)
from core.project import Project
from core.stage_settings import get_stage_settings, save_stage_settings


def _pid(project: Project) -> str:
    return project.id[:8]


def _build_clean_editor_rows(
    df: pd.DataFrame,
    schema: dict[str, str],
    clean_settings: dict[str, Any],
    target: str | None,
) -> pd.DataFrame:
    summary = column_summary(df, schema)
    missing_map = clean_settings.get("missing_strategy", {})
    drop_set = set(clean_settings.get("drop_columns", []))

    rows = []
    for _, row in summary.iterrows():
        col = row["column"]
        if col == target:
            continue
        default = missing_map.get(col) or recommend_missing_strategy(
            row["schema_type"], row["missing_pct"]
        )
        rows.append({
            "column": col,
            "schema_type": row["schema_type"],
            "missing_pct": row["missing_pct"],
            "unique": row["unique"],
            "drop": col in drop_set,
            "missing_strategy": default,
        })
    return pd.DataFrame(rows)


def _build_fe_editor_rows(
    df: pd.DataFrame,
    schema: dict[str, str],
    fe_settings: dict[str, Any],
    target: str | None,
) -> pd.DataFrame:
    encoding = fe_settings.get("encoding", {})
    date_features = set(fe_settings.get("date_features", []))
    log_transform = set(fe_settings.get("log_transform", []))
    drop_set = set(fe_settings.get("drop_columns", []))

    rows = []
    for col in df.columns:
        if col == target:
            continue
        stype = schema.get(col, "text")
        nunique = int(df[col].nunique(dropna=True))
        rows.append({
            "column": col,
            "schema_type": stype,
            "unique": nunique,
            "drop": col in drop_set,
            "encoding": encoding.get(col, recommend_encoding(stype, nunique)),
            "extract_dates": col in date_features,
            "log_transform": col in log_transform,
        })
    return pd.DataFrame(rows)


def _apply_clean_from_editor(edited: pd.DataFrame, clean_settings: dict[str, Any]) -> dict[str, Any]:
    drop_cols = edited.loc[edited["drop"] == True, "column"].tolist()  # noqa: E712
    missing_strategy = {}
    for _, row in edited.iterrows():
        if row["drop"]:
            continue
        strat = row["missing_strategy"]
        if strat and strat != "none":
            missing_strategy[row["column"]] = strat
    clean_settings["drop_columns"] = drop_cols
    clean_settings["missing_strategy"] = missing_strategy
    return clean_settings


def _apply_fe_from_editor(edited: pd.DataFrame, fe_settings: dict[str, Any]) -> dict[str, Any]:
    drop_cols = edited.loc[edited["drop"] == True, "column"].tolist()  # noqa: E712
    encoding: dict[str, str] = {}
    date_features: list[str] = []
    log_transform: list[str] = []
    for _, row in edited.iterrows():
        col = row["column"]
        if row["drop"]:
            continue
        enc = row.get("encoding", "none")
        if enc and enc != "none" and row["schema_type"] == "categorical":
            encoding[col] = enc
        if row.get("extract_dates") and row["schema_type"] == "datetime":
            date_features.append(col)
        if row.get("log_transform") and row["schema_type"] == "numeric":
            log_transform.append(col)
    fe_settings["drop_columns"] = drop_cols
    fe_settings["encoding"] = encoding
    fe_settings["date_features"] = date_features
    fe_settings["log_transform"] = log_transform
    return fe_settings


def render_preanalysis_workbench(project: Project) -> None:
    """Main pre-analysis UI: inspect columns and configure clean / FE / selection."""
    df = project.get_working_data()
    schema = project.meta.get("schema", {})
    target = project.meta.get("target_column")
    key = _pid(project)

    st.header("Pre-Analysis Workbench")
    st.caption(
        "Inspect each column, choose imputation (mean, median, mode), drop columns, "
        "and configure feature engineering before running pipeline stages."
    )

    miss = missing_matrix_summary(df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", len(df.columns))
    m3.metric("Missing cells", f"{miss['missing_cell_pct']}%")
    m4.metric("Duplicate rows", f"{miss['duplicate_rows']:,}")

    tab_overview, tab_clean, tab_fe, tab_fs, tab_run = st.tabs(
        ["Overview", "Clean", "Feature Engineering", "Feature Selection", "Run Stages"]
    )

    clean_settings = get_stage_settings(project, "clean")
    fe_settings = get_stage_settings(project, "feature_eng")
    fs_settings = get_stage_settings(project, "feature_select")

    with tab_overview:
        st.subheader("Column statistics")
        summary = column_summary(df, schema)
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "missing_pct": st.column_config.NumberColumn(format="%.1f%%"),
                "sample": st.column_config.TextColumn(width="medium"),
            },
        )

        if target:
            st.caption(f"Target column: **{target}** (protected from drop)")

        corr = correlation_preview(df, target=target)
        if not corr.empty:
            st.subheader("Numeric correlations")
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Correlation matrix (numeric columns)",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True, key=f"corr_{key}")

        with st.expander("Missing value breakdown"):
            miss_cols = summary[summary["missing"] > 0][["column", "missing", "missing_pct"]]
            if miss_cols.empty:
                st.success("No missing values detected.")
            else:
                st.dataframe(miss_cols, use_container_width=True, hide_index=True)

    with tab_clean:
        st.subheader("Cleaning decisions")
        st.caption("Per-column: drop, or impute missing values with mean / median / mode.")

        clean_rows = _build_clean_editor_rows(df, schema, clean_settings, target)
        if clean_rows.empty:
            st.info("No columns to configure.")
        else:
            edited_clean = st.data_editor(
                clean_rows,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "column": st.column_config.TextColumn(disabled=True),
                    "schema_type": st.column_config.TextColumn(disabled=True),
                    "missing_pct": st.column_config.NumberColumn(disabled=True, format="%.1f"),
                    "unique": st.column_config.NumberColumn(disabled=True),
                    "drop": st.column_config.CheckboxColumn("Drop column"),
                    "missing_strategy": st.column_config.SelectboxColumn(
                        "Impute missing",
                        options=list(MISSING_STRATEGIES),
                    ),
                },
                key=f"clean_editor_{key}",
            )

            o1, o2, o3 = st.columns(3)
            with o1:
                clean_settings["outlier_method"] = st.selectbox(
                    "Outlier method",
                    OUTLIER_METHODS,
                    index=list(OUTLIER_METHODS).index(
                        clean_settings.get("outlier_method", "iqr")
                    )
                    if clean_settings.get("outlier_method", "iqr") in OUTLIER_METHODS
                    else 1,
                    key=f"outlier_m_{key}",
                )
            with o2:
                clean_settings["outlier_action"] = st.selectbox(
                    "Outlier action",
                    OUTLIER_ACTIONS,
                    index=list(OUTLIER_ACTIONS).index(
                        clean_settings.get("outlier_action", "clip")
                    ),
                    key=f"outlier_a_{key}",
                )
            with o3:
                clean_settings["outlier_threshold"] = st.number_input(
                    "Outlier threshold",
                    min_value=0.5,
                    max_value=5.0,
                    value=float(clean_settings.get("outlier_threshold", 1.5)),
                    step=0.1,
                    key=f"outlier_t_{key}",
                )

            clean_settings["drop_duplicates"] = st.checkbox(
                "Drop duplicate rows",
                value=bool(clean_settings.get("drop_duplicates", True)),
                key=f"dup_{key}",
            )
            clean_settings["constant_value"] = st.text_input(
                "Constant fill value (for 'constant' strategy)",
                value=str(clean_settings.get("constant_value", "Unknown")),
                key=f"const_{key}",
            )

            if st.button("Save cleaning settings", key=f"save_clean_{key}"):
                clean_settings = _apply_clean_from_editor(edited_clean, clean_settings)
                save_stage_settings(project, "clean", clean_settings)
                fe_settings["drop_columns"] = clean_settings.get("drop_columns", [])
                save_stage_settings(project, "feature_eng", fe_settings)
                st.success("Cleaning settings saved.")
                st.rerun()

    with tab_fe:
        st.subheader("Feature engineering decisions")
        st.caption("Drop ID/text columns, encode categoricals, extract date parts, log-transform skewed numerics.")

        fe_rows = _build_fe_editor_rows(df, schema, fe_settings, target)
        if fe_rows.empty:
            st.info("No columns to configure.")
        else:
            edited_fe = st.data_editor(
                fe_rows,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "column": st.column_config.TextColumn(disabled=True),
                    "schema_type": st.column_config.TextColumn(disabled=True),
                    "unique": st.column_config.NumberColumn(disabled=True),
                    "drop": st.column_config.CheckboxColumn("Drop column"),
                    "encoding": st.column_config.SelectboxColumn(
                        "Encoding",
                        options=list(ENCODING_OPTIONS),
                    ),
                    "extract_dates": st.column_config.CheckboxColumn("Date parts"),
                    "log_transform": st.column_config.CheckboxColumn("Log transform"),
                },
                key=f"fe_editor_{key}",
            )

            fe_settings["restaurant_labor"] = st.checkbox(
                "Restaurant labor features (if applicable)",
                value=bool(fe_settings.get("restaurant_labor", False)),
                key=f"rest_labor_{key}",
            )
            fe_settings["restaurant_peer_benchmarks"] = st.checkbox(
                "Peer benchmarks",
                value=bool(fe_settings.get("restaurant_peer_benchmarks", True)),
                key=f"rest_peer_{key}",
            )

            if st.button("Save feature engineering settings", key=f"save_fe_{key}"):
                fe_settings = _apply_fe_from_editor(edited_fe, fe_settings)
                save_stage_settings(project, "feature_eng", fe_settings)
                clean_settings["drop_columns"] = fe_settings.get("drop_columns", [])
                save_stage_settings(project, "clean", clean_settings)
                st.success("Feature engineering settings saved.")
                st.rerun()

    with tab_fs:
        st.subheader("Feature selection")
        methods = st.multiselect(
            "Selection methods",
            options=list(FS_METHODS),
            default=[m for m in fs_settings.get("methods", ["correlation", "variance"]) if m in FS_METHODS],
            key=f"fs_methods_{key}",
        )
        fs_settings["methods"] = methods

        c1, c2 = st.columns(2)
        with c1:
            fs_settings["correlation_threshold"] = st.slider(
                "Correlation threshold",
                0.5, 0.99,
                float(fs_settings.get("correlation_threshold", 0.9)),
                key=f"fs_corr_{key}",
            )
            fs_settings["variance_threshold"] = st.slider(
                "Variance threshold",
                0.0, 0.1,
                float(fs_settings.get("variance_threshold", 0.01)),
                step=0.001,
                format="%.3f",
                key=f"fs_var_{key}",
            )
        with c2:
            fs_settings["rfe_n_features"] = st.number_input(
                "RFE target features",
                min_value=2, max_value=100,
                value=int(fs_settings.get("rfe_n_features", 10)),
                key=f"fs_rfe_{key}",
            )
            fs_settings["importance_top_k"] = st.number_input(
                "Importance top-K",
                min_value=2, max_value=100,
                value=int(fs_settings.get("importance_top_k", 20)),
                key=f"fs_imp_{key}",
            )

        if st.button("Save feature selection settings", key=f"save_fs_{key}"):
            save_stage_settings(project, "feature_select", fs_settings)
            st.success("Feature selection settings saved.")

    with tab_run:
        st.subheader("Run with your settings")
        st.caption("Settings from the tabs above are applied when you run each stage.")

        saved_clean = get_stage_settings(project, "clean")
        saved_fe = get_stage_settings(project, "feature_eng")
        saved_fs = get_stage_settings(project, "feature_select")

        with st.expander("Current saved settings", expanded=False):
            st.json({
                "clean": {
                    "drop_columns": saved_clean.get("drop_columns", []),
                    "missing_strategy": saved_clean.get("missing_strategy", {}),
                    "outlier_method": saved_clean.get("outlier_method"),
                },
                "feature_eng": {
                    "drop_columns": saved_fe.get("drop_columns", []),
                    "encoding": saved_fe.get("encoding", {}),
                    "date_features": saved_fe.get("date_features", []),
                    "log_transform": saved_fe.get("log_transform", []),
                },
                "feature_select": saved_fs,
            })

        r1, r2, r3 = st.columns(3)
        pipe = Pipeline(project)

        with r1:
            if st.button("Run Clean", type="primary", use_container_width=True, key=f"run_clean_{key}"):
                with st.spinner("Cleaning..."):
                    res = pipe.run_stage("clean", saved_clean)
                    if res.status == "success":
                        st.success(f"Done: {res.summary}")
                        st.rerun()
                    else:
                        st.error(res.error)

        with r2:
            if st.button("Run Feature Eng", type="primary", use_container_width=True, key=f"run_fe_{key}"):
                with st.spinner("Feature engineering..."):
                    res = pipe.run_stage("feature_eng", saved_fe)
                    if res.status == "success":
                        st.success(f"Done: {res.summary}")
                        st.rerun()
                    else:
                        st.error(res.error)

        with r3:
            if st.button("Run Feature Select", type="primary", use_container_width=True, key=f"run_fs_{key}"):
                with st.spinner("Selecting features..."):
                    res = pipe.run_stage("feature_select", saved_fs)
                    if res.status == "success":
                        st.success(f"Kept {res.summary.get('selected_features')} features")
                        st.rerun()
                    else:
                        st.error(res.error)

        if st.button(
            "Run all three (Clean → FE → Select)",
            use_container_width=True,
            key=f"run_all_pre_{key}",
        ):
            with st.spinner("Running pre-analysis pipeline..."):
                for stage, settings in [
                    ("clean", saved_clean),
                    ("feature_eng", saved_fe),
                    ("feature_select", saved_fs),
                ]:
                    res = pipe.run_stage(stage, settings)
                    if res.status != "success":
                        st.error(f"{stage} failed: {res.error}")
                        break
                else:
                    st.success("Clean → Feature Eng → Feature Select complete.")
                    st.rerun()

        st.divider()
        st.subheader("Preview after stages")
        preview = project.get_working_data()
        st.caption(f"Working data: {len(preview):,} rows × {len(preview.columns)} columns")
        st.dataframe(preview.head(15), use_container_width=True, height=280)