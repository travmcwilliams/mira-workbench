"""
stages/eda.py

EDA stage: autoviz if enabled, otherwise manual plots (hist, bar, corr, pair).
Saves PNGs to artifacts/eda/.
See grok_handoff.md Section 6.6.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core.library_catalog import get_enabled_libraries
from core.pipeline import StageResult
from core.project import Project


def default_settings(project: Project) -> dict[str, Any]:
    return {
        "method": "auto",
        "max_numeric_plots": 3,
        "max_categorical_plots": 3,
        "include_pairplot": True,
    }


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    if df is None or len(df) == 0:
        return StageResult(status="error", error="No data for EDA.")

    eda_dir = project.artifact_path("eda")
    eda_dir.mkdir(parents=True, exist_ok=True)

    enabled = get_enabled_libraries(project)
    artifacts: list[Path] = []
    summary: dict[str, Any] = {"n_plots": 0}

    target = project.meta.get("target_column")

    method = settings.get("method", "auto")
    use_autoviz = method == "autoviz" or (method == "auto" and "autoviz" in enabled)

    if use_autoviz:
        try:
            from autoviz.AutoViz_Class import AutoViz_Class

            AV = AutoViz_Class()
            # Autoviz writes to current dir by default; we chdir to eda_dir
            orig_cwd = os.getcwd()
            os.chdir(eda_dir)
            try:
                AV.AutoViz(
                    filename="",
                    dfte=df,
                    depVar=target or "",
                    verbose=0,
                    chart_format="png",
                )
            finally:
                os.chdir(orig_cwd)
            # Collect generated pngs
            pngs = list(eda_dir.glob("*.png"))
            artifacts.extend(pngs)
            summary["n_plots"] = len(pngs)
            summary["method"] = "autoviz"
        except Exception as e:
            summary["autoviz_error"] = str(e)
            # fall through to manual

    if not artifacts:
        # Manual EDA
        sns.set_style("whitegrid")
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()[:10]  # limit

        max_num = int(settings.get("max_numeric_plots", 3))
        max_cat = int(settings.get("max_categorical_plots", 3))

        # Histograms for numeric
        if num_cols:
            n_plot = min(max_num, len(num_cols))
            fig, axes = plt.subplots(n_plot, 1, figsize=(8, 3 * n_plot))
            if n_plot == 1:
                axes = [axes]
            for ax, col in zip(axes, num_cols[:n_plot], strict=False):
                sns.histplot(df[col].dropna(), ax=ax, kde=True)
                ax.set_title(f"Distribution of {col}")
            hist_path = eda_dir / "numeric_distributions.png"
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()
            artifacts.append(hist_path)

        # Bar charts for top categoricals
        if cat_cols:
            for col in cat_cols[:max_cat]:
                fig, ax = plt.subplots(figsize=(8, 4))
                df[col].value_counts().head(10).plot(kind="bar", ax=ax)
                ax.set_title(f"Top values for {col}")
                bar_path = eda_dir / f"bar_{col}.png"
                plt.tight_layout()
                plt.savefig(bar_path)
                plt.close()
                artifacts.append(bar_path)

        # Correlation heatmap (numeric only)
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            corr_path = eda_dir / "correlation_heatmap.png"
            plt.tight_layout()
            plt.savefig(corr_path)
            plt.close()
            artifacts.append(corr_path)

        # Simple pair plot of top correlated with target (if target numeric)
        if settings.get("include_pairplot", True) and target and target in num_cols and len(num_cols) > 1:
            # Exclude the target itself explicitly: a feature perfectly collinear with
            # the target ties at corr 1.0 and can sort ahead of it, which would leave a
            # duplicate target column in pair_df and crash seaborn.
            ranked = df[num_cols].corr()[target].abs().sort_values(ascending=False).index.tolist()
            top_corr = [c for c in ranked if c != target][:4]
            if top_corr:
                pair_df = df[top_corr + [target]].dropna()
                if len(pair_df) > 10:
                    g = sns.pairplot(pair_df, hue=None, corner=True)
                    g.fig.suptitle("Pair plot (top features vs target)", y=1.02)
                    pair_path = eda_dir / "pairplot.png"
                    g.savefig(pair_path)
                    plt.close()
                    artifacts.append(pair_path)

        summary["n_plots"] = len(artifacts)
        summary["method"] = "manual"

    return StageResult(
        status="success",
        output_df=df,
        artifacts=artifacts,
        summary=summary,
    )


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    import streamlit as st

    settings["method"] = st.selectbox(
        "EDA method",
        ["auto", "autoviz", "manual"],
        index=["auto", "autoviz", "manual"].index(settings.get("method", "auto")),
    )
    c1, c2 = st.columns(2)
    with c1:
        settings["max_numeric_plots"] = st.number_input(
            "Max numeric plots", 1, 10, int(settings.get("max_numeric_plots", 3))
        )
    with c2:
        settings["max_categorical_plots"] = st.number_input(
            "Max categorical plots", 1, 10, int(settings.get("max_categorical_plots", 3))
        )
    settings["include_pairplot"] = st.checkbox(
        "Include pair plot vs target",
        value=bool(settings.get("include_pairplot", True)),
    )
    return settings


def code_template(settings: dict[str, Any]) -> str:
    return """# EDA
# Manual or AutoViz plots generated.
# See artifacts/eda/ for PNGs.
print("EDA plots created.")
"""
