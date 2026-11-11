"""Plotly figure builders for server performance charts."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.server_analytics import cohort_distribution_frame

_METRIC_OPTIONS = {
    "tips_per_hour": "Tips per hour",
    "total_tips": "Total tips",
    "sales_per_hour": "Sales per hour",
    "tips_pct_sales": "Tips % of sales",
}


def build_server_figures(
    servers: pd.DataFrame,
    *,
    rank_metric: str = "tips_per_hour",
    cohort_median: float | None = None,
) -> dict[str, go.Figure]:
    """Return named Plotly figures for the server performance dashboard."""
    figures: dict[str, go.Figure] = {}
    if servers.empty:
        return figures

    med = cohort_median
    if med is None and rank_metric in servers.columns:
        med = float(servers[rank_metric].median())

    long_df = cohort_distribution_frame(servers)

    if not long_df.empty:
        fig_box = px.box(
            long_df,
            x="metric",
            y="value",
            points="all",
            hover_data=["display_name", "total_hours"],
            title="Server cohort — box & whisker with individual servers",
            labels={"metric": "Metric", "value": "Value"},
        )
        fig_box.update_traces(marker=dict(size=8, opacity=0.75), jitter=0.35, pointpos=0)
        figures["01_box_all_metrics"] = fig_box

        fig_violin = px.violin(
            long_df,
            x="metric",
            y="value",
            box=True,
            points="all",
            hover_data=["display_name"],
            title="Distribution shape (violin + embedded box)",
        )
        figures["02_violin_all_metrics"] = fig_violin

    if rank_metric in servers.columns and med is not None:
        fig_one = go.Figure()
        vals = servers[rank_metric].dropna()
        fig_one.add_trace(
            go.Box(
                y=vals,
                name="All servers",
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                marker_color="#2E86AB",
                line_color="#1a5276",
            )
        )
        fig_one.add_hline(
            y=med,
            line_dash="dash",
            line_color="#F18F01",
            annotation_text="Median",
            annotation_position="right",
        )
        fig_one.update_layout(
            title=f"{_METRIC_OPTIONS.get(rank_metric, rank_metric)} — whisker plot",
            yaxis_title=_METRIC_OPTIONS.get(rank_metric, rank_metric),
            showlegend=False,
            height=420,
        )
        figures[f"03_box_{rank_metric}"] = fig_one

    if "tips_per_hour" in servers.columns:
        fig_hist = px.histogram(
            servers,
            x="tips_per_hour",
            nbins=min(12, max(4, len(servers) // 2)),
            marginal="box",
            hover_data=["display_name", "total_hours", "total_tips"],
            title="Tips per hour — histogram with marginal whisker",
        )
        fig_hist.add_vline(
            x=servers["tips_per_hour"].median(),
            line_dash="dash",
            line_color="#F18F01",
            annotation_text="Median",
        )
        figures["04_histogram_tips_per_hour"] = fig_hist

    pct_col = f"{rank_metric}_pct_above_median"
    if pct_col in servers.columns:
        delta_df = servers.sort_values(pct_col, ascending=True).copy()
        colors = ["#2E86AB" if v >= 0 else "#A23B72" for v in delta_df[pct_col]]
        fig_delta = go.Figure(
            go.Bar(
                x=delta_df[pct_col],
                y=delta_df["display_name"],
                orientation="h",
                marker_color=colors,
                hovertemplate=(
                    "%{y}<br>% vs median: %{x:.1f}%<br>"
                    + f"{_METRIC_OPTIONS.get(rank_metric, rank_metric)}: "
                    + "%{customdata:.2f}<extra></extra>"
                ),
                customdata=delta_df[rank_metric],
            )
        )
        fig_delta.add_vline(x=0, line_color="gray", line_width=1)
        fig_delta.update_layout(
            title=f"How much better than server median — {_METRIC_OPTIONS.get(rank_metric, rank_metric)}",
            xaxis_title="% above (+) or below (−) cohort median",
            yaxis_title="Server",
            height=max(360, len(delta_df) * 28),
        )
        figures["05_pct_vs_median"] = fig_delta

    if "total_hours" in servers.columns and "tips_per_hour" in servers.columns:
        fig_h = px.scatter(
            servers,
            x="total_hours",
            y="tips_per_hour",
            size="total_tips" if "total_tips" in servers.columns else None,
            color=pct_col if pct_col in servers.columns else None,
            hover_name="display_name",
            hover_data=["total_tips", "net_sales", "sales_per_hour"],
            title="Hours worked vs tips per hour",
            color_continuous_scale="RdYlGn",
        )
        fig_h.add_hline(
            y=servers["tips_per_hour"].median(),
            line_dash="dash",
            line_color="#F18F01",
            annotation_text="Median tips/hr",
        )
        figures["06_scatter_hours_vs_tips_hr"] = fig_h

    if "sales_per_hour" in servers.columns and "tips_per_hour" in servers.columns:
        fig_st = px.scatter(
            servers,
            x="sales_per_hour",
            y="tips_per_hour",
            text="display_name",
            title="Sales/hour vs tips/hour (server efficiency map)",
        )
        fig_st.update_traces(textposition="top center", marker=dict(size=12))
        fig_st.add_hline(y=servers["tips_per_hour"].median(), line_dash="dot", line_color="#ccc")
        fig_st.add_vline(x=servers["sales_per_hour"].median(), line_dash="dot", line_color="#ccc")
        figures["07_scatter_sales_vs_tips_hr"] = fig_st

    metrics_to_plot = [
        m for m in ("tips_per_hour", "sales_per_hour", "tips_pct_sales") if m in servers.columns
    ]
    if len(metrics_to_plot) >= 2:
        fig_grid = make_subplots(
            rows=1,
            cols=len(metrics_to_plot),
            subplot_titles=[_METRIC_OPTIONS.get(m, m) for m in metrics_to_plot],
        )
        for i, m in enumerate(metrics_to_plot, start=1):
            ordered = servers.sort_values(m, ascending=True)
            fig_grid.add_trace(
                go.Box(
                    y=ordered[m],
                    name=_METRIC_OPTIONS.get(m, m),
                    boxpoints="outliers",
                    marker_color="#2E86AB",
                ),
                row=1,
                col=i,
            )
        fig_grid.update_layout(
            title="Related server metrics — side-by-side whiskers",
            showlegend=False,
            height=400,
        )
        figures["08_whisker_grid"] = fig_grid

    return figures