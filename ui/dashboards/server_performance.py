"""Server leaderboard — box plots, violins, and peer delta charts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.chart_export import export_server_chart_pack
from core.server_analytics import (
    prepare_server_cohort,
    server_leaderboard,
    top_vs_bottom_summary,
)
from core.server_charts import build_server_figures

_METRIC_OPTIONS = {
    "tips_per_hour": "Tips per hour",
    "total_tips": "Total tips",
    "sales_per_hour": "Sales per hour",
    "tips_pct_sales": "Tips % of sales",
}


def _fmt_pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.0f}%"


def _fmt_money(val: float) -> str:
    return f"${val:,.2f}"


def render_server_performance_dashboard(
    df: pd.DataFrame,
    *,
    key_prefix: str = "srv",
    reports_dir: str | None = None,
) -> None:
    st.markdown("##### Server performance")
    st.caption(
        "Pure **Server** role rows only — compares tip productivity, sales rate, and "
        "how far above or below the server cohort median each person sits."
    )

    min_hours = st.slider(
        "Min server hours (exclude short stints)",
        min_value=0.0,
        max_value=200.0,
        value=20.0,
        step=5.0,
        key=f"{key_prefix}_min_srv_hours",
    )
    servers, meta = prepare_server_cohort(df, min_hours=min_hours)

    if servers.empty:
        st.warning("No server rows match the current filters.")
        return

    rank_metric = st.selectbox(
        "Rank servers by",
        options=list(_METRIC_OPTIONS.keys()),
        format_func=lambda k: _METRIC_OPTIONS[k],
        key=f"{key_prefix}_rank_metric",
    )

    summary = top_vs_bottom_summary(servers, metric=rank_metric, n=3)
    med = summary.get("cohort_median", 0.0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Servers in cohort", f"{meta['server_count']:,}")
    with c2:
        st.metric(
            f"Median { _METRIC_OPTIONS[rank_metric]}",
            _fmt_money(med) if rank_metric != "tips_pct_sales" else f"{med:.1%}",
        )
    with c3:
        st.metric(
            "Top 3 avg vs median",
            _fmt_pct(summary.get("top_pct_above_median", 0)),
        )
    with c4:
        spread = summary.get("spread_top_vs_bottom", 0)
        if rank_metric == "tips_pct_sales":
            st.metric("Top vs bottom spread", f"{spread:.1%}")
        else:
            st.metric("Top vs bottom spread", _fmt_money(spread))

    # Download chart pack
    with st.expander("Download charts", expanded=True):
        st.caption(
            "Exports whisker plots, violins, scatter charts, and leaderboard CSV "
            "as a ZIP (HTML + PNG when available)."
        )
        if st.button("Build chart download pack", type="primary", key=f"{key_prefix}_build_zip"):
            out = reports_dir or "reports"
            with st.spinner("Rendering charts..."):
                result = export_server_chart_pack(
                    df,
                    out,
                    min_hours=min_hours,
                    rank_metric=rank_metric,
                    prefix="restaurant_servers",
                )
            st.session_state[f"{key_prefix}_chart_pack"] = result

        pack = st.session_state.get(f"{key_prefix}_chart_pack")
        if pack and not pack.get("error"):
            png_note = "PNG included" if pack.get("png_export") else "HTML only (install kaleido for PNG)"
            st.success(
                f"Ready: **{len(pack.get('figures', []))} charts**, "
                f"leaderboard CSV, combined HTML — {png_note}"
            )
            st.download_button(
                label="Download ZIP (all server charts)",
                data=pack["zip_bytes"],
                file_name=pack["zip_path"].name,
                mime="application/zip",
                use_container_width=True,
                key=f"{key_prefix}_dl_zip",
            )
            board = server_leaderboard(servers, metric=rank_metric)
            st.download_button(
                label="Download leaderboard CSV",
                data=board.to_csv(index=False).encode("utf-8"),
                file_name="server_leaderboard.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"{key_prefix}_dl_csv",
            )

    # Leaderboard table
    board = server_leaderboard(servers, metric=rank_metric)
    if not board.empty:
        show = board.copy()
        pct_col = f"{rank_metric}_pct_above_median"
        if pct_col in show.columns:
            show["vs median"] = show[pct_col].map(lambda v: _fmt_pct(v) if pd.notna(v) else "—")
        st.dataframe(
            show[
                [
                    c
                    for c in (
                        "display_name",
                        rank_metric,
                        "vs median" if "vs median" in show.columns else pct_col,
                        f"{rank_metric}_percentile",
                        "total_hours",
                        "total_tips",
                        "sales_per_hour",
                    )
                    if c in show.columns
                ]
            ].rename(
                columns={
                    "display_name": "Server",
                    rank_metric: _METRIC_OPTIONS[rank_metric],
                    f"{rank_metric}_percentile": "Percentile",
                    "total_hours": "Hours",
                    "total_tips": "Total tips",
                    "sales_per_hour": "Sales/hr",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    figures = build_server_figures(servers, rank_metric=rank_metric, cohort_median=med)

    tab_box, tab_shape, tab_delta, tab_scatter = st.tabs(
        ["Whisker plots", "Distribution shape", "Vs median", "Hours & productivity"]
    )

    with tab_box:
        if "01_box_all_metrics" in figures:
            st.plotly_chart(
                figures["01_box_all_metrics"],
                use_container_width=True,
                key=f"{key_prefix}_box_all",
            )
        key_one = f"03_box_{rank_metric}"
        if key_one in figures:
            st.plotly_chart(figures[key_one], use_container_width=True, key=f"{key_prefix}_box_one")

    with tab_shape:
        if "02_violin_all_metrics" in figures:
            st.plotly_chart(figures["02_violin_all_metrics"], use_container_width=True, key=f"{key_prefix}_violin")
        if "04_histogram_tips_per_hour" in figures:
            st.plotly_chart(figures["04_histogram_tips_per_hour"], use_container_width=True, key=f"{key_prefix}_hist")

    with tab_delta:
        if "05_pct_vs_median" in figures:
            st.plotly_chart(figures["05_pct_vs_median"], use_container_width=True, key=f"{key_prefix}_delta")
            pct_col = f"{rank_metric}_pct_above_median"
            top3 = servers.nlargest(3, rank_metric)
            st.markdown("**Top 3 servers**")
            for _, row in top3.iterrows():
                pct = row.get(pct_col, 0)
                val = row[rank_metric]
                if rank_metric == "tips_pct_sales":
                    val_s = f"{val:.1%}"
                else:
                    val_s = _fmt_money(val)
                st.write(
                    f"- **{row['display_name']}**: {val_s} "
                    f"({_fmt_pct(pct)} vs server median)"
                )

    with tab_scatter:
        if "06_scatter_hours_vs_tips_hr" in figures:
            st.plotly_chart(figures["06_scatter_hours_vs_tips_hr"], use_container_width=True, key=f"{key_prefix}_scatter_h")
        if "07_scatter_sales_vs_tips_hr" in figures:
            st.plotly_chart(figures["07_scatter_sales_vs_tips_hr"], use_container_width=True, key=f"{key_prefix}_scatter_st")
        if "08_whisker_grid" in figures:
            st.plotly_chart(figures["08_whisker_grid"], use_container_width=True, key=f"{key_prefix}_grid")