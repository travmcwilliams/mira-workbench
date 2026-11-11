"""Dashboard for canonical MIRA records-shaped datasets."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.theme import PLOTLY_TEMPLATE

from core.dataset_profiles import normalize_column


def _col(df: pd.DataFrame, name: str) -> str | None:
    target = normalize_column(name)
    for c in df.columns:
        if normalize_column(c) == target:
            return c
    return None


def render_canonical_records_dashboard(df: pd.DataFrame, *, key_prefix: str = "rec") -> None:
    entity = _col(df, "entity_name")
    value = _col(df, "primary_value")
    period = _col(df, "period")
    source = _col(df, "source_id")
    location = _col(df, "location_text")

    if not entity or not value:
        st.info("Records dashboard needs entity_name and primary_value columns.")
        return

    work = df.copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work = work.dropna(subset=[value])

    if source:
        st.markdown("##### Records by source")
        by_src = work.groupby(source, dropna=False)[value].agg(["count", "sum", "mean"]).reset_index()
        by_src.columns = [source, "count", "total", "mean"]
        fig_src = px.bar(by_src, x=source, y="count", title="Row count by source", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_src, use_container_width=True, key=f"{key_prefix}_by_src")

    top_n = st.slider("Top entities", 5, 30, 15, key=f"{key_prefix}_topn")
    top = (
        work.groupby(entity, dropna=False)[value]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig_top = px.bar(
        top,
        x=value,
        y=entity,
        orientation="h",
        title=f"Top {top_n} entities by {value}",
        template=PLOTLY_TEMPLATE,
    )
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top, use_container_width=True, key=f"{key_prefix}_top")

    if period:
        st.markdown("##### Value over time")
        ts = work.copy()
        ts[period] = pd.to_datetime(ts[period], errors="coerce")
        ts = ts.dropna(subset=[period])
        if not ts.empty:
            agg = ts.groupby(period)[value].sum().reset_index()
            fig_ts = px.line(agg, x=period, y=value, title=f"Total {value} by {period}", template=PLOTLY_TEMPLATE)
            fig_ts.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_ts, use_container_width=True, key=f"{key_prefix}_ts")

    if location:
        st.markdown("##### By location")
        loc_agg = (
            work.groupby(location, dropna=False)[value]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        fig_loc = px.bar(loc_agg, x=location, y=value, title=f"{value} by {location}", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_loc, use_container_width=True, key=f"{key_prefix}_loc")