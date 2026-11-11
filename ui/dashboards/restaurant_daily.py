"""Interactive Plotly dashboard for daily restaurant operations data."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.theme import PLOTLY_TEMPLATE
from plotly.subplots import make_subplots


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    normalized = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def prepare_restaurant_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_col = _resolve_col(out, "trading_day_date", "date", "business_date")
    if date_col:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col])
        out = out.sort_values(date_col)

    dow_col = _resolve_col(out, "day_of_week")
    if dow_col:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        out[dow_col] = pd.Categorical(out[dow_col], categories=order, ordered=True)

    sales_col = _resolve_col(out, "net_sales")
    if sales_col:
        out[sales_col] = pd.to_numeric(out[sales_col], errors="coerce").fillna(0)

    for col_name in ("labor_pct", "tips", "labor_hours", "wages", "sales_per_labor_hour", "guests"):
        col = _resolve_col(out, col_name)
        if col:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _kpi(col: pd.Series, fmt: str) -> str:
    if col is None or col.empty:
        return "—"
    val = col.mean()
    if pd.isna(val):
        return "—"
    return fmt.format(val)


def render_restaurant_daily_dashboard(df: pd.DataFrame, *, key_prefix: str = "rd") -> None:
    prepared = prepare_restaurant_daily_df(df)
    date_col = _resolve_col(prepared, "trading_day_date", "date")
    dow_col = _resolve_col(prepared, "day_of_week")
    sales_col = _resolve_col(prepared, "net_sales")
    tips_col = _resolve_col(prepared, "tips")
    labor_pct_col = _resolve_col(prepared, "labor_pct")
    labor_hours_col = _resolve_col(prepared, "labor_hours")
    splh_col = _resolve_col(prepared, "sales_per_labor_hour")
    guests_col = _resolve_col(prepared, "guests")

    st.markdown("##### Filters")
    f1, f2, f3 = st.columns(3)

    working = prepared.copy()
    with f1:
        min_sales = st.number_input(
            "Min net sales",
            min_value=0.0,
            value=0.0,
            step=100.0,
            key=f"{key_prefix}_min_sales",
        )
    if sales_col:
        working = working[working[sales_col] >= min_sales]

    with f2:
        if dow_col:
            days = sorted(working[dow_col].dropna().unique().tolist(), key=str)
            selected_days = st.multiselect(
                "Day of week",
                options=days,
                default=days,
                key=f"{key_prefix}_dow",
            )
            if selected_days:
                working = working[working[dow_col].isin(selected_days)]

    with f3:
        if date_col and not working.empty:
            min_d = working[date_col].min().date()
            max_d = working[date_col].max().date()
            start, end = st.date_input(
                "Date range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
                key=f"{key_prefix}_dates",
            )
            if isinstance(start, tuple):
                start, end = start
            working = working[
                (working[date_col].dt.date >= start) & (working[date_col].dt.date <= end)
            ]

    if working.empty:
        st.warning("No rows match the current filters.")
        return

    st.caption(f"Showing **{len(working):,}** days after filters")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Avg labor %", _kpi(working[labor_pct_col] if labor_pct_col else None, "{:.1%}"))
    with k2:
        st.metric("Avg tips / day", _kpi(working[tips_col] if tips_col else None, "${:,.0f}"))
    with k3:
        st.metric("Avg net sales / day", _kpi(working[sales_col] if sales_col else None, "${:,.0f}"))
    with k4:
        st.metric(
            "Avg sales / labor hr",
            _kpi(working[splh_col] if splh_col else None, "${:,.1f}"),
        )

    # Labor % by day of week
    if dow_col and labor_pct_col:
        dow_stats = (
            working.groupby(dow_col, observed=True)[labor_pct_col]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        fig_dow = px.bar(
            dow_stats,
            x=dow_col,
            y="mean",
            error_y=dow_stats["mean"] - dow_stats["median"],
            labels={"mean": "Mean labor %", dow_col: "Day"},
            title="Labor cost % by day of week",
            hover_data=["median", "count"],
            template=PLOTLY_TEMPLATE,
        )
        fig_dow.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_dow, use_container_width=True, key=f"{key_prefix}_dow_chart")

    # Time series: tips + sales
    if date_col and (tips_col or sales_col):
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        if sales_col:
            fig_ts.add_trace(
                go.Scatter(
                    x=working[date_col],
                    y=working[sales_col],
                    name="Net sales",
                    mode="lines",
                    hovertemplate="%{x|%Y-%m-%d}<br>Sales: $%{y:,.0f}<extra></extra>",
                ),
                secondary_y=False,
            )
        if tips_col:
            fig_ts.add_trace(
                go.Scatter(
                    x=working[date_col],
                    y=working[tips_col],
                    name="Tips",
                    mode="lines",
                    line={"dash": "dot"},
                    hovertemplate="%{x|%Y-%m-%d}<br>Tips: $%{y:,.0f}<extra></extra>",
                ),
                secondary_y=True,
            )
        fig_ts.update_layout(
            title="Tips & net sales trends",
            template=PLOTLY_TEMPLATE,
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
        fig_ts.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_ts, use_container_width=True, key=f"{key_prefix}_tips_sales")

    # Sales per labor hour trend
    if date_col and splh_col:
        fig_splh = px.line(
            working,
            x=date_col,
            y=splh_col,
            title="Sales per labor hour over time",
            labels={splh_col: "Sales / labor hr", date_col: "Date"},
            template=PLOTLY_TEMPLATE,
        )
        fig_splh.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.1f}/hr<extra></extra>")
        fig_splh.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_splh, use_container_width=True, key=f"{key_prefix}_splh")

    # Labor % trend
    if date_col and labor_pct_col:
        fig_lp = px.line(
            working,
            x=date_col,
            y=labor_pct_col,
            color=dow_col if dow_col else None,
            title="Labor % over time (colored by day of week)",
            labels={labor_pct_col: "Labor %", date_col: "Date"},
            template=PLOTLY_TEMPLATE,
        )
        fig_lp.update_layout(yaxis_tickformat=".0%")
        fig_lp.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_lp, use_container_width=True, key=f"{key_prefix}_labor_pct")

    # Productivity scatter
    if labor_hours_col and sales_col:
        fig_sc = px.scatter(
            working,
            x=labor_hours_col,
            y=sales_col,
            color=labor_pct_col if labor_pct_col else dow_col,
            size=guests_col if guests_col else None,
            hover_data=[date_col, tips_col, splh_col] if date_col else [tips_col],
            title="Labor hours vs net sales",
            labels={labor_hours_col: "Labor hours", sales_col: "Net sales"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_sc, use_container_width=True, key=f"{key_prefix}_scatter")