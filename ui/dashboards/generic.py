"""Generic interactive Plotly explorer for any tabular dataset."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.theme import PLOTLY_TEMPLATE

from core.data_loader import detect_schema


def _date_column(df: pd.DataFrame, schema: dict[str, str]) -> str | None:
    for col, dtype in schema.items():
        if col in df.columns and dtype == "datetime":
            return col
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return None


def render_generic_dashboard(df: pd.DataFrame, *, key_prefix: str = "gen") -> None:
    schema = detect_schema(df)
    numeric_cols = [c for c in df.columns if schema.get(c) == "numeric"]
    categorical_cols = [c for c in df.columns if schema.get(c) == "categorical"]
    date_col = _date_column(df, schema)

    if not numeric_cols:
        st.info("No numeric columns detected for interactive charts.")
        return

    st.markdown("##### Chart builder")

    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox(
            "X axis",
            options=df.columns.tolist(),
            index=df.columns.get_loc(date_col) if date_col and date_col in df.columns else 0,
            key=f"{key_prefix}_x",
        )
    with c2:
        default_y = numeric_cols[0]
        y_col = st.selectbox(
            "Y axis",
            options=numeric_cols,
            index=numeric_cols.index(default_y),
            key=f"{key_prefix}_y",
        )
    with c3:
        color_col = st.selectbox(
            "Color (optional)",
            options=["(none)"] + categorical_cols + numeric_cols,
            key=f"{key_prefix}_color",
        )
    color = None if color_col == "(none)" else color_col

    chart_type = st.radio(
        "Chart type",
        options=["Line", "Scatter", "Bar", "Histogram", "Box"],
        horizontal=True,
        key=f"{key_prefix}_type",
    )

    if chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col, color=color, title=f"{y_col} over {x_col}", template=PLOTLY_TEMPLATE)
        fig.update_xaxes(rangeslider_visible=True)
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=color, title=f"{y_col} vs {x_col}", template=PLOTLY_TEMPLATE)
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col, color=color, title=f"{y_col} by {x_col}", template=PLOTLY_TEMPLATE)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=y_col, color=color, title=f"Distribution of {y_col}", template=PLOTLY_TEMPLATE)
    else:
        fig = px.box(df, x=x_col if x_col in categorical_cols else color, y=y_col, color=color, template=PLOTLY_TEMPLATE)

    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_main")

    if len(numeric_cols) >= 2:
        st.markdown("##### Correlation heatmap")
        corr = df[numeric_cols].corr()
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmid=0,
                hovertemplate="%{x} vs %{y}<br>r=%{z:.2f}<extra></extra>",
            )
        )
        fig_hm.update_layout(title="Numeric correlations", template=PLOTLY_TEMPLATE, height=480)
        st.plotly_chart(fig_hm, use_container_width=True, key=f"{key_prefix}_corr")