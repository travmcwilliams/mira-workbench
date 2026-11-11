"""Interactive Plotly dashboard for employee/role-level labor exports."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.theme import PLOTLY_TEMPLATE

from core.restaurant_features import engineer_labor_features
from ui.dashboards.server_performance import render_server_performance_dashboard


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    normalized = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def prepare_employee_labor_df(df: pd.DataFrame, *, aggregate: bool = False) -> pd.DataFrame:
    engineered, _ = engineer_labor_features(
        df,
        target=None,
        add_peer_benchmarks=True,
        aggregate_employee=aggregate,
    )
    out = engineered.copy()

    if "employee_key" in out.columns:
        out["_employee"] = out["employee_key"]
    else:
        last_col = _resolve_col(out, "last_name")
        first_col = _resolve_col(out, "first_name")
        if last_col and first_col:
            out["_employee"] = (
                out[first_col].astype(str).str.strip() + " " + out[last_col].astype(str).str.strip()
            ).str.strip()
        elif last_col:
            out["_employee"] = out[last_col].astype(str).str.strip()

    return out


def render_employee_labor_dashboard(
    df: pd.DataFrame,
    *,
    key_prefix: str = "el",
    reports_dir: str | None = None,
) -> None:
    view_mode = st.radio(
        "View",
        options=["By role (row-level)", "By employee (aggregated)"],
        horizontal=True,
        key=f"{key_prefix}_view",
    )
    aggregate = view_mode.startswith("By employee")
    prepared = prepare_employee_labor_df(df, aggregate=aggregate)

    role_col = _resolve_col(prepared, "role") or (
        "primary_role" if "primary_role" in prepared.columns else None
    )
    tips_col = _resolve_col(prepared, "total_tips")
    hours_col = _resolve_col(prepared, "total_hours") or _resolve_col(prepared, "reg_hrs")
    sales_col = _resolve_col(prepared, "net_sales")
    pay_col = _resolve_col(prepared, "total_pay")

    st.markdown("##### Filters")
    working = prepared.copy()

    if role_col:
        roles = sorted(working[role_col].dropna().unique().tolist())
        selected_roles = st.multiselect(
            "Roles",
            options=roles,
            default=roles,
            key=f"{key_prefix}_roles",
        )
        if selected_roles:
            working = working[working[role_col].isin(selected_roles)]

    max_hours = float(working[hours_col].max()) if hours_col and len(working) else 100.0
    min_hours = st.slider(
        "Min hours",
        min_value=0.0,
        max_value=max_hours,
        value=0.0,
        step=1.0,
        key=f"{key_prefix}_min_hours",
    )
    if hours_col:
        working = working[working[hours_col] >= min_hours]

    if "is_multi_role" in working.columns and not aggregate:
        multi_only = st.checkbox("Multi-role employees only", key=f"{key_prefix}_multi")
        if multi_only:
            working = working[working["is_multi_role"] == 1]

    if working.empty:
        st.warning("No rows match the current filters.")
        return

    label = "employees" if aggregate else "employee-role rows"
    st.caption(f"Showing **{len(working):,}** {label}")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total tips", f"${working[tips_col].sum():,.0f}" if tips_col else "—")
    with k2:
        st.metric("Total hours", f"{working[hours_col].sum():,.0f}" if hours_col else "—")
    with k3:
        st.metric("Attributed sales", f"${working[sales_col].sum():,.0f}" if sales_col else "—")
    with k4:
        st.metric("Total pay", f"${working[pay_col].sum():,.0f}" if pay_col else "—")
    with k5:
        if "implied_shifts" in working.columns:
            st.metric("Implied shifts", f"{working['implied_shifts'].sum():,.0f}")
        elif aggregate and "n_roles" in working.columns:
            st.metric("Avg roles/person", f"{working['n_roles'].mean():.1f}")

    tab_ops, tab_peer, tab_mix, tab_servers = st.tabs(
        ["Ops", "Peer benchmarks", "Role mix", "Server rankings"]
    )

    with tab_ops:
        if role_col and tips_col:
            by_role = (
                working.groupby(role_col, observed=True)
                .agg({tips_col: "sum", hours_col: "sum"} if hours_col else {tips_col: "sum"})
                .reset_index()
            )
            fig_tips = px.bar(
                by_role.sort_values(tips_col, ascending=True),
                x=tips_col,
                y=role_col,
                orientation="h",
                title="Total tips by role",
                labels={tips_col: "Tips ($)", role_col: "Role"},
                template=PLOTLY_TEMPLATE,
            )
            fig_tips.update_traces(hovertemplate="%{y}<br>Tips: $%{x:,.0f}<extra></extra>")
            st.plotly_chart(fig_tips, use_container_width=True, key=f"{key_prefix}_tips_role")

        if sales_col and tips_col and hours_col:
            fig_sc = px.scatter(
                working,
                x=hours_col,
                y=tips_col,
                color=role_col if role_col else None,
                size=sales_col,
                hover_data=["_employee"] if "_employee" in working.columns else None,
                title="Hours vs tips (bubble size = net sales)",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_sc, use_container_width=True, key=f"{key_prefix}_scatter")

    with tab_peer:
        peer_col = (
            "tips_vs_store_median"
            if aggregate and "tips_vs_store_median" in working.columns
            else "tips_per_hour_vs_role_median"
            if "tips_per_hour_vs_role_median" in working.columns
            else None
        )
        if peer_col and "_employee" in working.columns and tips_col:
            top_peer = working.nlargest(20, tips_col)
            fig_peer = px.bar(
                top_peer.sort_values(peer_col),
                x=peer_col,
                y="_employee",
                color=role_col if role_col else None,
                orientation="h",
                title="Tip productivity vs peers (ratio to median)",
                labels={peer_col: "vs median", "_employee": "Employee"},
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_peer, use_container_width=True, key=f"{key_prefix}_peer")

        if "sales_per_hour_vs_role_median" in working.columns and role_col:
            fig_sph = px.box(
                working,
                x=role_col,
                y="sales_per_hour_vs_role_median",
                title="Sales/hour vs role median (distribution)",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_sph, use_container_width=True, key=f"{key_prefix}_sph_box")

    with tab_mix:
        if aggregate and "n_roles" in working.columns:
            mix = working[working["n_roles"] > 1].copy()
            if not mix.empty and tips_col:
                fig_multi = px.scatter(
                    mix,
                    x="n_roles",
                    y=tips_col,
                    size="total_hours" if "total_hours" in mix.columns else None,
                    color="primary_role_family" if "primary_role_family" in mix.columns else None,
                    hover_data=["_employee", "primary_role"]
                    if "_employee" in mix.columns
                    else None,
                    title="Multi-role staff: roles held vs total tips",
                    template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_multi, use_container_width=True, key=f"{key_prefix}_multi")

        share_cols = [
            c
            for c in working.columns
            if c.startswith("employee_") and c.endswith("_hours_share")
        ]
        if share_cols and "_employee" in working.columns and aggregate:
            melted = working[["_employee"] + share_cols].melt(
                id_vars="_employee", var_name="segment", value_name="share"
            )
            melted["segment"] = melted["segment"].str.replace("employee_", "").str.replace(
                "_hours_share", ""
            )
            top_emp = (
                working.nlargest(15, tips_col)[["_employee"]].drop_duplicates()
                if tips_col
                else working[["_employee"]].drop_duplicates().head(15)
            )
            melted = melted[melted["_employee"].isin(top_emp["_employee"])]
            fig_mix = px.bar(
                melted,
                x="share",
                y="_employee",
                color="segment",
                orientation="h",
                barmode="stack",
                title="Hours mix by segment (top earners)",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_mix, use_container_width=True, key=f"{key_prefix}_mix")

        if not aggregate and "role_hours_share" in working.columns and "_employee" in working.columns:
            multi_rows = working[working.get("is_multi_role", 0) == 1]
            if not multi_rows.empty:
                fig_share = px.bar(
                    multi_rows.sort_values("role_hours_share", ascending=True).tail(30),
                    x="role_hours_share",
                    y="_employee",
                    color=role_col if role_col else None,
                    orientation="h",
                    title="Role hours share (multi-role rows)",
                    template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_share, use_container_width=True, key=f"{key_prefix}_share")

    with tab_servers:
        render_server_performance_dashboard(
            df, key_prefix=f"{key_prefix}_srv", reports_dir=reports_dir
        )