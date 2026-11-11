"""Interactive dashboard router for Streamlit."""

from __future__ import annotations

import streamlit as st

from core.dataset_profiles import detect_dataset_profile, profile_label
from core.project import Project
from ui.dashboards.canonical_records import render_canonical_records_dashboard
from ui.dashboards.employee_labor import render_employee_labor_dashboard
from ui.dashboards.generic import render_generic_dashboard
from ui.dashboards.restaurant_daily import render_restaurant_daily_dashboard


def render_interactive_dashboard(project: Project) -> None:
    """Render profile-specific interactive charts with hover, zoom, and filters."""
    if not project.has_raw_data():
        return

    df = project.get_working_data()
    profile = detect_dataset_profile(df)

    st.header("Interactive Dashboard")
    st.caption(
        f"Profile: **{profile_label(profile)}** — hover for values, drag to zoom, "
        "use range sliders on time series, adjust filters above each chart group."
    )

    key_prefix = f"dash_{project.id[:8]}"

    if profile == "restaurant_daily":
        render_restaurant_daily_dashboard(df, key_prefix=key_prefix)
    elif profile == "employee_labor":
        reports_dir = str(project.root / "reports")
        render_employee_labor_dashboard(df, key_prefix=key_prefix, reports_dir=reports_dir)
    elif profile == "canonical_records":
        render_canonical_records_dashboard(df, key_prefix=key_prefix)
    else:
        render_generic_dashboard(df, key_prefix=key_prefix)

    with st.expander("Switch dashboard type"):
        override = st.selectbox(
            "Force dashboard view",
            options=["auto", "restaurant_daily", "employee_labor", "canonical_records", "generic"],
            index=0,
            key=f"{key_prefix}_override",
        )
        if override != "auto" and override != profile:
            if override == "restaurant_daily":
                render_restaurant_daily_dashboard(df, key_prefix=f"{key_prefix}_o_rd")
            elif override == "employee_labor":
                render_employee_labor_dashboard(
                    df,
                    key_prefix=f"{key_prefix}_o_el",
                    reports_dir=str(project.root / "reports"),
                )
            elif override == "canonical_records":
                render_canonical_records_dashboard(df, key_prefix=f"{key_prefix}_o_rec")
            else:
                render_generic_dashboard(df, key_prefix=f"{key_prefix}_o_gen")