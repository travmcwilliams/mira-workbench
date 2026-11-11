"""Feature engineering for original POS labor report exports (employee × role rows)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Canonical column names after normalization
_COL = {
    "reg_hrs": ("reg hrs", "reg_hrs"),
    "ot_hrs": ("1.5x hrs", "1.5x_hrs", "ot_hours"),
    "pay_rate": ("payperhour", "pay_per_hour"),
    "total_pay": ("total pay", "total_pay"),
    "net_sales": ("net sales", "net_sales"),
    "total_tips": ("total tips", "total_tips"),
    "cash_tips": ("cash tips", "cash_tips"),
    "credit_tips": ("credit tips", "credit_tips"),
    "tender_tips": ("tender tips", "tender_tips"),
    "contributions": ("contributions",),
    "distributions": ("distributions",),
    "role": ("role",),
    "non_rev": ("non revenue receipts", "non_revenue_receipts"),
    "auto_grat": ("auto grat", "auto_grat"),
    "first_name": ("first name", "first_name"),
    "last_name": ("last name", "last_name"),
    "employee_id": ("employee id", "employee_id"),
}

_ROLE_FAMILY: dict[str, str] = {
    "server": "foh_tipped",
    "bartender": "foh_tipped",
    "host": "foh_support",
    "busser/runner": "foh_support",
    "dish/porter": "boh",
    "manager": "management",
    "training": "training",
}

_ROLE_FAMILIES = ("foh_tipped", "foh_support", "boh", "management", "training", "other")

# Raw columns superseded by engineered features (safe to drop after engineering)
_SUPERSEDED_RAW = {
    "reg hrs",
    "reg_hrs",
    "1.5x hrs",
    "1.5x_hrs",
    "cash tips",
    "cash_tips",
    "credit tips",
    "credit_tips",
    "tender tips",
    "tender_tips",
    "declared tips",
    "declared_tips",
    "employee id",
    "employee_id",
}

# Target-specific leakage: components that directly compose the target
_TARGET_LEAKAGE: dict[str, set[str]] = {
    "total_tips": {
        "cash tips",
        "cash_tips",
        "credit tips",
        "credit_tips",
        "tender tips",
        "tender_tips",
        "contributions",
        "distributions",
        "tips_per_hour",
        "tips_pct_sales",
        "credit_tip_share",
        "cash_tip_share",
        "tip_pool_net",
        "tip_pool_ratio",
        "tips_vs_role_median",
        "tips_per_hour_vs_role_median",
        "tips_vs_store_median",
        "tender_tip_share",
    },
    "total_pay": {
        "total pay",
        "total_pay",
        "effective_hourly",
        "labor_cost_per_sales_dollar",
        "pay_vs_role_median",
    },
    "net_sales": {
        "net sales",
        "net_sales",
        "sales_per_hour",
        "labor_cost_per_sales_dollar",
        "tips_pct_sales",
        "sales_per_hour_vs_role_median",
        "sales_vs_role_median",
        "sales_vs_store_median",
    },
}

_SUM_COLUMNS = (
    "total_hours",
    "reg_hours_component",
    "ot_hours_component",
    "total_pay",
    "net_sales",
    "total_tips",
    "contributions",
    "distributions",
    "non_revenue_receipts",
    "auto_grat",
    "cash_tips",
    "credit_tips",
    "tender_tips",
    "tip_pool_net",
)


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _resolve(df: pd.DataFrame, keys: tuple[str, ...]) -> str | None:
    mapping = {_norm(c): c for c in df.columns}
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def is_restaurant_labor_export(df: pd.DataFrame) -> bool:
    cols = {_norm(c) for c in df.columns}
    return "role" in cols and bool(
        cols & {"reg_hrs", "total_tips", "total_pay", "net_sales"}
    )


def _employee_key(df: pd.DataFrame) -> pd.Series:
    first = _resolve(df, _COL["first_name"])
    last = _resolve(df, _COL["last_name"])
    emp_id = _resolve(df, _COL["employee_id"])
    if first and last:
        return (
            df[first].astype(str).str.strip() + " " + df[last].astype(str).str.strip()
        ).str.strip()
    if last:
        return df[last].astype(str).str.strip()
    if emp_id:
        return df[emp_id].astype(str).str.strip()
    return pd.Series(range(len(df)), index=df.index).astype(str)


def _role_family_series(df: pd.DataFrame, role_col: str) -> pd.Series:
    return (
        df[role_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda r: _ROLE_FAMILY.get(r, "other"))
    )


def _add_shift_features(
    out: pd.DataFrame,
    *,
    shift_hours: float,
    engineered: list[str],
) -> None:
    out["implied_shifts"] = np.where(
        out["total_hours"] > 0, out["total_hours"] / shift_hours, 0.0
    )
    out["is_overtime_heavy"] = (out["ot_share"] >= 0.15).astype(int)
    out["is_high_hour_role"] = (out["total_hours"] >= shift_hours * 40).astype(int)
    engineered.extend(["implied_shifts", "is_overtime_heavy", "is_high_hour_role"])


def _add_peer_benchmarks(
    out: pd.DataFrame,
    *,
    role_col: str | None,
    tips_s: pd.Series,
    sales_s: pd.Series,
    pay_s: pd.Series,
    engineered: list[str],
) -> None:
    if not role_col:
        return

    role = out[role_col]
    tips = tips_s
    sales = sales_s
    hours = out["total_hours"]

    tips_ph = np.where(hours > 0, tips / hours, np.nan)
    sales_ph = np.where(hours > 0, sales / hours, np.nan)

    role_tips_median = (
        pd.Series(tips_ph, index=out.index).groupby(role, observed=True).transform("median")
    )
    role_sales_median = (
        pd.Series(sales_ph, index=out.index).groupby(role, observed=True).transform("median")
    )
    role_tips_total_median = tips.groupby(role, observed=True).transform("median")
    role_sales_total_median = sales.groupby(role, observed=True).transform("median")
    role_pay_median = pay_s.groupby(role, observed=True).transform("median")

    out["tips_per_hour_vs_role_median"] = np.where(
        role_tips_median > 0, tips_ph / role_tips_median, np.nan
    )
    out["sales_per_hour_vs_role_median"] = np.where(
        role_sales_median > 0, sales_ph / role_sales_median, np.nan
    )
    out["tips_vs_role_median"] = np.where(
        role_tips_total_median > 0, tips / role_tips_total_median, np.nan
    )
    out["sales_vs_role_median"] = np.where(
        role_sales_total_median > 0, sales / role_sales_total_median, np.nan
    )
    out["pay_vs_role_median"] = np.where(
        role_pay_median > 0, pay_s / role_pay_median, np.nan
    )
    engineered.extend(
        [
            "tips_per_hour_vs_role_median",
            "sales_per_hour_vs_role_median",
            "tips_vs_role_median",
            "sales_vs_role_median",
            "pay_vs_role_median",
        ]
    )


def _add_role_mix_features(
    out: pd.DataFrame,
    *,
    role_col: str | None,
    emp_key: pd.Series,
    reg_s: pd.Series,
    ot_s: pd.Series,
    engineered: list[str],
) -> None:
    if not role_col:
        return

    out["_employee_key"] = emp_key
    grp = out.groupby("_employee_key", observed=True)

    out["n_roles"] = grp[role_col].transform("nunique")
    out["is_multi_role"] = (out["n_roles"] > 1).astype(int)
    out["employee_total_hours"] = grp["total_hours"].transform("sum")
    out["role_hours_share"] = np.where(
        out["employee_total_hours"] > 0,
        out["total_hours"] / out["employee_total_hours"],
        0.0,
    )

    max_hours = grp["total_hours"].transform("max")
    out["is_primary_role"] = (out["total_hours"] >= max_hours).astype(int)

    if "role_family" in out.columns:
        emp_total = grp["total_hours"].transform("sum")
        for family in _ROLE_FAMILIES:
            col = f"employee_{family}_hours_share"
            fam_sum = (
                out.loc[out["role_family"].eq(family)]
                .groupby("_employee_key", observed=True)["total_hours"]
                .sum()
            )
            out[col] = (
                emp_key.map(fam_sum).fillna(0) / emp_total.replace(0, np.nan)
            ).fillna(0.0)
            engineered.append(col)

        out["employee_foh_hours_share"] = (
            out.get("employee_foh_tipped_hours_share", 0)
            + out.get("employee_foh_support_hours_share", 0)
        )
        out["employee_is_mostly_foh"] = (out["employee_foh_hours_share"] >= 0.5).astype(int)
        out["employee_is_mostly_boh"] = (
            out.get("employee_boh_hours_share", 0) >= 0.5
        ).astype(int)
        engineered.extend(
            [
                "employee_foh_hours_share",
                "employee_is_mostly_foh",
                "employee_is_mostly_boh",
            ]
        )

    engineered.extend(
        [
            "n_roles",
            "is_multi_role",
            "employee_total_hours",
            "role_hours_share",
            "is_primary_role",
        ]
    )


def _aggregate_to_employee(
    df: pd.DataFrame,
    *,
    role_col: str | None,
    target_col: str | None,
    engineered: list[str],
) -> pd.DataFrame:
    if "_employee_key" not in df.columns:
        return df

    work = df.copy()
    emp = work["_employee_key"]

    # Preserve raw column names for summable fields
    raw_pay = _resolve(work, _COL["total_pay"])
    raw_sales = _resolve(work, _COL["net_sales"])
    raw_tips = _resolve(work, _COL["total_tips"])
    raw_contrib = _resolve(work, _COL["contributions"])
    raw_dist = _resolve(work, _COL["distributions"])
    raw_non_rev = _resolve(work, _COL["non_rev"])
    raw_auto = _resolve(work, _COL["auto_grat"])
    raw_cash = _resolve(work, _COL["cash_tips"])
    raw_credit = _resolve(work, _COL["credit_tips"])
    raw_tender = _resolve(work, _COL["tender_tips"])

    if raw_pay and "total_pay" not in work.columns:
        work["total_pay"] = pd.to_numeric(work[raw_pay], errors="coerce").fillna(0)
    if raw_sales and "net_sales" not in work.columns:
        work["net_sales"] = pd.to_numeric(work[raw_sales], errors="coerce").fillna(0)
    if raw_tips and "total_tips" not in work.columns:
        work["total_tips"] = pd.to_numeric(work[raw_tips], errors="coerce").fillna(0)

    if "reg_hours_component" not in work.columns:
        reg_col = _resolve(work, _COL["reg_hrs"])
        work["reg_hours_component"] = (
            pd.to_numeric(work[reg_col], errors="coerce").fillna(0) if reg_col else 0.0
        )
    if "ot_hours_component" not in work.columns:
        ot_col = _resolve(work, _COL["ot_hrs"])
        work["ot_hours_component"] = (
            pd.to_numeric(work[ot_col], errors="coerce").fillna(0) if ot_col else 0.0
        )

    agg_spec: dict[str, Any] = {
        "total_hours": "sum",
        "reg_hours_component": "sum",
        "ot_hours_component": "sum",
        "n_roles": "max",
        "is_multi_role": "max",
        "employee_foh_hours_share": "first",
        "employee_is_mostly_foh": "max",
        "employee_is_mostly_boh": "max",
    }
    for col in (
        "total_pay",
        "net_sales",
        "total_tips",
        "tip_pool_net",
        "non_revenue_receipts",
        "auto_grat",
        "cash_tips",
        "credit_tips",
        "tender_tips",
        "contributions",
        "distributions",
    ):
        if col in work.columns:
            agg_spec[col] = "sum"

    for family in _ROLE_FAMILIES:
        col = f"employee_{family}_hours_share"
        if col in work.columns:
            agg_spec[col] = "first"

    if role_col and role_col in work.columns:
        # Primary role = role with most hours per employee
        idx = work.groupby(emp, observed=True)["total_hours"].idxmax()
        primary_roles = work.loc[idx].set_index("_employee_key")[role_col]
        grouped = work.groupby(emp, observed=True).agg(agg_spec)
        grouped["primary_role"] = primary_roles
        if "role_family" in work.columns:
            fam_idx = work.groupby(emp, observed=True)["total_hours"].idxmax()
            grouped["primary_role_family"] = (
                work.loc[fam_idx].set_index("_employee_key")["role_family"]
            )
    else:
        grouped = work.groupby(emp, observed=True).agg(agg_spec)

    grouped = grouped.reset_index().rename(columns={"_employee_key": "employee_key"})
    grouped["ot_share"] = np.where(
        grouped["total_hours"] > 0,
        grouped["ot_hours_component"] / grouped["total_hours"],
        0.0,
    )
    grouped["effective_hourly"] = np.where(
        grouped["total_hours"] > 0,
        grouped["total_pay"] / grouped["total_hours"],
        0.0,
    )
    grouped["has_sales"] = (grouped["net_sales"] > 0).astype(int)
    grouped["sales_per_hour"] = np.where(
        grouped["total_hours"] > 0,
        grouped["net_sales"] / grouped["total_hours"],
        0.0,
    )
    grouped["labor_cost_per_sales_dollar"] = np.where(
        grouped["net_sales"] > 0,
        grouped["total_pay"] / grouped["net_sales"],
        np.nan,
    )
    grouped["tips_per_hour"] = np.where(
        grouped["total_hours"] > 0,
        grouped["total_tips"] / grouped["total_hours"],
        0.0,
    )
    grouped["tips_pct_sales"] = np.where(
        grouped["net_sales"] > 0,
        grouped["total_tips"] / grouped["net_sales"],
        np.nan,
    )

    if "cash_tips" in grouped.columns and "credit_tips" in grouped.columns:
        grouped["credit_tip_share"] = np.where(
            grouped["total_tips"] > 0,
            grouped["credit_tips"] / grouped["total_tips"],
            np.nan,
        )
        grouped["cash_tip_share"] = np.where(
            grouped["total_tips"] > 0,
            grouped["cash_tips"] / grouped["total_tips"],
            np.nan,
        )
    if "tender_tips" in grouped.columns:
        grouped["tender_tip_share"] = np.where(
            grouped["total_tips"] > 0,
            grouped["tender_tips"] / grouped["total_tips"],
            np.nan,
        )
    if "contributions" in grouped.columns and "distributions" in grouped.columns:
        grouped["tip_pool_ratio"] = np.where(
            grouped["contributions"] > 0,
            grouped["distributions"] / grouped["contributions"],
            np.nan,
        )

    store_tips_median = grouped["total_tips"].median()
    store_sales_median = grouped["net_sales"].median()
    store_hours_median = grouped["total_hours"].median()
    grouped["tips_vs_store_median"] = np.where(
        store_tips_median > 0, grouped["total_tips"] / store_tips_median, np.nan
    )
    grouped["sales_vs_store_median"] = np.where(
        store_sales_median > 0, grouped["net_sales"] / store_sales_median, np.nan
    )
    grouped["hours_vs_store_median"] = np.where(
        store_hours_median > 0, grouped["total_hours"] / store_hours_median, np.nan
    )

    grouped["implied_shifts"] = np.where(
        grouped["total_hours"] > 0, grouped["total_hours"] / 8.0, 0.0
    )
    grouped["is_overtime_heavy"] = (grouped["ot_share"] >= 0.15).astype(int)

    if "primary_role_family" in grouped.columns:
        grouped["is_tipped_role"] = grouped["primary_role_family"].eq("foh_tipped").astype(int)
        grouped["is_boh"] = grouped["primary_role_family"].eq("boh").astype(int)

    # Restore target column name if different from normalized
    if target_col and target_col != "total_tips" and "total_tips" in grouped.columns:
        grouped = grouped.rename(columns={"total_tips": target_col})

    new_cols = [
        "employee_key",
        "primary_role",
        "primary_role_family",
        "ot_share",
        "effective_hourly",
        "sales_per_hour",
        "tips_per_hour",
        "tips_vs_store_median",
        "sales_vs_store_median",
        "hours_vs_store_median",
        "implied_shifts",
    ]
    engineered.extend([c for c in new_cols if c in grouped.columns])

    return grouped


def engineer_labor_features(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    add_peer_benchmarks: bool = True,
    aggregate_employee: bool = False,
    shift_hours: float = 8.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build restaurant ops features from raw POS labor rows.

    Row-level mode keeps one row per employee × role (good for role-specific models).
    Employee aggregation collapses multi-role staff to one row per person.

    Returns transformed frame + summary of engineered columns.
    """
    if not is_restaurant_labor_export(df):
        return df, {"applied": False}

    out = df.copy()
    reg = _resolve(out, _COL["reg_hrs"])
    ot = _resolve(out, _COL["ot_hrs"])
    pay_rate = _resolve(out, _COL["pay_rate"])
    total_pay = _resolve(out, _COL["total_pay"])
    net_sales = _resolve(out, _COL["net_sales"])
    total_tips = _resolve(out, _COL["total_tips"])
    cash = _resolve(out, _COL["cash_tips"])
    credit = _resolve(out, _COL["credit_tips"])
    tender = _resolve(out, _COL["tender_tips"])
    contrib = _resolve(out, _COL["contributions"])
    dist = _resolve(out, _COL["distributions"])
    role = _resolve(out, _COL["role"])
    non_rev = _resolve(out, _COL["non_rev"])
    auto_grat = _resolve(out, _COL["auto_grat"])

    engineered: list[str] = []

    def _num(col: str | None) -> pd.Series:
        if col and col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").fillna(0)
        return pd.Series(0.0, index=out.index)

    reg_s = _num(reg)
    ot_s = _num(ot)
    out["reg_hours_component"] = reg_s
    out["ot_hours_component"] = ot_s
    out["total_hours"] = reg_s + ot_s
    out["ot_share"] = np.where(out["total_hours"] > 0, ot_s / out["total_hours"], 0.0)
    engineered.extend(["total_hours", "ot_share"])

    pay_s = _num(total_pay)
    out["total_pay"] = pay_s
    out["effective_hourly"] = np.where(out["total_hours"] > 0, pay_s / out["total_hours"], 0.0)
    engineered.append("effective_hourly")

    if pay_rate:
        rate_s = _num(pay_rate)
        out["pay_rate_gap"] = np.where(rate_s > 0, out["effective_hourly"] - rate_s, 0.0)
        engineered.append("pay_rate_gap")

    sales_s = _num(net_sales)
    out["net_sales"] = sales_s
    out["has_sales"] = (sales_s > 0).astype(int)
    out["sales_per_hour"] = np.where(out["total_hours"] > 0, sales_s / out["total_hours"], 0.0)
    out["labor_cost_per_sales_dollar"] = np.where(sales_s > 0, pay_s / sales_s, np.nan)
    engineered.extend(["has_sales", "sales_per_hour", "labor_cost_per_sales_dollar"])

    tips_s = _num(total_tips)
    out["total_tips"] = tips_s
    out["tips_per_hour"] = np.where(out["total_hours"] > 0, tips_s / out["total_hours"], 0.0)
    out["tips_pct_sales"] = np.where(sales_s > 0, tips_s / sales_s, np.nan)
    engineered.extend(["tips_per_hour", "tips_pct_sales"])

    if cash and credit:
        cash_s, credit_s = _num(cash), _num(credit)
        out["cash_tips"] = cash_s
        out["credit_tips"] = credit_s
        out["credit_tip_share"] = np.where(tips_s > 0, credit_s / tips_s, np.nan)
        out["cash_tip_share"] = np.where(tips_s > 0, cash_s / tips_s, np.nan)
        engineered.extend(["credit_tip_share", "cash_tip_share"])
    if tender:
        tender_s = _num(tender)
        out["tender_tips"] = tender_s
        out["tender_tip_share"] = np.where(tips_s > 0, tender_s / tips_s, np.nan)
        engineered.append("tender_tip_share")

    if contrib and dist:
        c_s, d_s = _num(contrib), _num(dist)
        out["contributions"] = c_s
        out["distributions"] = d_s
        out["tip_pool_net"] = d_s - c_s
        out["tip_pool_ratio"] = np.where(c_s > 0, d_s / c_s, np.nan)
        engineered.extend(["tip_pool_net", "tip_pool_ratio"])

    if non_rev:
        nr = _num(non_rev)
        out["non_revenue_receipts"] = nr
        out["non_revenue_pct_sales"] = np.where(sales_s > 0, nr / sales_s, np.nan)
        engineered.append("non_revenue_pct_sales")
    if auto_grat:
        ag = _num(auto_grat)
        out["auto_grat"] = ag
        out["auto_grat_pct_sales"] = np.where(sales_s > 0, ag / sales_s, np.nan)
        engineered.append("auto_grat_pct_sales")

    if role:
        out["role_family"] = _role_family_series(out, role)
        out["is_tipped_role"] = out["role_family"].eq("foh_tipped").astype(int)
        out["is_boh"] = out["role_family"].eq("boh").astype(int)
        engineered.extend(["role_family", "is_tipped_role", "is_boh"])

    _add_shift_features(out, shift_hours=shift_hours, engineered=engineered)

    emp_key = _employee_key(out)
    _add_role_mix_features(
        out,
        role_col=role,
        emp_key=emp_key,
        reg_s=reg_s,
        ot_s=ot_s,
        engineered=engineered,
    )

    if add_peer_benchmarks:
        _add_peer_benchmarks(
            out,
            role_col=role,
            tips_s=tips_s,
            sales_s=sales_s,
            pay_s=pay_s,
            engineered=engineered,
        )

    if not aggregate_employee and "_employee_key" in out.columns:
        out = out.rename(columns={"_employee_key": "employee_key"})

    aggregated = False
    if aggregate_employee:
        target_col = total_tips if total_tips else "total_tips"
        out = _aggregate_to_employee(
            out,
            role_col=role,
            target_col=target_col,
            engineered=engineered,
        )
        aggregated = True

    # Drop identifiers and raw columns replaced by engineered features
    drop: list[str] = []
    target_norm = _norm(target) if target else ""
    id_cols = {_norm("last name"), _norm("first name"), _norm("employee id")}
    for col in out.columns:
        ncol = _norm(col)
        if ncol in id_cols:
            drop.append(col)
        elif ncol in _SUPERSEDED_RAW and ncol != target_norm:
            drop.append(col)
        elif ncol == "role" and aggregated:
            drop.append(col)

    if target_norm:
        leak = _TARGET_LEAKAGE.get(target_norm, set())
        for col in out.columns:
            if _norm(col) in leak and _norm(col) != target_norm:
                if col not in drop:
                    drop.append(col)

    drop = [c for c in drop if c in out.columns and _norm(c) != target_norm]
    out = out.drop(columns=drop, errors="ignore")

    # Drop internal staging columns
    for staging in ("reg_hours_component", "ot_hours_component"):
        if staging in out.columns and staging not in engineered:
            out = out.drop(columns=[staging], errors="ignore")

    return out, {
        "applied": True,
        "engineered_columns": engineered,
        "dropped_raw_columns": drop,
        "row_count": len(out),
        "aggregated_employee": aggregated,
        "peer_benchmarks": add_peer_benchmarks,
    }