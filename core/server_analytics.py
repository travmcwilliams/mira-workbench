"""Server performance analytics — rankings, peer deltas, cohort distributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.restaurant_features import engineer_labor_features

_SERVER_ROLE_NAMES = {"server"}

_RANK_METRICS: tuple[tuple[str, str], ...] = (
    ("tips_per_hour", "Tips per hour ($)"),
    ("total_tips", "Total tips ($)"),
    ("sales_per_hour", "Sales per hour ($)"),
    ("tips_pct_sales", "Tips % of sales"),
    ("effective_hourly", "Effective hourly ($)"),
)


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    norm = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    return None


def prepare_server_cohort(
    df: pd.DataFrame,
    *,
    min_hours: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Engineer features and return server-only rows with ranking columns.
    Uses role-level rows (not employee aggregation) so pure server shifts are isolated.
    """
    engineered, fe_summary = engineer_labor_features(
        df,
        target=None,
        add_peer_benchmarks=True,
        aggregate_employee=False,
    )
    role_col = _resolve_col(engineered, "role")
    if not role_col:
        return pd.DataFrame(), {"error": "No role column", "applied": False}

    servers = engineered[
        engineered[role_col].astype(str).str.strip().str.lower().isin(_SERVER_ROLE_NAMES)
    ].copy()

    if min_hours > 0 and "total_hours" in servers.columns:
        servers = servers[servers["total_hours"] >= min_hours]

    if servers.empty:
        return servers, {
            "applied": True,
            "server_count": 0,
            "min_hours": min_hours,
            "restaurant_features": fe_summary,
        }

    if "employee_key" not in servers.columns:
        servers["employee_key"] = servers.index.astype(str)

    servers["display_name"] = servers["employee_key"]

    for metric, _label in _RANK_METRICS:
        if metric not in servers.columns:
            continue
        vals = pd.to_numeric(servers[metric], errors="coerce")
        med = float(vals.median())
        servers[f"{metric}_median"] = med
        servers[f"{metric}_delta"] = vals - med
        servers[f"{metric}_pct_above_median"] = np.where(
            med > 0, (vals / med - 1.0) * 100.0, np.nan
        )
        servers[f"{metric}_percentile"] = vals.rank(pct=True, method="average") * 100.0

    # Composite productivity score: average percentile across core tip metrics
    pct_cols = [
        c
        for c in (
            "tips_per_hour_percentile",
            "total_tips_percentile",
            "sales_per_hour_percentile",
        )
        if c in servers.columns
    ]
    if pct_cols:
        servers["composite_percentile"] = servers[pct_cols].mean(axis=1)

    return servers, {
        "applied": True,
        "server_count": len(servers),
        "min_hours": min_hours,
        "cohort_medians": {
            metric: float(servers[metric].median())
            for metric, _ in _RANK_METRICS
            if metric in servers.columns
        },
        "restaurant_features": fe_summary,
    }


def server_leaderboard(
    servers: pd.DataFrame,
    *,
    metric: str = "tips_per_hour",
    top_n: int | None = None,
) -> pd.DataFrame:
    """Ranked server table with peer deltas for one metric."""
    if servers.empty or metric not in servers.columns:
        return pd.DataFrame()

    cols = [
        "display_name",
        "employee_key",
        metric,
        f"{metric}_median",
        f"{metric}_delta",
        f"{metric}_pct_above_median",
        f"{metric}_percentile",
        "total_hours",
        "total_tips",
        "net_sales",
        "sales_per_hour",
        "tips_per_hour",
        "composite_percentile",
    ]
    seen: set[str] = set()
    cols = [c for c in cols if c in servers.columns and not (c in seen or seen.add(c))]
    out = servers[cols].sort_values(metric, ascending=False)
    if top_n:
        out = out.head(top_n)
    return out.reset_index(drop=True)


def cohort_distribution_frame(servers: pd.DataFrame) -> pd.DataFrame:
    """Long-format frame for box/violin plots across server metrics."""
    rows: list[dict[str, Any]] = []
    for metric, label in _RANK_METRICS:
        if metric not in servers.columns:
            continue
        for _, rec in servers.iterrows():
            val = rec.get(metric)
            if pd.isna(val):
                continue
            rows.append(
                {
                    "metric": label,
                    "metric_key": metric,
                    "value": float(val),
                    "display_name": rec.get("display_name", rec.get("employee_key")),
                    "total_hours": rec.get("total_hours"),
                }
            )
    return pd.DataFrame(rows)


def top_vs_bottom_summary(
    servers: pd.DataFrame,
    metric: str = "tips_per_hour",
    n: int = 3,
) -> dict[str, Any]:
    """How much better top performers are vs bottom and vs cohort median."""
    if servers.empty or metric not in servers.columns:
        return {}

    ranked = servers.sort_values(metric, ascending=False)
    med = float(ranked[metric].median())
    top = ranked.head(n)
    bottom = ranked.tail(n)

    def _avg_pct_above(sub: pd.DataFrame) -> float:
        col = f"{metric}_pct_above_median"
        if col in sub.columns:
            return float(sub[col].mean())
        return float(((sub[metric] / med - 1.0) * 100.0).mean()) if med > 0 else 0.0

    return {
        "metric": metric,
        "cohort_median": med,
        "top_names": top["display_name"].tolist(),
        "top_values": [float(v) for v in top[metric]],
        "top_pct_above_median": _avg_pct_above(top),
        "bottom_names": bottom["display_name"].tolist(),
        "bottom_values": [float(v) for v in bottom[metric]],
        "bottom_pct_above_median": _avg_pct_above(bottom),
        "spread_top_vs_bottom": float(top[metric].mean() - bottom[metric].mean()),
        "spread_top_vs_median": float(top[metric].mean() - med),
    }