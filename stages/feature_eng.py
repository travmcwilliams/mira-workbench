"""
stages/feature_eng.py

Feature engineering: encoding, date features, interactions, log transforms, optional Featuretools.
See grok_handoff.md Section 6.6.

Persists fitted encoders in artifacts/encoders.pkl for reuse in inference.
"""

from __future__ import annotations

import pickle
from typing import Any

import pandas as pd

from core.pipeline import StageResult
from core.project import Project
from core.restaurant_features import engineer_labor_features, is_restaurant_labor_export

# Features that are algebraically derived from the target (avoid leakage)
_DERIVED_TARGET_EXCLUSIONS: dict[str, set[str]] = {
    "labor_pct": {
        "wages",
        "net_sales",
        "gross",
        "tax",
        "sales_per_labor_hour",
    },
    # sales_per_labor_hour = net_sales / labor_hours; exclude both sides of the
    # ratio and every column that embeds one of them (labor_pct = wages/net_sales,
    # avg_check = net_sales/checks, labor_hours = regular + ot).
    "sales_per_labor_hour": {
        "net_sales",
        "gross",
        "tax",
        "labor_hours",
        "regular_hours",
        "wages",
        "labor_pct",
        "avg_check",
    },
    # Artifact-outcome verdict proxies. The n_* counts are aggregate
    # attempt-counts of the very pipeline whose pass/fail is the label: a stuck
    # artifact accrues many attempts AND fails, so the counts half-encode the
    # verdict (same hazard class as attempt_index). min_score/threshold/the sibling
    # verdict fields are direct read-outs of the gate decision. Exclude them.
    "passed": {
        "n_post_ok",
        "n_blocked_post",
        "n_escalated",
        "min_score",
        "threshold",
        "n_pre",
        "n_model_call",
        "n_post",
    },
    "gate_score": {
        "n_post_ok",
        "n_blocked_post",
        "n_escalated",
        "passed",
        "n_pre",
        "n_model_call",
        "n_post",
    },
}


def default_settings(project: Project) -> dict[str, Any]:
    schema = project.meta.get("schema", {})
    df = project.get_working_data() if project.has_raw_data() else pd.DataFrame()
    target = project.meta.get("target_column")

    encoding: dict[str, str] = {}
    date_features: list[str] = []
    drop_columns: list[str] = []

    for col, dtype in schema.items():
        if col not in df.columns or col == target:
            continue
        if dtype in {"id", "text"}:
            drop_columns.append(col)
            continue
        if dtype == "datetime":
            date_features.append(col)
            continue
        if dtype == "categorical":
            nunique = df[col].nunique(dropna=True)
            encoding[col] = "onehot" if nunique <= 12 else "label"

    labor = is_restaurant_labor_export(df)
    return {
        "encoding": encoding,
        "date_features": date_features,
        "cyclical_features": list(date_features),
        "cyclical_encoding": True,
        "drop_columns": drop_columns,
        "interactions": [],
        "featuretools": False,
        "log_transform": [],
        # Lag / sequence features. Off by default (empty group/order/columns) so
        # cross-sectional projects are untouched; populate for sequential data
        # (e.g. group_key="lineage_root", order_key="created_at"). See run().
        "lag_features": {
            "group_key": "",
            "order_key": "",
            "columns": [],
            "lags": [],
            "windows": [],
            "diffs": [],
            "ewma": [],
            "attempt_index": False,
            "allow_target_proxy": False,
        },
        "restaurant_labor": labor,
        "restaurant_aggregate_employee": labor,
        "restaurant_peer_benchmarks": True,
        "restaurant_shift_hours": 8.0,
    }


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    if df is None or len(df) == 0:
        return StageResult(status="error", error="No data for feature engineering.")

    df = df.copy()
    encoders: dict[str, Any] = {}
    dropped: list[str] = []
    dropped_details: list[dict[str, str]] = []

    target = project.meta.get("target_column")
    restaurant_summary: dict[str, Any] = {}

    drop_reason_map = {
        col: reason
        for col, reason in (project.meta.get("schema", {}) or {}).items()
        if reason in {"id", "text"}
    }

    import numpy as np  # local import

    def record_drop(col: str, reason: str) -> None:
        dropped.append(col)
        dropped_details.append({"column": col, "reason": reason})

    cyclical_on = settings.get("cyclical_encoding", True)
    cyclical_cols = set(settings.get("cyclical_features", []))

    def add_cyclical(frame: pd.DataFrame, col: str, parsed: pd.Series) -> tuple[dict[str, int], list[str]]:
        # Additive sin/cos encoding for the natural periods present in a timestamp.
        # Formula: sin(2*pi*v/period), cos(2*pi*v/period).
        periods: dict[str, int] = {"dow": 7, "month": 12}
        values: dict[str, pd.Series] = {"dow": parsed.dt.dayofweek, "month": parsed.dt.month}
        has_time = bool(
            ((parsed.dt.hour != 0) | (parsed.dt.minute != 0) | (parsed.dt.second != 0)).any()
        )
        if has_time:
            periods["hour"] = 24
            values["hour"] = parsed.dt.hour
        new_cols: list[str] = []
        for part, period in periods.items():
            angle = 2.0 * np.pi * values[part] / period
            sin_name, cos_name = f"{col}_{part}_sin", f"{col}_{part}_cos"
            frame[sin_name] = np.sin(angle)
            frame[cos_name] = np.cos(angle)
            new_cols.extend([sin_name, cos_name])
        return periods, new_cols

    if settings.get("restaurant_labor") or is_restaurant_labor_export(df):
        df, restaurant_summary = engineer_labor_features(
            df,
            target=target,
            add_peer_benchmarks=settings.get("restaurant_peer_benchmarks", True),
            aggregate_employee=settings.get("restaurant_aggregate_employee", False),
            shift_hours=float(settings.get("restaurant_shift_hours", 8.0)),
        )

    # Lag / sequence features (leak-safe) — runs here, after restaurant labor and
    # BEFORE drop_columns, so the raw id/datetime group & order keys still exist.
    lag_cfg = settings.get("lag_features") or {}
    if lag_cfg.get("columns") and lag_cfg.get("group_key") and lag_cfg.get("order_key"):
        gk = lag_cfg["group_key"]
        ok = lag_cfg["order_key"]
        if gk in df.columns and ok in df.columns:
            lags = list(lag_cfg.get("lags", []))
            windows = list(lag_cfg.get("windows", []))
            diffs = list(lag_cfg.get("diffs", []))
            ewma = list(lag_cfg.get("ewma", []))
            # LEAKAGE GUARD 1: a non-positive lag/diff reads the current row's own
            # value at verdict time — refuse before touching the frame.
            for n in lags + diffs:
                if n <= 0:
                    raise ValueError(
                        f"lag_features: lag/diff {n} <= 0 reads the current row (leakage); "
                        f"lags/diffs must be >= 1"
                    )
            df = df.sort_values([gk, ok], kind="mergesort")
            g = df.groupby(gk, sort=False)
            lag_columns: list[str] = []
            for col in lag_cfg["columns"]:
                if col not in df.columns:
                    continue
                for n in lags:
                    nm = f"{col}_lag{n}"
                    df[nm] = g[col].shift(n)
                    lag_columns.append(nm)
                # LEAKAGE GUARD 2: compute every window on the shift(1) series so the
                # current row is never inside its own window (closed='left' equivalent).
                base = g[col].shift(1)
                for w in windows:
                    nm = f"{col}_rollmean{w}"
                    df[nm] = (
                        base.groupby(df[gk])
                        .rolling(w, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                    lag_columns.append(nm)
                for n in diffs:
                    nm = f"{col}_diff{n}"
                    df[nm] = df[col] - g[col].shift(n)
                    lag_columns.append(nm)
                for s in ewma:
                    nm = f"{col}_ewma{s}"
                    df[nm] = base.groupby(df[gk]).transform(lambda x, s=s: x.ewm(span=s).mean())
                    lag_columns.append(nm)
            # LEAKAGE GUARD 3: attempt_index is temporally safe but a strong target
            # proxy (a later attempt implies earlier ones failed). Double opt-in AND
            # it must never run when the target itself is being lagged.
            if lag_cfg.get("attempt_index"):
                if not lag_cfg.get("allow_target_proxy"):
                    raise ValueError(
                        "lag_features: attempt_index is a target proxy; "
                        "set allow_target_proxy=True to opt in explicitly"
                    )
                if target and target in lag_cfg["columns"]:
                    raise ValueError(
                        "lag_features: attempt_index would leak the per-row verdict target"
                    )
                df["attempt_index"] = g.cumcount()
                lag_columns.append("attempt_index")
            encoders["_lag_features"] = {
                "type": "lag_features",
                "config": lag_cfg,
                "columns": lag_columns,
            }

    for col in settings.get("drop_columns", []):
        if col in df.columns and col != target:
            df = df.drop(columns=[col])
            record_drop(col, drop_reason_map.get(col, "exclusion"))

    # Date features — extract parts, then drop raw datetime columns
    for col in settings.get("date_features", []):
        if col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = parsed.dt.year
                df[f"{col}_month"] = parsed.dt.month
                df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                encoders[col] = {
                    "type": "datetime_parts",
                    "columns": [f"{col}_year", f"{col}_month", f"{col}_dayofweek"],
                }
                if cyclical_on and col in cyclical_cols:
                    periods, cyc_cols = add_cyclical(df, col, parsed)
                    encoders[f"{col}_cyclical"] = {
                        "type": "datetime_cyclical",
                        "col": col,
                        "periods": periods,
                        "columns": cyc_cols,
                    }
                df = df.drop(columns=[col])
                record_drop(col, "datetime")
            except Exception:
                pass

    # Log transforms
    for col in settings.get("log_transform", []):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # Encoding — Role uses one-hot; role_family from engineer_labor_features too
    encoding = settings.get("encoding", {})
    for col, method in list(encoding.items()):
        if col not in df.columns:
            continue
        if method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            encoders[col] = {"type": "onehot", "columns": list(dummies.columns)}
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
        elif method == "label":
            df[col] = df[col].astype("category").cat.codes
            encoders[col] = {
                "type": "label",
                "mapping": list(df[col].astype("category").cat.categories),
            }
        # target encoding would require target and category_encoders lib

    for cat_col in ("role_family", "primary_role_family", "primary_role"):
        if cat_col not in df.columns or cat_col in encoding:
            continue
        nunique = df[cat_col].nunique(dropna=True)
        if nunique <= 12:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True, dtype=int)
            encoders[cat_col] = {"type": "onehot", "columns": list(dummies.columns)}
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[cat_col])
        else:
            df[cat_col] = df[cat_col].astype("category").cat.codes
            encoders[cat_col] = {
                "type": "label",
                "mapping": list(df[cat_col].astype("category").cat.categories),
            }

    if target:
        leak_cols = _DERIVED_TARGET_EXCLUSIONS.get(str(target).lower(), set())
        # Columns derived from leak columns (or from the target itself), such as
        # the clean stage's *_is_outlier flags, leak the same information.
        leak_cols = (
            leak_cols
            | {f"{c}_is_outlier" for c in leak_cols}
            | {f"{target}_is_outlier"}
        )
        for col in leak_cols:
            if col in df.columns and col != target:
                df = df.drop(columns=[col])
                record_drop(col, "exclusion")

    # Simple interactions (multiplicative)
    for pair in settings.get("interactions", []):
        if len(pair) == 2 and pair[0] in df.columns and pair[1] in df.columns:
            name = f"{pair[0]}_x_{pair[1]}"
            df[name] = df[pair[0]] * df[pair[1]]

    # Featuretools (optional, heavy) - placeholder only
    if settings.get("featuretools", False):
        print("Featuretools deep feature synthesis skipped in minimal implementation.")

    # Persist encoders
    enc_path = project.artifact_path("encoders.pkl")
    if encoders:
        with open(enc_path, "wb") as f:
            pickle.dump(encoders, f)

    prior_cols = len(project.get_working_data().columns) if project.has_raw_data() else 0
    summary = {
        "new_columns": len(df.columns) - prior_cols,
        "encoders_saved": len(encoders),
        "dropped_columns": dropped,
        "dropped_column_reasons": dropped_details,
        "restaurant_features": restaurant_summary,
        "lag_features": encoders.get("_lag_features", {}).get("columns", []),
    }

    return StageResult(
        status="success",
        output_df=df,
        artifacts=[enc_path] if encoders else [],
        summary=summary,
    )


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    return settings


def code_template(settings: dict[str, Any]) -> str:
    return f"""# Feature Engineering
# encoding = {settings.get("encoding", {})}
# date_features = {settings.get("date_features", [])}
print("Feature engineering applied.")
"""
