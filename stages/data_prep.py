"""
stages/data_prep.py

Data preparation stage: train/test split + optional scaling.
See grok_handoff.md Section 6.6.

Persists splits and scaler to artifacts/prep/ for later use (model, evaluate,
deploy/inference). The sklearn backend trains on the persisted (scaled) train
split and the deploy stage re-applies the same scaler at inference, so train and
eval see one feature representation. The PyCaret backend is self-contained and
consumes the raw working df directly (it does its own preprocessing).

Does not modify the main working dataframe.
"""

from __future__ import annotations

import pickle
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from core.pipeline import StageResult
from core.project import Project


def default_settings(project: Project) -> dict[str, Any]:
    """Default settings for data preparation."""
    return {
        "test_size": 0.2,
        "stratify": True,  # only used for classification
        "scaler": "standard",  # 'standard' | 'minmax' | 'none'
        "random_state": 42,
        # Leak-safety guardrail (see CLAUDE.md): temporal is the default so a bare
        # run holds out the most-recent rows and never look-ahead-leaks into a
        # forward-return target. Callers opt into 'random' explicitly for non
        # time-series data. 'random' | 'temporal'.
        "split_strategy": "temporal",
        "time_column": None,  # raw-data column to order a temporal split by
    }


def _temporal_order(
    project: Project, df: pd.DataFrame, time_column: str | None
) -> tuple[list[Any], str]:
    """Return ``df.index`` in chronological order for a temporal split.

    When ``time_column`` resolves in the project's *raw* data, rows are ordered
    by that timestamp (aligned to ``df`` by index, so the date orders the split
    but never enters the feature matrix). Otherwise the existing row order is
    assumed to already be chronological. Rows with an unparseable timestamp sort
    last (into the test tail).
    """
    if time_column:
        try:
            if project.has_raw_data():
                raw = project.get_raw_data()
                if time_column in raw.columns:
                    key = pd.to_datetime(raw[time_column], errors="coerce").reindex(df.index)
                    if key.notna().any():
                        ordered = key.sort_values(kind="stable", na_position="last").index
                        return list(ordered), f"temporal split ordered by raw '{time_column}'"
        except Exception:
            pass
    return list(df.index), "temporal split on existing row order (no time_column resolved)"


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    """Perform split and scaling. Save artifacts. Return success without changing working df."""
    if df is None or len(df) < 10:
        return StageResult(
            status="error", error="Insufficient data for train/test split (need at least 10 rows)."
        )

    target = project.meta.get("target_column")
    task_type = project.meta.get("task_type", "regression")

    if not target or target not in df.columns:
        return StageResult(status="error", error="Target column not set or not present in data.")

    X = df.drop(columns=[target])
    y = df[target]

    test_size = float(settings.get("test_size", 0.2))
    random_state = int(settings.get("random_state", 42))
    # Leak-safe default (CLAUDE.md): fall back to temporal when a caller passes a
    # partial settings dict straight to run_stage() without merging stage defaults.
    split_strategy = str(settings.get("split_strategy", "temporal")).lower()
    use_stratify = (
        bool(settings.get("stratify", True))
        and task_type == "classification"
        and split_strategy != "temporal"
    )

    stratify = y if use_stratify else None
    summary_note: str | None = None

    if split_strategy == "temporal":
        # Contiguous time-ordered holdout: train on the oldest rows, test on the
        # most recent. Avoids the look-ahead leakage a random split causes on
        # overlapping-window time-series. No shuffling; scaler still fit on train.
        order_index, summary_note = _temporal_order(project, df, settings.get("time_column"))
        n_test = max(1, int(round(len(df) * test_size)))
        n_test = min(n_test, len(df) - 1)  # keep at least one training row
        train_idx, test_idx = order_index[:-n_test], order_index[-n_test:]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError as e:
            # Fallback: retry without stratify if classes are too rare
            if use_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=None
                )
                summary_note = "Stratify disabled (too few samples per class)"
            else:
                return StageResult(status="error", error=f"Train/test split failed: {e}")

    scaler_type = settings.get("scaler", "standard")
    scaler = None
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()

    if scaler is not None:
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    prep_dir = project.artifact_path("prep")
    prep_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(prep_dir / "X_train.parquet")
    X_test.to_parquet(prep_dir / "X_test.parquet")
    y_train.to_frame(name=target).to_parquet(prep_dir / "y_train.parquet")
    y_test.to_frame(name=target).to_parquet(prep_dir / "y_test.parquet")

    if scaler is not None:
        with open(prep_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    summary: dict[str, Any] = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "scaler": scaler_type,
        "split_strategy": split_strategy,
        "stratify_used": stratify is not None,
        "test_size": test_size,
    }
    if summary_note:
        summary["note"] = summary_note

    artifacts = [
        prep_dir / "X_train.parquet",
        prep_dir / "X_test.parquet",
        prep_dir / "y_train.parquet",
        prep_dir / "y_test.parquet",
    ]
    if scaler is not None:
        artifacts.append(prep_dir / "scaler.pkl")

    return StageResult(
        status="success",
        output_df=None,  # do not overwrite working df; modeling uses original
        artifacts=artifacts,
        summary=summary,
    )


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    import streamlit as st

    settings["test_size"] = st.slider(
        "Test size",
        0.1, 0.4,
        float(settings.get("test_size", 0.2)),
        step=0.05,
    )
    settings["scaler"] = st.selectbox(
        "Scaler",
        ["standard", "minmax", "none"],
        index=["standard", "minmax", "none"].index(settings.get("scaler", "standard")),
    )
    settings["split_strategy"] = st.selectbox(
        "Split strategy",
        ["random", "temporal"],
        index=["random", "temporal"].index(settings.get("split_strategy", "temporal")),
        help="temporal (default) = train on oldest rows, test on most recent (no look-ahead).",
    )
    if settings["split_strategy"] == "temporal":
        settings["time_column"] = (
            st.text_input(
                "Time column (in raw data) to order by",
                value=settings.get("time_column") or "",
            )
            or None
        )
    settings["stratify"] = st.checkbox(
        "Stratify split (classification)",
        value=bool(settings.get("stratify", True)),
    )
    settings["random_state"] = st.number_input(
        "Random seed",
        min_value=0,
        value=int(settings.get("random_state", 42)),
    )
    return settings
