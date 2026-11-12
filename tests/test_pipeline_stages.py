"""Pipeline stage tests for encoding and feature selection fixes."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from core.data_loader import detect_schema, load_file, suggest_target
from core.pipeline import Pipeline
from core.project import Project


@pytest.fixture
def temp_projects_dir(monkeypatch, tmp_path):
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(original_cwd)


def _run_until(project: Project, until: str) -> None:
    pipeline = Pipeline(project)
    results = pipeline.run_all(until=until)
    assert results, "pipeline returned no stage results"
    assert results[-1].status == "success", results[-1].error


def test_detect_schema_marks_employee_names_as_text(temp_projects_dir):
    root = Path(__file__).resolve().parents[1]
    df = load_file(root / "data" / "sample" / "restaurant_labor_sample.csv")
    schema = detect_schema(df)
    assert schema["Last Name"] in {"id", "text"}
    assert schema["Role"] == "categorical"


def test_feature_eng_drops_id_columns(temp_projects_dir):
    root = Path(__file__).resolve().parents[1]
    df = load_file(root / "data" / "sample" / "restaurant_labor_sample.csv")
    project = Project.create(name="Labor FE", task_type="regression")
    project.set_raw_data(df)
    project.meta["schema"] = detect_schema(df)
    project.meta["target_column"] = "Total Tips"
    project.save_meta()

    _run_until(project, "feature_eng")
    working = project.get_working_data()
    assert "Last Name" not in working.columns
    assert "First Name" not in working.columns
    assert "employee_key" in working.columns
    assert len(working) < len(df)
    assert any(c.startswith("primary_role_") for c in working.columns)


def test_detect_schema_never_marks_continuous_float_as_id():
    df = pd.DataFrame(
        {
            "tone_ma5": [0.11, 0.23, 0.37, 0.41, 0.52, 0.66, 0.74, 0.83, 0.91, 1.02, 1.14, 1.27],
            "tone_z20": [-1.2, -0.9, -0.7, -0.4, -0.2, 0.1, 0.3, 0.55, 0.8, 1.05, 1.31, 1.6],
        }
    )

    schema = detect_schema(df)

    assert schema["tone_ma5"] == "numeric"
    assert schema["tone_z20"] == "numeric"


def test_feature_eng_keeps_continuous_float_features_with_high_uniqueness(temp_projects_dir):
    df = pd.DataFrame(
        {
            "row_id": range(1, 21),
            "tone_ma5": [0.101 + i * 0.017 for i in range(20)],
            "tone_z20": [-1.25 + i * 0.14 for i in range(20)],
            "target": [10.0 + i for i in range(20)],
        }
    )
    project = Project.create(name="Float honesty", task_type="regression")
    project.set_raw_data(df)
    project.meta["schema"] = detect_schema(df)
    project.meta["target_column"] = "target"
    project.save_meta()

    result = Pipeline(project).run_stage("feature_eng")
    assert result.status == "success", result.error

    working = project.get_working_data()
    assert "tone_ma5" in working.columns
    assert "tone_z20" in working.columns
    assert "row_id" not in working.columns
    assert result.summary["dropped_columns"] == ["row_id"]
    assert result.summary["dropped_column_reasons"] == [{"column": "row_id", "reason": "id"}]


def test_feature_eng_reports_drop_reasons_for_id_text_and_exclusion(temp_projects_dir):
    df = pd.DataFrame(
        {
            "row_id": range(1, 13),
            "session_uuid": [f"user-{i:02d}" for i in range(12)],
            "note_text": [f"note {i // 2} with extra context" for i in range(12)],
            "labor_pct": [0.20 + i * 0.01 for i in range(12)],
            "wages": [100.5 + i * 5.25 for i in range(12)],
            "feature": [1.5 + i for i in range(12)],
        }
    )
    project = Project.create(name="Drop reasons", task_type="regression")
    project.set_raw_data(df)
    project.meta["schema"] = detect_schema(df)
    project.meta["target_column"] = "labor_pct"
    project.save_meta()

    result = Pipeline(project).run_stage("feature_eng")
    assert result.status == "success", result.error

    reasons = {item["column"]: item["reason"] for item in result.summary["dropped_column_reasons"]}
    assert reasons["row_id"] == "id"
    assert reasons["session_uuid"] == "id"
    assert reasons["note_text"] == "text"
    assert reasons["wages"] == "exclusion"
    assert set(result.summary["dropped_columns"]) == {"row_id", "session_uuid", "note_text", "wages"}


def test_feature_eng_adds_cyclical_datetime_features(temp_projects_dir):
    df = pd.DataFrame(
        {
            "order_ts": pd.date_range("2024-01-01 09:30:00", periods=24, freq="7h"),
            "order_date": pd.date_range("2024-01-01", periods=24, freq="D"),
            "amount": [10.0 + i for i in range(24)],
            "target": [100.0 + i * 2 for i in range(24)],
        }
    )
    project = Project.create(name="Cyclical FE", task_type="regression")
    project.set_raw_data(df)
    project.meta["schema"] = {
        "order_ts": "datetime",
        "order_date": "datetime",
        "amount": "numeric",
        "target": "numeric",
    }
    project.meta["target_column"] = "target"
    project.save_meta()

    result = Pipeline(project).run_stage("feature_eng")
    assert result.status == "success", result.error
    working = project.get_working_data()

    # sin/cos pairs present for day-of-week and month on both datetime columns
    for base in ("order_ts", "order_date"):
        for part in ("dow", "month"):
            for fn in ("sin", "cos"):
                col = f"{base}_{part}_{fn}"
                assert col in working.columns, f"missing {col}"
                assert working[col].between(-1.0, 1.0).all()

    # order_ts carries a time-of-day -> hour_* emitted
    assert "order_ts_hour_sin" in working.columns
    assert "order_ts_hour_cos" in working.columns

    # order_date is midnight-only -> no hour_* columns
    assert not any(c.startswith("order_date_hour_") for c in working.columns)

    # raw datetime columns dropped; parts still present (additive, not replaced)
    assert "order_ts" not in working.columns
    assert "order_ts_dayofweek" in working.columns


def test_daily_ops_pipeline_through_feature_select(temp_projects_dir):
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "sample" / "restaurant_daily_sample.csv"
    if not path.exists():
        pytest.skip("restaurant_daily_sample.csv not bundled")
    df = load_file(path)
    project = Project.create(name="Daily Select", task_type="regression")
    project.set_raw_data(df)
    project.meta["schema"] = detect_schema(df)
    project.meta["target_column"] = "labor_pct"
    project.save_meta()

    _run_until(project, "feature_select")
    working = project.get_working_data()
    assert "labor_pct" in working.columns
    feature_cols = [c for c in working.columns if c != "labor_pct"]
    assert all(pd.api.types.is_numeric_dtype(working[c]) for c in feature_cols)


def test_restaurant_pipeline_through_data_prep(temp_projects_dir):
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "sample" / "restaurant_daily_sample.csv"
    if not path.exists():
        pytest.skip("restaurant_daily_sample.csv not bundled")
    df = load_file(path)
    project = Project.create(name="Restaurant Prep", task_type="regression")
    project.set_raw_data(df)
    project.meta["schema"] = detect_schema(df)
    project.meta["target_column"] = "labor_pct"
    project.save_meta()

    _run_until(project, "data_prep")
    working = project.get_working_data()
    assert "labor_pct" in working.columns
    assert "wages" not in working.columns
    assert "net_sales" not in working.columns
    feature_cols = [c for c in working.columns if c != "labor_pct"]
    assert all(pd.api.types.is_numeric_dtype(working[c]) for c in feature_cols)


def test_clean_drops_rows_with_missing_target_and_never_imputes_it(temp_projects_dir):
    """Rows with a missing/blank label must be dropped, not median-imputed.
    Imputing the target fabricates labels and silently corrupts training."""
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5],
            "passed": [0, 1, None, 1, 0],  # row f1==3 is unlabeled
        }
    )
    project = Project.create(name="Missing target", task_type="classification")
    project.set_raw_data(df)
    project.meta["schema"] = {"f1": "numeric", "passed": "numeric"}
    project.meta["target_column"] = "passed"
    project.save_meta()

    # Run the clean stage directly — going through _run_until would drag in the
    # optional `profile` stage (ydata-profiling), which isn't installed in CI.
    result = Pipeline(project).run_stage("clean")
    assert result.status == "success", result.error
    working = project.get_working_data()

    # The unlabeled row is gone; no target value was fabricated.
    assert working["passed"].isna().sum() == 0
    assert set(working["passed"].unique()) <= {0, 1}
    assert 3 not in working["f1"].values  # the unlabeled row was dropped
    assert len(working) == 4


def test_data_prep_temporal_split_orders_by_time(temp_projects_dir):
    """Temporal split must train on the oldest rows and test on the most recent,
    ordered by a raw-data time column that never enters the feature matrix."""
    import numpy as np

    n = 50
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(
        {
            "trade_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    # feature_eng would have dropped the raw datetime; the working features do not
    # contain trade_date, but the raw data (used to order the split) still does.
    working = raw.drop(columns=["trade_date"])

    project = Project.create(name="Temporal split", task_type="regression")
    project.set_raw_data(raw)
    project.set_working_data(working)
    project.meta["schema"] = {"f1": "numeric", "f2": "numeric", "y": "numeric"}
    project.meta["target_column"] = "y"
    project.save_meta()

    result = Pipeline(project).run_stage(
        "data_prep",
        {"test_size": 0.2, "scaler": "none", "split_strategy": "temporal", "time_column": "trade_date"},
    )
    assert result.status == "success", result.error
    assert result.summary["split_strategy"] == "temporal"
    assert (result.summary["n_train"], result.summary["n_test"]) == (40, 10)

    from core.model_backend import load_prep_splits

    X_train, X_test, _, _ = load_prep_splits(project)
    assert "trade_date" not in X_train.columns  # date orders the split, is not a feature
    train_dates = raw["trade_date"].loc[X_train.index]
    test_dates = raw["trade_date"].loc[X_test.index]
    assert train_dates.max() < test_dates.min()  # no look-ahead


def test_data_prep_defaults_to_temporal_split(temp_projects_dir):
    """Leak-safety guardrail: with no explicit split_strategy the data_prep stage
    must default to a temporal (leak-safe) split, not a random one â€” a bare run
    should never look-ahead-leak into a forward-return target."""
    from stages.data_prep import default_settings

    n = 40
    working = pd.DataFrame(
        {
            "f1": range(n),
            "f2": [i * 0.5 for i in range(n)],
            "y": [i % 3 for i in range(n)],
        }
    )
    project = Project.create(name="Default split", task_type="regression")
    project.set_raw_data(working)
    project.set_working_data(working)
    project.meta["schema"] = {"f1": "numeric", "f2": "numeric", "y": "numeric"}
    project.meta["target_column"] = "y"
    project.save_meta()

    assert default_settings(project)["split_strategy"] == "temporal"

    # Run with no split_strategy override; the merged stage default must apply.
    result = Pipeline(project).run_stage("data_prep", {"scaler": "none"})
    assert result.status == "success", result.error
    assert result.summary["split_strategy"] == "temporal"


def test_walkthrough_wizard_defaults_to_temporal_split():
    """Leak-safety guardrail: the guided (UI) wizard must also default the
    train/test split to temporal, matching the CLI and data_prep defaults â€”
    otherwise a wizard-driven run silently writes a random (leaking) split."""
    from ui.walkthrough_page import WIZARD_OPTIONS

    split_field = next(
        f for f in WIZARD_OPTIONS["data_prep"] if f.get("setting") == "split_strategy"
    )
    assert split_field["default"] == "temporal"


def _prep_project_with_scaler(name: str):
    """A project that has run data_prep with a standard scaler, so a scaler.pkl
    exists in artifacts/prep/ and prep splits are persisted."""
    import numpy as np

    n = 120
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({"f1": rng.normal(10, 5, n), "f2": rng.normal(100, 20, n)})
    raw["y"] = 3 * raw["f1"] - 0.5 * raw["f2"] + rng.normal(0, 1, n)

    project = Project.create(name=name, task_type="regression")
    project.set_raw_data(raw)
    project.set_working_data(raw)
    project.meta["schema"] = {"f1": "numeric", "f2": "numeric", "y": "numeric"}
    project.meta["target_column"] = "y"
    project.save_meta()

    result = Pipeline(project).run_stage("data_prep", {"scaler": "standard"})
    assert result.status == "success", result.error
    assert (project.artifact_path("prep") / "scaler.pkl").exists()
    return project, raw


def test_deploy_sklearn_backend_prepends_scaler_and_predicts_on_raw(temp_projects_dir):
    """Sklearn backend trains on the data_prep *scaled* split, so the deploy
    pipeline must re-apply that scaler to raw input â€” otherwise inference feeds
    the model an unscaled representation it never trained on."""
    import joblib
    import numpy as np

    project, raw = _prep_project_with_scaler("Deploy Sklearn")
    pipeline = Pipeline(project)
    assert pipeline.run_stage("model", {"prefer_pycaret": False}).status == "success"
    deploy_result = pipeline.run_stage("deploy", {})
    assert deploy_result.status == "success", deploy_result.error
    assert deploy_result.summary["scaler_prepended"] is True

    full_pipeline = joblib.load(project.artifact_path("pipeline.pkl"))
    assert [name for name, _ in full_pipeline.steps] == ["scaler", "model"]

    # The exported pipeline consumes RAW columns; predictions should track y.
    new_raw = raw.drop(columns=["y"]).iloc[:10]
    preds = full_pipeline.predict(new_raw)
    corr = np.corrcoef(preds, raw["y"].iloc[:10])[0, 1]
    assert corr > 0.9, f"raw-input inference is off (corr={corr:.3f})"


def test_deploy_pycaret_backend_does_not_prepend_scaler(temp_projects_dir, monkeypatch):
    """PyCaret's saved model is a self-contained pipeline trained on the raw df.
    Prepending the data_prep scaler would double-transform (train-raw/eval-scaled
    mismatch), so deploy must NOT prepend it for the pycaret backend."""
    from sklearn.linear_model import LinearRegression

    import stages.deploy as deploy_stage
    from core.model_backend import set_model_backend

    project, raw = _prep_project_with_scaler("Deploy PyCaret")
    # Simulate the pycaret backend without importing pycaret: mark the backend and
    # hand deploy a self-contained model via the load hook.
    set_model_backend(project, "pycaret")
    fitted = LinearRegression().fit(raw[["f1", "f2"]], raw["y"])
    monkeypatch.setattr(deploy_stage, "load_saved_model", lambda _project: fitted)

    result = deploy_stage.run(project, raw, {})
    assert result.status == "success", result.error
    assert result.summary["backend"] == "pycaret"
    assert result.summary["scaler_prepended"] is False

    import joblib

    full_pipeline = joblib.load(project.artifact_path("pipeline.pkl"))
    assert [name for name, _ in full_pipeline.steps] == ["model"]


def _synthetic_select_frame(n: int = 300):
    """Two *independently* informative features plus three noise features. The
    label depends on both drivers, so neither good feature is redundant with the
    other (permutation importance rewards both)."""
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, n)  # independent driver 1
    b = rng.normal(0, 1, n)  # independent driver 2
    y = (a + b > 0).astype(int)
    return pd.DataFrame(
        {
            "good1": a + rng.normal(0, 0.05, n),
            "good2": b + rng.normal(0, 0.05, n),
            "noise1": rng.normal(0, 1, n),
            "noise2": rng.normal(0, 1, n),
            "noise3": rng.normal(0, 1, n),
            "target": y,
        }
    )


def test_feature_select_mutual_info_keeps_top_k_informative(temp_projects_dir):
    import stages.feature_select as fs

    df = _synthetic_select_frame()
    project = Project.create(name="MI select", task_type="classification")
    project.set_raw_data(df)
    project.meta["target_column"] = "target"
    project.save_meta()

    result = fs.run(project, df, {"methods": ["mutual_info"], "mi_top_k": 2})
    assert result.status == "success", result.error

    cols = set(result.output_df.columns)
    assert "target" in cols  # target always retained
    features = cols - {"target"}
    assert len(features) == 2  # only mi_top_k features survive
    assert features == {"good1", "good2"}  # the informative pair, not the noise


def test_feature_select_permutation_keeps_top_k_informative(temp_projects_dir):
    import stages.feature_select as fs

    df = _synthetic_select_frame()
    project = Project.create(name="Perm select", task_type="classification")
    project.set_raw_data(df)
    project.meta["target_column"] = "target"
    project.save_meta()

    result = fs.run(project, df, {"methods": ["permutation"], "permutation_top_k": 2})
    assert result.status == "success", result.error

    cols = set(result.output_df.columns)
    assert "target" in cols  # target always retained
    features = cols - {"target"}
    assert len(features) == 2  # only permutation_top_k features survive
    assert features == {"good1", "good2"}  # the informative pair, not the noise


def _lag_frame():
    """Two ordered groups. Within each group val increases by seq; a lag1 of val
    must therefore surface the PRIOR row's value, never the current row's."""
    return pd.DataFrame(
        {
            "grp": ["A", "A", "A", "B", "B"],
            "seq": [1, 2, 3, 1, 2],
            "val": [10.0, 20.0, 30.0, 40.0, 50.0],
            "target": [0, 1, 0, 1, 0],
        }
    )


def _lag_project(name: str, df: pd.DataFrame):
    project = Project.create(name=name, task_type="classification")
    project.set_raw_data(df)
    project.meta["schema"] = {
        "grp": "categorical",
        "seq": "numeric",
        "val": "numeric",
        "target": "numeric",
    }
    project.meta["target_column"] = "target"
    project.save_meta()
    return project


def test_feature_eng_lag_pulls_prior_group_row_value(temp_projects_dir):
    """lag1 must read the strictly-earlier row within the group (leak-safe): the
    current row's own value is never its own lag, and lags never bleed across groups."""
    import stages.feature_eng as fe

    df = _lag_frame()
    project = _lag_project("Lag leak-safe", df)

    result = fe.run(
        project,
        df,
        {"lag_features": {"group_key": "grp", "order_key": "seq", "columns": ["val"], "lags": [1]}},
    )
    assert result.status == "success", result.error
    out = result.output_df
    assert "val_lag1" in out.columns
    assert result.summary["lag_features"] == ["val_lag1"]

    by_key = {(r.grp, int(r.seq)): r for r in out.itertuples()}
    # A@seq2 sees A@seq1's value (10), not its own (20).
    assert by_key[("A", 2)].val_lag1 == 10.0
    assert by_key[("A", 3)].val_lag1 == 20.0
    assert by_key[("B", 2)].val_lag1 == 40.0
    # First row of each group has no predecessor -> NaN; never bleeds across groups.
    assert pd.isna(by_key[("A", 1)].val_lag1)
    assert pd.isna(by_key[("B", 1)].val_lag1)


def test_feature_eng_lag_refuses_non_positive_lag(temp_projects_dir):
    """A lag of 0 (or below) reads the current row's own value at verdict time —
    the deterministic leakage guard must refuse it outright."""
    import stages.feature_eng as fe

    df = _lag_frame()
    project = _lag_project("Lag zero", df)

    with pytest.raises(ValueError, match="leakage|>= 1"):
        fe.run(
            project,
            df,
            {"lag_features": {"group_key": "grp", "order_key": "seq", "columns": ["val"], "lags": [0]}},
        )


def test_feature_eng_attempt_index_withheld_without_target_proxy_optin(temp_projects_dir):
    """attempt_index is temporally safe but a strong target proxy: it must stay off
    unless allow_target_proxy=True, and then equal the group cumcount."""
    import stages.feature_eng as fe

    df = _lag_frame()

    # Un-opted -> refused.
    project = _lag_project("Attempt withheld", df)
    with pytest.raises(ValueError, match="allow_target_proxy"):
        fe.run(
            project,
            df,
            {
                "lag_features": {
                    "group_key": "grp",
                    "order_key": "seq",
                    "columns": ["val"],
                    "attempt_index": True,
                    "allow_target_proxy": False,
                }
            },
        )

    # Opted in -> present and equal to the strictly-prior-row count within each group.
    project2 = _lag_project("Attempt opted", df)
    result = fe.run(
        project2,
        df,
        {
            "lag_features": {
                "group_key": "grp",
                "order_key": "seq",
                "columns": ["val"],
                "attempt_index": True,
                "allow_target_proxy": True,
            }
        },
    )
    assert result.status == "success", result.error
    out = result.output_df
    assert "attempt_index" in out.columns
    by_key = {(r.grp, int(r.seq)): int(r.attempt_index) for r in out.itertuples()}
    assert by_key[("A", 1)] == 0
    assert by_key[("A", 2)] == 1
    assert by_key[("A", 3)] == 2
    assert by_key[("B", 1)] == 0
    assert by_key[("B", 2)] == 1


def test_feature_eng_drops_verdict_proxies_for_passed_target(temp_projects_dir):
    """With target `passed`, the attempt-count / verdict-readout columns are
    target proxies and must be dropped by feature_eng's leakage exclusions."""
    import stages.feature_eng as fe

    proxies = [
        "n_post_ok",
        "n_blocked_post",
        "n_escalated",
        "min_score",
        "threshold",
        "n_pre",
        "n_model_call",
        "n_post",
    ]
    data = {p: [float(i) for i in range(6)] for p in proxies}
    data["legit_feature"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    data["passed"] = [0, 1, 0, 1, 0, 1]
    df = pd.DataFrame(data)

    project = Project.create(name="Verdict proxies", task_type="classification")
    project.set_raw_data(df)
    project.meta["schema"] = {c: "numeric" for c in df.columns}
    project.meta["target_column"] = "passed"
    project.save_meta()

    result = fe.run(project, df, fe.default_settings(project))
    assert result.status == "success", result.error
    out = result.output_df
    for col in proxies:
        assert col not in out.columns, f"{col} should be excluded as a verdict proxy"
    assert "legit_feature" in out.columns
    assert "passed" in out.columns


def test_suggest_target_for_restaurant_daily(temp_projects_dir):
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "sample" / "restaurant_daily_sample.csv"
    if not path.exists():
        pytest.skip("restaurant_daily_sample.csv not bundled")
    df = load_file(path)
    schema = detect_schema(df)
    assert schema["trading_day_date"] == "datetime"
    suggestion = suggest_target(df, schema, task_type="regression")
    assert suggestion["target_column"] == "labor_pct"


def test_suggest_target_classification_accepts_integer_coded_label():
    """Titanic-shaped data: the label is 0/1 numeric, not a string category."""
    df = pd.DataFrame({
        "PassengerId": range(1, 41),
        "Survived": [0, 1] * 20,
        "Pclass": [1, 2, 3, 3] * 10,
        "Sex": ["male", "female"] * 20,
        "Fare": [7.25 + i for i in range(40)],
    })
    schema = {"PassengerId": "id", "Survived": "numeric", "Pclass": "numeric",
              "Sex": "categorical", "Fare": "numeric"}
    for task in ("classification", None):
        suggestion = suggest_target(df, schema, task_type=task)
        assert suggestion["target_column"] == "Survived", (task, suggestion)
        assert suggestion["task_type"] == "classification"
