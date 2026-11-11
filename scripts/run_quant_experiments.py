"""Run Mira quant research experiments A/B/C with leakage-safe defaults.

Experiments:
  A — Cross-sectional top-decile classification (fwd_ret_5d)
  B — Macro-augmented regression vs technical-only baseline
  C — Forward-return horizon sweep (5d / 10d / 20d)

Usage:
    python scripts/run_quant_experiments.py
    python scripts/run_quant_experiments.py --experiments A C
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_stock_panel import (  # noqa: E402
    DEFAULT_DB,
    compute_ticker_features,
    load_bars,
    load_top_tickers,
)
from build_stock_panel_full import _macro_by_date  # noqa: E402
from core.data_loader import detect_schema  # noqa: E402
from core.ingest import ingest_dataframe  # noqa: E402
from core.model_backend import load_prep_splits, load_saved_model  # noqa: E402
from core.pipeline import STAGE_ORDER, Pipeline  # noqa: E402
from core.project import Project  # noqa: E402
from core.stage_settings import get_stage_settings, save_stage_settings  # noqa: E402

PANEL_PATH = ROOT / "data" / "sample" / "stock_daily_features_500.csv"
EXP_DIR = ROOT / "data" / "experiments"
RESULTS_PATH = EXP_DIR / "experiment_results.json"

TECH_FEATURES_BASE = [
    "ret_1d",
    "close_vs_sma20",
    "sma20_vs_sma50",
    "rsi_14",
    "atr_pct",
    "vol_20d_ann",
    "volume_z20",
    "hl_spread_bps",
    "dist_52w_high",
    "dist_52w_low",
    "gap_open",
]

MACRO_COLS = [
    "macro_vixcls",
    "macro_dgs10",
    "macro_t10y2y",
    "macro_dff",
    "macro_dexuseu",
    "macro_unrate",
    "macro_cpiaucsl",
]

HORIZON_DROP = {
    5: ["ret_5d"],
    10: ["ret_5d", "ret_10d"],
    20: ["ret_5d", "ret_10d"],
}


def _tech_features(horizon: int = 5) -> list[str]:
    drop = set(HORIZON_DROP.get(horizon, []))
    feats = [c for c in TECH_FEATURES_BASE if c not in drop]
    if horizon < 10:
        feats.insert(1, "ret_5d")
    if horizon < 20:
        feats.insert(2 if horizon < 10 else 1, "ret_10d")
    return feats


def _schema_override(
    df: pd.DataFrame,
    *,
    target: str,
    feature_cols: list[str],
    task: str,
) -> dict[str, str]:
    schema = detect_schema(df)
    for col in ("ticker",):
        if col in schema:
            schema[col] = "id"
    for col in ("sector",):
        if col in schema:
            schema[col] = "text"
    if "trade_date" in schema:
        schema["trade_date"] = "datetime"
    for col in feature_cols:
        if col in schema:
            schema[col] = "numeric"
    if target in schema:
        schema[target] = "categorical" if task == "classification" else "numeric"
    for col in ("fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d"):
        if col in df.columns and col != target and col in schema:
            schema[col] = "numeric"
    return schema


def _configure_project(
    project: Project,
    *,
    feature_cols: list[str],
    drop_extra: list[str] | None = None,
) -> None:
    drop_cols = ["sector", "ticker", "trade_date", *(drop_extra or [])]
    save_stage_settings(
        project,
        "clean",
        {
            "drop_columns": ["sector"],
            # IQR clip treats binary target 1s as outliers → all-zero labels.
            "outlier_method": "none",
        },
    )
    save_stage_settings(
        project,
        "feature_eng",
        {
            "drop_columns": drop_cols,
            "date_features": [],
            "encoding": {},
            "interactions": [],
            "featuretools": False,
        },
    )
    save_stage_settings(
        project,
        "data_prep",
        {
            "split_strategy": "temporal",
            "time_column": "trade_date",
            "test_size": 0.2,
            "stratify": False,
            "scaler": "standard",
        },
    )
    save_stage_settings(
        project,
        "model",
        {"n_models": 3, "cv_folds": 2, "prefer_pycaret": False},
    )


def _run_pipeline(project: Project, *, from_stage: str = "clean", through: str = "evaluate") -> dict[str, Any]:
    pipeline = Pipeline(project)
    stage_results: dict[str, Any] = {}
    start = STAGE_ORDER.index(from_stage)
    stop = STAGE_ORDER.index(through)

    for name in STAGE_ORDER[start : stop + 1]:
        settings = get_stage_settings(project, name)
        result = pipeline.run_stage(name, settings)
        stage_results[name] = {
            "status": result.status,
            "summary": result.summary,
            "error": result.error,
        }
        print(f"  {name:16} {result.status.upper()}")
        if result.status == "error":
            raise RuntimeError(f"Stage {name} failed: {result.error}")
    return stage_results


def _regression_baselines(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    zeros = np.zeros(len(y_test))
    mean_pred = np.full(len(y_test), float(y_train.mean()))
    return {
        "r2_predict_zero": float(r2_score(y_test, zeros)),
        "r2_predict_mean": float(r2_score(y_test, mean_pred)),
        "mae_predict_zero": float(mean_absolute_error(y_test, zeros)),
        "mae_predict_mean": float(mean_absolute_error(y_test, mean_pred)),
    }


def _classification_baselines(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    majority = int(y_train.mode().iloc[0]) if not y_train.mode().empty else 0
    maj_pred = np.full(len(y_test), majority)
    return {
        "accuracy_majority_class": float(accuracy_score(y_test, maj_pred)),
        "f1_majority_class": float(f1_score(y_test, maj_pred, zero_division=0)),
        "balanced_accuracy_random": 0.5,
        "prevalence_positive": float(y_test.mean()),
    }


def _cross_sectional_ic(
    raw_df: pd.DataFrame,
    test_index: pd.Index,
    y_pred: np.ndarray,
    *,
    return_col: str = "fwd_ret_5d",
) -> dict[str, float]:
    meta = raw_df.loc[test_index, ["trade_date", return_col]].copy()
    meta["y_pred"] = y_pred
    meta = meta.dropna(subset=[return_col, "y_pred"])
    if meta.empty:
        return {"mean_daily_ic": np.nan, "ic_tstat_approx": np.nan, "n_days": 0}

    daily_ics: list[float] = []
    for _, grp in meta.groupby("trade_date"):
        if len(grp) < 20:
            continue
        rho, _ = spearmanr(grp["y_pred"], grp[return_col])
        if np.isfinite(rho):
            daily_ics.append(float(rho))

    if not daily_ics:
        return {"mean_daily_ic": np.nan, "ic_tstat_approx": np.nan, "n_days": 0}

    arr = np.array(daily_ics)
    mean_ic = float(arr.mean())
    std_ic = float(arr.std(ddof=1)) if len(arr) > 1 else np.nan
    tstat = mean_ic / (std_ic / np.sqrt(len(arr))) if std_ic and std_ic > 0 else np.nan
    return {
        "mean_daily_ic": mean_ic,
        "ic_std": std_ic,
        "ic_tstat_approx": float(tstat) if np.isfinite(tstat) else np.nan,
        "n_days": len(arr),
    }


def _collect_metrics(
    project: Project,
    raw_df: pd.DataFrame,
    *,
    task: str,
    return_col: str | None = None,
) -> dict[str, Any]:
    metrics_path = project.artifact_path("eval") / "metrics.json"
    model_metrics: dict[str, Any] = {}
    if metrics_path.exists():
        model_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    splits = load_prep_splits(project)
    if splits is None:
        return {"model": model_metrics, "error": "no prep splits"}

    X_train, X_test, y_train, y_test = splits
    model = load_saved_model(project)
    y_pred = model.predict(X_test)

    out: dict[str, Any] = {"model": model_metrics, "n_test": int(len(y_test))}
    if task == "regression":
        out["baselines"] = _regression_baselines(y_train, y_test)
    else:
        out["baselines"] = _classification_baselines(y_train, y_test)

    ret_col = return_col or project.meta.get("target_column")
    if ret_col in raw_df.columns:
        out["cross_sectional"] = _cross_sectional_ic(raw_df, y_test.index, y_pred, return_col=ret_col)

    if task == "classification" and return_col in raw_df.columns:
        pos_mask = y_pred == 1
        if pos_mask.any():
            actual_top = raw_df.loc[y_test.index, return_col].groupby(
                raw_df.loc[y_test.index, "trade_date"]
            ).rank(pct=True) >= 0.9
            out["precision_top_decile"] = float((actual_top & pos_mask).sum() / pos_mask.sum())
        else:
            out["precision_top_decile"] = 0.0

    return out


def run_mira_experiment(
    df: pd.DataFrame,
    *,
    name: str,
    target: str,
    task: str,
    feature_cols: list[str],
    drop_extra: list[str] | None = None,
    return_col: str | None = None,
) -> dict[str, Any]:
    print(f"\n{'=' * 60}\nExperiment: {name}\n{'=' * 60}")
    t0 = time.perf_counter()

    project = Project.create(name=name, task_type=task)
    ingest_dataframe(project, df, source_name=f"{name}.csv", auto_profile=False)

    schema = _schema_override(df, target=target, feature_cols=feature_cols, task=task)
    project.meta["schema"] = schema
    project.meta["target_column"] = target
    project.meta["task_type"] = task
    project.save_meta()

    _configure_project(project, feature_cols=feature_cols, drop_extra=drop_extra)

    print(f"  rows={len(df):,}  target={target}  task={task}  features={len(feature_cols)}")
    print(f"  schema override: {sum(1 for v in schema.values() if v == 'numeric')} numeric cols")

    stage_results = _run_pipeline(project)
    metrics = _collect_metrics(project, df, task=task, return_col=return_col or target)

    elapsed = time.perf_counter() - t0
    result = {
        "name": name,
        "target": target,
        "task": task,
        "features": feature_cols,
        "n_rows": len(df),
        "project_id": project.id,
        "project_path": str(project.root),
        "stages": stage_results,
        "metrics": metrics,
        "elapsed_sec": round(elapsed, 1),
    }
    print(f"  done in {elapsed:.1f}s — R²/acc={metrics.get('model', {})}")
    return result


def prepare_experiment_a(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["top_decile"] = df.groupby("trade_date")["fwd_ret_5d"].transform(
        lambda s: (s.rank(method="first", pct=True) >= 0.9).astype(int)
    )
    return df


def prepare_experiment_b(panel: pd.DataFrame, db_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    try:
        macro = _macro_by_date(con, panel["trade_date"])
    finally:
        con.close()
    return panel.merge(macro, on="trade_date", how="left")


def build_horizon_panel(db_path: Path, *, n_tickers: int = 500) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    try:
        tickers = load_top_tickers(con, n_tickers)
        frames: list[pd.DataFrame] = []
        for ticker, sector, symbol_id in tickers:
            bars = load_bars(con, symbol_id)
            if len(bars) < 252:
                continue
            feat = compute_ticker_features(bars, fwd_days=5)
            close = feat["close"].astype(float)
            feat["fwd_ret_10d"] = close.shift(-10) / close - 1.0
            feat["fwd_ret_20d"] = close.shift(-20) / close - 1.0
            feat["ticker"] = ticker
            feat["sector"] = sector
            feat["trade_date"] = pd.to_datetime(feat["trade_date"]).dt.strftime("%Y-%m-%d")
            cols = [
                "trade_date", "ticker", "sector",
                "ret_1d", "ret_5d", "ret_10d",
                "close_vs_sma20", "sma20_vs_sma50", "rsi_14", "atr_pct",
                "vol_20d_ann", "volume_z20", "hl_spread_bps",
                "dist_52w_high", "dist_52w_low", "gap_open",
                "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d",
            ]
            out = feat[cols].dropna(subset=["ret_1d", "ret_5d", "rsi_14", "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d"])
            if not out.empty:
                frames.append(out)
        if not frames:
            raise RuntimeError("No horizon panel rows produced")
        return pd.concat(frames, ignore_index=True).sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    finally:
        con.close()


def run_experiment_a(panel: pd.DataFrame) -> dict[str, Any]:
    df = prepare_experiment_a(panel)
    out_path = EXP_DIR / "exp_a_top_decile.csv"
    df.to_csv(out_path, index=False)
    feats = _tech_features(5)
    return run_mira_experiment(
        df,
        name="ExpA_TopDecile_Classification",
        target="top_decile",
        task="classification",
        feature_cols=feats,
        drop_extra=["fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d"],
        return_col="fwd_ret_5d",
    )


def run_experiment_b(panel: pd.DataFrame, db_path: Path) -> list[dict[str, Any]]:
    macro_panel = prepare_experiment_b(panel, db_path)
    macro_path = EXP_DIR / "exp_b_macro_panel.csv"
    macro_panel.to_csv(macro_path, index=False)

    feats_tech = _tech_features(5)
    results = []

    results.append(
        run_mira_experiment(
            macro_panel,
            name="ExpB_TechnicalOnly_5d",
            target="fwd_ret_5d",
            task="regression",
            feature_cols=feats_tech,
            drop_extra=["top_decile", "fwd_ret_10d", "fwd_ret_20d", *MACRO_COLS],
        )
    )

    feats_macro = feats_tech + MACRO_COLS
    results.append(
        run_mira_experiment(
            macro_panel,
            name="ExpB_TechnicalPlusMacro_5d",
            target="fwd_ret_5d",
            task="regression",
            feature_cols=feats_macro,
            drop_extra=["top_decile", "fwd_ret_10d", "fwd_ret_20d"],
        )
    )
    return results


def run_experiment_c(horizon_panel: pd.DataFrame) -> list[dict[str, Any]]:
    horizon_path = EXP_DIR / "exp_c_horizon_panel.csv"
    horizon_panel.to_csv(horizon_path, index=False)

    results = []
    for horizon, target in [(5, "fwd_ret_5d"), (10, "fwd_ret_10d"), (20, "fwd_ret_20d")]:
        drop_extra = [t for t in ("fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d") if t != target]
        drop_extra.append("top_decile")
        results.append(
            run_mira_experiment(
                horizon_panel.dropna(subset=[target]),
                name=f"ExpC_Horizon_{horizon}d",
                target=target,
                task="regression",
                feature_cols=_tech_features(horizon),
                drop_extra=drop_extra,
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quant experiments A/B/C")
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--rebuild-horizon", action="store_true", help="Rebuild C panel from market.db")
    args = parser.parse_args()

    if not args.panel.exists():
        print(f"ERROR: panel not found: {args.panel}", file=sys.stderr)
        return 1

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {"experiments": {}, "meta": {"panel": str(args.panel)}}

    print(f"Loading panel: {args.panel}")
    panel = pd.read_csv(args.panel)
    print(f"  {len(panel):,} rows, {panel['ticker'].nunique()} tickers")

    if "A" in args.experiments:
        all_results["experiments"]["A"] = run_experiment_a(panel)

    if "B" in args.experiments:
        if not args.db.exists():
            print(f"ERROR: market.db not found: {args.db}", file=sys.stderr)
            return 1
        all_results["experiments"]["B"] = run_experiment_b(panel, args.db)

    if "C" in args.experiments:
        horizon_path = EXP_DIR / "exp_c_horizon_panel.csv"
        if args.rebuild_horizon or not horizon_path.exists():
            print("Building horizon panel from market.db …")
            horizon_panel = build_horizon_panel(args.db)
            horizon_panel.to_csv(horizon_path, index=False)
            print(f"  wrote {horizon_path} ({len(horizon_panel):,} rows)")
        else:
            horizon_panel = pd.read_csv(horizon_path)
        all_results["experiments"]["C"] = run_experiment_c(horizon_panel)

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())