"""
stages/explain.py

Model explainability with SHAP (primary) and permutation importance fallback.

Saves summary.png, bar.png, etc. under artifacts/shap/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.model_backend import load_saved_model, raw_estimator
from core.pipeline import StageResult
from core.project import Project


def default_settings(project: Project) -> dict[str, Any]:
    """Default settings for explain stage."""
    return {
        "sample_size": 500,  # max rows to use for SHAP
        "random_state": 42,
    }


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    """Generate SHAP (or fallback) explanations for the best model."""
    if df is None or len(df) < 5:
        return StageResult(status="error", error="Insufficient data for explainability.")

    target = project.meta.get("target_column")
    if not target or target not in df.columns:
        return StageResult(status="error", error="Target column not set or missing.")

    shap_dir = project.artifact_path("shap")
    shap_dir.mkdir(parents=True, exist_ok=True)

    # The model is trained on the data_prep feature representation (scaled when
    # a scaler was fit). Explain on that same representation so SHAP /
    # permutation importance see the inputs the model actually consumes.
    # Index is preserved so the target alignment below still holds.
    X = df.drop(columns=[target])
    scaler_path = project.artifact_path("prep") / "scaler.pkl"
    if scaler_path.exists():
        import pickle

        with open(scaler_path, "rb") as handle:
            scaler = pickle.load(handle)
        try:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        except Exception:
            pass

    sample_size = min(int(settings.get("sample_size", 500)), len(X))
    X_sample = X.sample(n=sample_size, random_state=settings.get("random_state", 42))

    artifacts: list[Path] = []
    summary: dict[str, Any] = {"method": "unknown"}

    # Try SHAP first
    try:
        import shap

        model = load_saved_model(project)
        raw_model = raw_estimator(model)

        explainer = shap.Explainer(raw_model, X_sample)
        sv = explainer(X_sample)

        # Beeswarm summary
        fig = shap.plots.beeswarm(sv, show=False)
        summary_png = shap_dir / "summary.png"
        fig.figure.savefig(summary_png, bbox_inches="tight")
        artifacts.append(summary_png)

        # Bar plot
        fig2 = shap.plots.bar(sv, show=False)
        bar_png = shap_dir / "bar.png"
        fig2.figure.savefig(bar_png, bbox_inches="tight")
        artifacts.append(bar_png)

        summary = {
            "method": "shap",
            "n_samples": len(X_sample),
            "plots": ["summary.png", "bar.png"],
        }

        return StageResult(
            status="success",
            output_df=df,
            artifacts=artifacts,
            summary=summary,
        )

    except Exception as shap_err:
        # Fallback to permutation importance
        try:
            import matplotlib.pyplot as plt
            from sklearn.inspection import permutation_importance

            model = load_saved_model(project)

            # Compute permutation importance (on a small sample for speed)
            result = permutation_importance(
                model,
                X_sample,
                df[target].loc[X_sample.index],
                n_repeats=5,
                random_state=42,
                n_jobs=-1,
            )

            # Create bar plot
            sorted_idx = result.importances_mean.argsort()
            fig, ax = plt.subplots(figsize=(8, max(4, len(X.columns) * 0.3)))
            ax.boxplot(
                result.importances[sorted_idx].T,
                vert=False,
            )
            ax.set_yticklabels(X.columns[sorted_idx])
            ax.set_title("Permutation Importance (fallback)")
            ax.set_xlabel("Importance")

            perm_png = shap_dir / "permutation_importance.png"
            fig.tight_layout()
            fig.savefig(perm_png, bbox_inches="tight")
            plt.close(fig)

            artifacts.append(perm_png)

            summary = {
                "method": "permutation_importance",
                "shap_error": str(shap_err),
                "n_samples": len(X_sample),
                "plots": ["permutation_importance.png"],
            }

            return StageResult(
                status="success",
                output_df=df,
                artifacts=artifacts,
                summary=summary,
            )

        except Exception as fallback_err:
            return StageResult(
                status="error",
                error=f"SHAP failed ({shap_err}) and permutation fallback also failed: {fallback_err}",
            )


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    """Placeholder for explain settings."""
    return settings
