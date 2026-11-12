"""
stages/deploy.py

Final deploy stage: package the full preprocessing + model pipeline,
generate inference template and requirements.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline

from core.library_catalog import get_enabled_libraries
from core.model_backend import get_model_backend, load_saved_model
from core.pipeline import StageResult
from core.project import Project


def default_settings(project: Project) -> dict[str, Any]:
    return {}


def run(project: Project, df: pd.DataFrame | None, settings: dict[str, Any]) -> StageResult:
    """Build a full inference pipeline (prep + model) and export artifacts."""
    target = project.meta.get("target_column")
    if not target:
        return StageResult(status="error", error="No target column set.")

    exports_dir = project.root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []

    # --- Build combined pipeline.pkl for raw-input inference ---
    # Scale-representation correctness: the
    # sklearn backend trains the model on the data_prep *scaled* split, so its
    # inference pipeline must re-apply that same scaler to raw input. The PyCaret
    # backend's saved model is already a self-contained pipeline trained on the
    # raw df (it does its own preprocessing), so prepending the data_prep scaler
    # would feed it scaled features it never saw in training — a train-raw /
    # eval-scaled mismatch. Only prepend the scaler for the sklearn backend.
    backend = get_model_backend(project)
    prep_dir = project.artifact_path("prep")
    scaler = None
    scaler_path = prep_dir / "scaler.pkl"
    if backend == "sklearn" and scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    try:
        trained_model = load_saved_model(project)
    except Exception as e:
        return StageResult(status="error", error=f"Could not load best model: {e}")

    # Create a simple sklearn-style pipeline for inference
    steps = []
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("model", trained_model))

    full_pipeline = SklearnPipeline(steps)
    pipeline_path = project.artifact_path("pipeline.pkl")
    joblib.dump(full_pipeline, pipeline_path)
    artifacts.append(pipeline_path)

    # --- Generate inference.py ---
    inference_code = f'''"""
Auto-generated inference script for {project.meta.get("name", "project")}.
Usage:
    from inference import predict
    predictions = predict(new_data_df)
"""
import joblib
import pandas as pd

# Load the full preprocessing + model pipeline
_pipeline = joblib.load("pipeline.pkl")


def predict(new_data: pd.DataFrame) -> pd.Series:
    """Run inference on new data (must have the same columns as training data)."""
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data)
    return pd.Series(_pipeline.predict(new_data), name="prediction")


if __name__ == "__main__":
    # Example usage
    print("Inference template ready. Replace with your data loading logic.")
'''

    inference_path = exports_dir / "inference.py"
    inference_path.write_text(inference_code)
    artifacts.append(inference_path)

    # --- Generate requirements.txt from enabled libraries ---
    enabled = get_enabled_libraries(project)
    # Add core ones that are always needed
    base_reqs = ["pandas", "joblib", "scikit-learn"]
    all_reqs = sorted(set(base_reqs + enabled))
    # Map some names
    req_map = {
        "sklearn": "scikit-learn",
        "ydata_profiling": "ydata-profiling",
        "category_encoders": "category-encoders",
        "feature_engine": "feature-engine",
        "featuretools": "featuretools",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "shap": "shap",
        "pycaret": "pycaret",  # if user wants to reproduce exactly
    }
    req_lines = []
    for lib in all_reqs:
        req_lines.append(req_map.get(lib, lib))

    # Add explicit versions if we want, but for simplicity use names
    req_text = "\n".join(sorted(set(req_lines))) + "\n"
    req_path = exports_dir / "requirements.txt"
    req_path.write_text(req_text)
    artifacts.append(req_path)

    summary = {
        "pipeline_path": str(pipeline_path.relative_to(project.root)),
        "inference_path": str(inference_path.relative_to(project.root)),
        "requirements_path": str(req_path.relative_to(project.root)),
        "enabled_libraries": enabled,
        "backend": backend,
        "scaler_prepended": scaler is not None,
    }

    return StageResult(
        status="success",
        output_df=df,
        artifacts=artifacts,
        summary=summary,
    )


def code_template(settings: dict[str, Any]) -> str:
    """Code snippet for the notebook export."""
    return """# Deploy / Export
# The full pipeline was saved as pipeline.pkl during the deploy stage.
# See exports/inference.py and exports/requirements.txt for standalone usage.
import joblib
pipeline = joblib.load("pipeline.pkl")
# predictions = pipeline.predict(new_data)
print("Full pipeline loaded for inference.")
"""


def render_ui(
    project: Project, settings: dict[str, Any], last_result: StageResult | None
) -> dict[str, Any]:
    return settings
