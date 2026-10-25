"""Safe persistence for application-trained scikit-learn pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
from sklearn.pipeline import Pipeline


class ModelArtifactError(ValueError):
    """A managed model artifact is missing, invalid, or outside its storage root."""


def artifact_path(model_folder: str, filename: str) -> Path:
    """Resolve a generated artifact name strictly within the configured model folder."""
    root = Path(model_folder).resolve()
    candidate_name = Path(filename)
    if candidate_name.name != filename or candidate_name.suffix != ".joblib":
        raise ModelArtifactError("Model artifact reference is invalid.")
    candidate = (root / filename).resolve()
    if candidate.parent != root:
        raise ModelArtifactError("Model artifact path is outside the managed model directory.")
    return candidate


def save_pipeline(pipeline: Pipeline, model_folder: str) -> str:
    """Save one fitted pipeline to a unique, exclusively-created filename."""
    root = Path(model_folder).resolve()
    root.mkdir(parents=True, exist_ok=True)
    filename = f"model_{uuid4().hex}.joblib"
    destination = artifact_path(str(root), filename)
    try:
        with destination.open("xb") as artifact_file:
            joblib.dump(pipeline, artifact_file)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return filename


def load_pipeline(model_folder: str, filename: str) -> Pipeline:
    """Load only a recorded application-managed pipeline artifact."""
    path = artifact_path(model_folder, filename)
    if not path.is_file():
        raise ModelArtifactError("The saved model artifact is missing. Retrain the model before scoring.")
    try:
        pipeline: Any = joblib.load(path)
    except Exception as exc:
        raise ModelArtifactError("The saved model artifact could not be loaded. Retrain the model before scoring.") from exc
    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict"):
        raise ModelArtifactError("The stored artifact is not a supported application model pipeline.")
    return pipeline


def remove_artifact(model_folder: str, filename: str) -> None:
    """Remove one managed artifact after validating its path."""
    artifact_path(model_folder, filename).unlink(missing_ok=True)
