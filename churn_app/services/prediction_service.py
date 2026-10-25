"""Schema-checked prediction, probability, and operational risk calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .ml_service import _is_numeric


class PredictionValidationError(ValueError):
    """The selected model and scoring dataset are not safely compatible."""


def _risk_category(probability: float | None, high_threshold: float, medium_threshold: float) -> str:
    """Convert a valid churn probability into an operational, non-guaranteed band."""
    if probability is None:
        return "unavailable"
    if probability >= high_threshold:
        return "high"
    if probability >= medium_threshold:
        return "medium"
    return "low"


def score_dataset(
    frame: pd.DataFrame,
    dataset_configuration: dict[str, Any] | None,
    model_version: Any,
    pipeline: Pipeline,
    high_threshold: float,
    medium_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a dataset against saved feature metadata and return row predictions."""
    config = dataset_configuration or {}
    if not config.get("features"):
        raise PredictionValidationError("Configure the scoring dataset's feature columns before generating predictions.")
    expected_features = list(model_version.feature_columns)
    if list(config["features"]) != expected_features:
        raise PredictionValidationError(
            "The scoring dataset feature columns or order do not match this model. Configure the same feature columns in the same order."
        )
    missing_columns = [name for name in expected_features if name not in frame.columns]
    if missing_columns:
        raise PredictionValidationError(f"The scoring CSV is missing required model features: {', '.join(missing_columns)}.")
    if model_version.date_feature and config.get("date") != model_version.date_feature:
        raise PredictionValidationError("The scoring dataset must map the same date feature role used by this model.")

    for name in expected_features:
        expected_kind = model_version.feature_kinds[name]
        if expected_kind == "date":
            parsed_dates = pd.to_datetime(frame[name], errors="coerce", utc=True)
            if parsed_dates.isna().any():
                raise PredictionValidationError(f"Date feature '{name}' contains missing or unreadable values.")
        elif expected_kind == "numeric" and not _is_numeric(frame[name]):
            raise PredictionValidationError(f"Feature '{name}' must be numeric to match the saved model schema.")
        elif expected_kind == "categorical" and _is_numeric(frame[name]):
            raise PredictionValidationError(f"Feature '{name}' must be categorical to match the saved model schema.")

    features = frame.loc[:, expected_features].copy()
    for name, kind in model_version.feature_kinds.items():
        if kind == "numeric":
            features[name] = features[name].replace([np.inf, -np.inf], np.nan)
        elif kind == "categorical":
            features[name] = features[name].map(lambda value: str(value) if pd.notna(value) else np.nan)

    try:
        predicted = np.asarray(pipeline.predict(features), dtype=int)
    except Exception as exc:
        raise PredictionValidationError(f"The saved model could not score this dataset: {exc}") from exc

    probabilities: np.ndarray | None = None
    if hasattr(pipeline, "predict_proba"):
        try:
            model_classes = list(pipeline.classes_)
            if 1 in model_classes:
                probabilities = np.asarray(pipeline.predict_proba(features)[:, model_classes.index(1)], dtype=float)
        except Exception as exc:
            raise PredictionValidationError(f"The model's probability scores could not be calculated: {exc}") from exc
        if probabilities is not None and (
            len(probabilities) != len(frame)
            or not np.isfinite(probabilities).all()
            or not np.all((probabilities >= 0) & (probabilities <= 1))
        ):
            raise PredictionValidationError("The model returned invalid probability scores; no risk categories were generated.")

    id_column = config.get("customer_id")
    if id_column and id_column not in frame.columns:
        raise PredictionValidationError("The configured customer identifier is missing from the scoring CSV.")
    target_column = config.get("target")
    has_actual = bool(target_column and target_column in frame.columns)
    rows: list[dict[str, Any]] = []
    risk_counts = {"high": 0, "medium": 0, "low": 0, "unavailable": 0}
    for index, predicted_value in enumerate(predicted):
        probability = float(probabilities[index]) if probabilities is not None else None
        risk = _risk_category(probability, high_threshold, medium_threshold)
        risk_counts[risk] += 1
        customer_id = None
        if id_column:
            value = frame.iloc[index][id_column]
            customer_id = None if pd.isna(value) else str(value)
        actual = None
        if has_actual:
            value = frame.iloc[index][target_column]
            actual = None if pd.isna(value) else str(value)
        rows.append({
            "row_number": index + 1,
            "customer_id": customer_id,
            "churn_probability": probability,
            "predicted_class": str(model_version.positive_class if predicted_value == 1 else model_version.negative_class),
            "risk_category": risk,
            "actual_outcome": actual,
        })

    summary = {
        "row_count": len(rows),
        "probabilities_available": probabilities is not None,
        "risk_counts": risk_counts,
        "actual_outcomes_included": has_actual,
        "positive_class": model_version.positive_class,
        "negative_class": model_version.negative_class,
    }
    return rows, summary
