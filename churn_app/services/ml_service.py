"""Leakage-aware baseline training and held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class TrainingValidationError(ValueError):
    """A dataset cannot safely support the requested supervised evaluation."""


@dataclass
class TrainedBaseline:
    """Fitted pipeline and evaluation evidence for one baseline algorithm."""

    algorithm: str
    algorithm_version: str
    pipeline: Pipeline
    feature_columns: list[str]
    feature_kinds: dict[str, str]
    date_feature: str | None
    preprocessing_config: dict[str, Any]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    split_metadata: dict[str, Any]
    selection_procedure: str


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Deterministically expand one parsed date column into calendar features."""

    def fit(self, values: Any, target: Any = None):
        return self

    def transform(self, values: Any) -> np.ndarray:
        series = pd.Series(np.asarray(values).reshape(-1))
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return np.column_stack((
            parsed.dt.year,
            parsed.dt.month,
            parsed.dt.day,
            parsed.dt.dayofweek,
            parsed.dt.hour,
        )).astype(float)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)


def _feature_schema(frame: pd.DataFrame, features: list[str], date_column: str | None, max_categories: int) -> tuple[dict[str, str], list[str], list[str], str | None]:
    if not features:
        raise TrainingValidationError("Select at least one feature column before training.")
    if len(features) != len(set(features)):
        raise TrainingValidationError("The configured feature list contains duplicate columns.")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise TrainingValidationError(f"Configured feature columns are missing from this CSV: {', '.join(missing)}.")

    date_feature = date_column if date_column in features else None
    numeric: list[str] = []
    categorical: list[str] = []
    kinds: dict[str, str] = {}
    for name in features:
        series = frame[name]
        if series.notna().sum() == 0:
            raise TrainingValidationError(f"Feature '{name}' contains no observed values and cannot be trained.")
        if name == date_feature:
            kinds[name] = "date"
        elif _is_numeric(series):
            kinds[name] = "numeric"
            numeric.append(name)
        else:
            kinds[name] = "categorical"
            categorical.append(name)
            distinct_count = int(series.nunique(dropna=True))
            if distinct_count > max_categories:
                raise TrainingValidationError(
                    f"Categorical feature '{name}' has {distinct_count} distinct values; the configured limit is {max_categories}. Exclude it or review the column mapping."
                )
    return kinds, numeric, categorical, date_feature


def _build_pipeline(algorithm: str, numeric: list[str], categorical: list[str], date_feature: str | None, random_state: int, settings: dict[str, Any]) -> tuple[Pipeline, dict[str, Any]]:
    scale_numeric = algorithm == "Logistic Regression"
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical))
    if date_feature:
        date_pipeline = Pipeline([("calendar_features", DateFeatureExtractor())])
        transformers.append(("date", date_pipeline, [date_feature]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    if algorithm == "Logistic Regression":
        estimator = LogisticRegression(class_weight="balanced", max_iter=1200, random_state=random_state)
    else:
        estimator = RandomForestClassifier(
            n_estimators=int(settings.get("RANDOM_FOREST_N_ESTIMATORS", 200)),
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=int(settings.get("TRAINING_N_JOBS", -1)),
        )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    preprocessing = {
        "numeric_imputer": "median",
        "numeric_scaling": "standard" if scale_numeric else "none",
        "categorical_imputer": "constant __missing__",
        "categorical_encoder": "one-hot, ignore unseen categories",
        "date_transform": "year, month, day, weekday, hour" if date_feature else None,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "date_feature": date_feature,
    }
    return pipeline, preprocessing


def _class_distribution(labels: np.ndarray, positive_class: str, negative_class: str) -> dict[str, Any]:
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    return {
        "positive_class": positive_class,
        "positive_count": positive_count,
        "negative_class": negative_class,
        "negative_count": negative_count,
    }


def _evaluate(pipeline: Pipeline, features: pd.DataFrame, labels: np.ndarray, positive_class: str, negative_class: str, split_role: str) -> dict[str, Any]:
    predicted = np.asarray(pipeline.predict(features), dtype=int)
    probabilities: np.ndarray | None = None
    auc: float | None = None
    auc_note: str | None = None
    if hasattr(pipeline, "predict_proba"):
        class_values = list(pipeline.classes_)
        if 1 in class_values:
            probabilities = np.asarray(pipeline.predict_proba(features)[:, class_values.index(1)], dtype=float)
            if np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1)) and len(np.unique(labels)) == 2:
                auc = float(roc_auc_score(labels, probabilities))
            else:
                probabilities = None
                auc_note = "ROC-AUC is unavailable because predicted probabilities were invalid or the evaluation split has one class."
        else:
            auc_note = "ROC-AUC is unavailable because the fitted model does not contain the configured positive class."
    else:
        auc_note = "This estimator does not provide probability scores."

    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    result: dict[str, Any] = {
        "split": split_role,
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
        "roc_auc": auc,
        "roc_auc_note": auc_note,
        "confusion_matrix": {
            "labels": [negative_class, positive_class],
            "negative_as_negative": int(matrix[0, 0]),
            "negative_as_positive": int(matrix[0, 1]),
            "positive_as_negative": int(matrix[1, 0]),
            "positive_as_positive": int(matrix[1, 1]),
        },
        "sample_count": int(len(labels)),
        "class_distribution": _class_distribution(labels, positive_class, negative_class),
    }
    return result


def _splits(labels: np.ndarray, dates: pd.Series | None, test_fraction: float, validation_fraction: float, minimum_class_samples: int, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise TrainingValidationError("The configured target contains only one of the selected classes.")
    if int(counts.min()) < minimum_class_samples:
        raise TrainingValidationError(
            f"At least {minimum_class_samples} labeled rows from each selected class are required for separate training, validation, and test splits; counts are {int(counts[0])} negative and {int(counts[1])} positive."
        )

    total = len(labels)
    test_count = max(2, int(ceil(total * test_fraction)))
    validation_count = max(2, int(ceil(total * validation_fraction)))
    train_count = total - test_count - validation_count
    if train_count < 2 or test_count >= total or validation_count >= total - test_count:
        raise TrainingValidationError("There are too few labeled rows for separate training, validation, and test sets. Add more labeled examples or use a less restrictive evaluation plan.")

    if dates is not None:
        ordered = np.argsort(dates.astype("int64").to_numpy(), kind="stable")
        train_end = total - validation_count - test_count
        validation_end = total - test_count
        train_indices = ordered[:train_end]
        validation_indices = ordered[train_end:validation_end]
        test_indices = ordered[validation_end:]
        strategy = "chronological: oldest rows train, next rows validate, newest rows test"
    else:
        all_indices = np.arange(total)
        development, test_indices = train_test_split(
            all_indices,
            test_size=test_count,
            random_state=random_state,
            stratify=labels,
        )
        development_labels = labels[development]
        train_indices, validation_indices = train_test_split(
            development,
            test_size=validation_count,
            random_state=random_state,
            stratify=development_labels,
        )
        strategy = "stratified random train/validation/test split"

    for label, indices in (("training", train_indices), ("validation", validation_indices), ("test", test_indices)):
        if len(np.unique(labels[indices])) != 2:
            raise TrainingValidationError(
                f"The {label} split does not contain both selected classes. A reliable evaluation cannot be computed with this class distribution."
            )
    return train_indices, validation_indices, test_indices, strategy


def train_baselines(frame: pd.DataFrame, configuration: dict[str, Any], settings: dict[str, Any]) -> list[TrainedBaseline]:
    """Train two baseline pipelines and reserve the test split for final reporting."""
    if not isinstance(configuration, dict):
        raise TrainingValidationError("Save the dataset's column configuration before training.")
    if configuration.get("dataset_type") != "labeled_historical":
        raise TrainingValidationError("Only a configured labeled historical dataset can train a supervised churn model.")
    target_column = configuration.get("target")
    positive_class = configuration.get("positive_class")
    negative_class = configuration.get("negative_class")
    features = configuration.get("features")
    if not target_column or target_column not in frame.columns:
        raise TrainingValidationError("Choose a valid historical target column before training.")
    if not positive_class or not negative_class or positive_class == negative_class:
        raise TrainingValidationError("Choose distinct, observed positive and negative target classes before training.")
    if not isinstance(features, list) or not features:
        raise TrainingValidationError("Select at least one feature column before training.")
    if target_column in features:
        raise TrainingValidationError("The target column cannot be used as a model feature.")
    if configuration.get("customer_id") in features:
        raise TrainingValidationError("The customer identifier cannot be used as a model feature.")
    if set(features) & set(configuration.get("excluded", [])):
        raise TrainingValidationError("A feature is also marked excluded. Update the saved column mapping before training.")

    missing_features = [name for name in features if name not in frame.columns]
    if missing_features:
        raise TrainingValidationError(f"Feature columns are missing from the stored CSV: {', '.join(missing_features)}.")

    all_labels = frame[target_column]
    labeled_mask = all_labels.notna()
    dropped_missing_targets = int((~labeled_mask).sum())
    observed = set(all_labels.loc[labeled_mask].map(str).unique())
    unexpected = observed - {str(positive_class), str(negative_class)}
    if unexpected:
        sample = ", ".join(sorted(unexpected)[:5])
        raise TrainingValidationError(
            f"The target contains values outside the configured positive and negative classes (for example: {sample}). Select a binary target or clean these rows first."
        )

    training_frame = frame.loc[labeled_mask].copy().reset_index(drop=True)
    y = (training_frame[target_column].map(str).to_numpy() == str(positive_class)).astype(int)
    if len(np.unique(y)) < 2:
        raise TrainingValidationError("The labeled target contains only one of the selected classes.")

    identifier = configuration.get("customer_id")
    if identifier:
        if identifier not in training_frame.columns:
            raise TrainingValidationError("The configured customer identifier column is missing from the stored CSV.")
        ids = training_frame[identifier]
        if ids.isna().any():
            raise TrainingValidationError("The configured customer identifier contains missing values in labeled rows. Fill or remove them before training.")
        if ids.duplicated().any():
            raise TrainingValidationError("Duplicate customer identifiers were found in labeled rows. Resolve duplicates to avoid customer leakage across evaluation splits.")

    date_column = configuration.get("date")
    dates: pd.Series | None = None
    if date_column:
        if date_column not in training_frame.columns:
            raise TrainingValidationError("The configured date column is missing from the stored CSV.")
        dates = pd.to_datetime(training_frame[date_column], errors="coerce", utc=True)
        if dates.isna().any():
            raise TrainingValidationError("The configured date column contains missing or unreadable dates; chronological evaluation cannot be performed.")

    max_categories = int(settings.get("MAX_CATEGORIES_PER_FEATURE", 500))
    feature_kinds, numeric_features, categorical_features, date_feature = _feature_schema(
        training_frame, features, date_column, max_categories
    )
    X = training_frame.loc[:, features].copy()
    for name in numeric_features:
        X[name] = X[name].replace([np.inf, -np.inf], np.nan)
    for name in categorical_features:
        X[name] = X[name].map(lambda value: str(value) if pd.notna(value) else np.nan)

    random_state = int(settings.get("TRAINING_RANDOM_STATE", 42))
    train_indices, validation_indices, test_indices, split_strategy = _splits(
        y,
        dates,
        float(settings.get("TRAINING_TEST_FRACTION", 0.20)),
        float(settings.get("TRAINING_VALIDATION_FRACTION", 0.20)),
        int(settings.get("TRAINING_MIN_CLASS_SAMPLES", 3)),
        random_state,
    )
    training_rows = np.concatenate((train_indices, validation_indices))
    imbalance_ratio = float(max(np.bincount(y)) / min(np.bincount(y)))
    warnings = []
    if imbalance_ratio >= 10:
        warnings.append(f"Severe class imbalance was detected (largest/smallest class ratio {imbalance_ratio:.1f}:1); class weights are enabled and recall/precision should be reviewed.")
    warnings.append("Review selected features for post-outcome information. Identifiers and excluded columns are not model inputs.")
    split_metadata = {
        "strategy": split_strategy,
        "random_state": random_state if dates is None else None,
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "test_count": int(len(test_indices)),
        "labeled_rows": int(len(y)),
        "dropped_missing_target_rows": dropped_missing_targets,
        "class_distribution": _class_distribution(y, str(positive_class), str(negative_class)),
        "class_imbalance_ratio": imbalance_ratio,
        "warnings": warnings,
    }
    selection = "Compare fixed baselines using validation positive-class F1 at the default 0.5 threshold; the test split is reserved for final evaluation and is not used for model selection."

    baselines = ["Logistic Regression", "Random Forest"]
    trained: list[TrainedBaseline] = []
    for algorithm in baselines:
        selection_pipeline, _ = _build_pipeline(algorithm, numeric_features, categorical_features, date_feature, random_state, settings)
        selection_pipeline.fit(X.iloc[train_indices], y[train_indices])
        validation_metrics = _evaluate(
            selection_pipeline, X.iloc[validation_indices], y[validation_indices], str(positive_class), str(negative_class), "validation"
        )

        final_pipeline, preprocessing_config = _build_pipeline(algorithm, numeric_features, categorical_features, date_feature, random_state, settings)
        final_pipeline.fit(X.iloc[training_rows], y[training_rows])
        test_metrics = _evaluate(
            final_pipeline, X.iloc[test_indices], y[test_indices], str(positive_class), str(negative_class), "held-out test"
        )
        trained.append(TrainedBaseline(
            algorithm=algorithm,
            algorithm_version=sklearn.__version__,
            pipeline=final_pipeline,
            feature_columns=list(features),
            feature_kinds=feature_kinds,
            date_feature=date_feature,
            preprocessing_config=preprocessing_config,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            split_metadata=split_metadata,
            selection_procedure=selection,
        ))
    return trained
