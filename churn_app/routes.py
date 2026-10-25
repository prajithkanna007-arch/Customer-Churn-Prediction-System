"""HTML page routes, dataset APIs, and service health endpoint."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from flask import Blueprint, Response, current_app, jsonify, render_template, request, stream_with_context
from sqlalchemy import insert
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from .extensions import db
from .models import Dataset, ModelVersion, PredictionRecord, PredictionRun, TrainingRun
from .services.csv_service import (
    CSVValidationError,
    parse_csv_bytes,
    quality_report,
    safe_preview,
    suggested_mapping,
    target_values,
)
from .services.ml_service import TrainingValidationError, train_baselines
from .services.model_storage import ModelArtifactError, load_pipeline, remove_artifact, save_pipeline
from .services.prediction_service import PredictionValidationError, score_dataset

main = Blueprint("main", __name__)

PAGES = {
    "dashboard": ("Dashboard", "A clear view of customer retention will appear here when data is connected."),
    "data": ("Data Management", "Upload and configure your own customer data."),
    "customers": ("Customer Directory", "Customer records will appear here after a dataset is configured."),
    "prediction": ("Churn Prediction", "Predictions require a compatible trained model and uploaded data."),
    "analysis": ("Churn Analysis", "Churn analysis will be available when suitable historical data is configured."),
    "retention": ("Retention Center", "Retention recommendations and action tracking are planned for a later phase."),
    "reports": ("Reports & Model Performance", "Reports and evaluation metrics will appear after a model has been trained."),
    "settings": ("Settings", "Workspace and data mapping settings will be added in a later phase."),
}


@main.get("/")
def dashboard():
    """Render the default dashboard."""
    return render_page("dashboard")


def render_page(page_key: str):
    """Render a page using shared navigation and an honest empty state."""
    title, message = PAGES[page_key]
    if page_key == "data":
        return render_template("data_management.html", page_key=page_key, title=title, message=message)
    if page_key == "prediction":
        return render_template("prediction.html", page_key=page_key, title=title, message=message)
    if page_key == "reports":
        return render_template("reports.html", page_key=page_key, title=title, message=message)
    return render_template("page.html", page_key=page_key, title=title, message=message)


@main.get("/data")
def data_management():
    return render_page("data")


@main.get("/customers")
def customers():
    return render_page("customers")


@main.get("/prediction")
def prediction():
    return render_page("prediction")


@main.get("/analysis")
def analysis():
    return render_page("analysis")


@main.get("/retention")
def retention():
    return render_page("retention")


@main.get("/reports")
def reports():
    return render_page("reports")


@main.get("/settings")
def settings():
    return render_page("settings")


@main.get("/health")
def health():
    """Return a lightweight liveness response."""
    return jsonify(status="ok"), 200


def _dataset_path(dataset: Dataset) -> Path:
    """Resolve a generated storage name and ensure it remains in upload storage."""
    root = Path(current_app.config["UPLOAD_FOLDER"]).resolve()
    path = (root / dataset.storage_filename).resolve()
    if path.parent != root or path.suffix.lower() != ".csv":
        raise CSVValidationError("Stored dataset path is invalid.")
    return path


def _load_dataset(dataset: Dataset):
    """Read and revalidate a stored CSV before serving data from it."""
    try:
        return parse_csv_bytes(_dataset_path(dataset).read_bytes())
    except FileNotFoundError as exc:
        raise CSVValidationError("The stored CSV file is missing. Delete this dataset and upload it again.") from exc


def _dataset_summary(dataset: Dataset) -> dict[str, Any]:
    return {
        "id": dataset.id,
        "original_filename": dataset.original_filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "uploaded_at": dataset.uploaded_at.isoformat() if dataset.uploaded_at else None,
        "validation_status": dataset.validation_status,
        "configuration": dataset.configuration,
        "dataset_type": dataset.dataset_type,
    }


def _training_run_summary(run: TrainingRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "dataset_id": run.dataset_id,
        "dataset_name": run.dataset.original_filename,
        "status": run.status,
        "selection_procedure": run.selection_procedure,
        "summary": run.summary,
        "error_message": run.error_message,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


def _model_summary(model: ModelVersion) -> dict[str, Any]:
    return {
        "id": model.id,
        "training_run_id": model.training_run_id,
        "dataset_id": model.dataset_id,
        "dataset_name": model.dataset.original_filename,
        "algorithm": model.algorithm,
        "algorithm_version": model.algorithm_version,
        "feature_columns": model.feature_columns,
        "column_mapping": model.column_mapping,
        "target_column": model.target_column,
        "positive_class": model.positive_class,
        "negative_class": model.negative_class,
        "validation_metrics": model.validation_metrics,
        "test_metrics": model.test_metrics,
        "selection_procedure": model.selection_procedure,
        "status": model.status,
        "created_at": model.created_at.isoformat() if model.created_at else None,
    }


def _prediction_run_summary(run: PredictionRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "dataset_id": run.dataset_id,
        "dataset_name": run.dataset.original_filename,
        "model_version_id": run.model_version_id,
        "algorithm": run.model_version.algorithm,
        "status": run.status,
        "row_count": run.row_count,
        "high_risk_threshold": run.high_risk_threshold,
        "medium_risk_threshold": run.medium_risk_threshold,
        "summary": run.summary,
        "error_message": run.error_message,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@main.post("/api/datasets/<int:dataset_id>/train")
def train_dataset_models(dataset_id: int):
    """Train and persist both supported baseline pipelines for a labeled dataset."""
    dataset = db.session.get(Dataset, dataset_id)
    if dataset is None:
        return jsonify(error="Dataset not found."), 404

    selection_procedure = (
        "Compare fixed baselines using validation positive-class F1 at threshold 0.5; "
        "reserve the held-out test split for final evaluation."
    )
    run = TrainingRun(dataset_id=dataset.id, status="running", selection_procedure=selection_procedure)
    db.session.add(run)
    db.session.commit()
    run_id = run.id
    artifact_names: list[str] = []

    try:
        parsed = _load_dataset(dataset)
        trained_models = train_baselines(parsed.frame, dataset.configuration, current_app.config)
        model_rows: list[ModelVersion] = []
        for trained in trained_models:
            artifact_name = save_pipeline(trained.pipeline, current_app.config["MODEL_FOLDER"])
            artifact_names.append(artifact_name)
            model_rows.append(ModelVersion(
                training_run_id=run.id,
                dataset_id=dataset.id,
                algorithm=trained.algorithm,
                algorithm_version=trained.algorithm_version,
                feature_columns=trained.feature_columns,
                feature_kinds=trained.feature_kinds,
                column_mapping=dict(dataset.configuration),
                target_column=dataset.configuration["target"],
                positive_class=str(dataset.configuration["positive_class"]),
                negative_class=str(dataset.configuration["negative_class"]),
                date_feature=trained.date_feature,
                preprocessing_config=trained.preprocessing_config,
                validation_metrics=trained.validation_metrics,
                test_metrics=trained.test_metrics,
                split_metadata=trained.split_metadata,
                selection_procedure=trained.selection_procedure,
                artifact_filename=artifact_name,
                status="ready",
            ))
        db.session.add_all(model_rows)
        db.session.flush()
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        run.summary = {
            "model_count": len(model_rows),
            "model_ids": [model.id for model in model_rows],
            "split": model_rows[0].split_metadata,
        }
        db.session.commit()
        return jsonify(
            training_run=_training_run_summary(run),
            models=[_model_summary(model) for model in model_rows],
        ), 201
    except (TrainingValidationError, CSVValidationError) as exc:
        message = str(exc)
        db.session.rollback()
        for artifact_name in artifact_names:
            try:
                remove_artifact(current_app.config["MODEL_FOLDER"], artifact_name)
            except ModelArtifactError:
                pass
        failed_run = db.session.get(TrainingRun, run_id)
        failed_run.status = "failed"
        failed_run.error_message = message
        failed_run.completed_at = datetime.now(timezone.utc)
        db.session.commit()
        return jsonify(training_run=_training_run_summary(failed_run), error=message), 422
    except Exception as exc:
        db.session.rollback()
        for artifact_name in artifact_names:
            try:
                remove_artifact(current_app.config["MODEL_FOLDER"], artifact_name)
            except ModelArtifactError:
                pass
        failed_run = db.session.get(TrainingRun, run_id)
        failed_run.status = "failed"
        failed_run.error_message = f"Training failed ({type(exc).__name__}). Check the dataset and try again."
        failed_run.completed_at = datetime.now(timezone.utc)
        db.session.commit()
        return jsonify(training_run=_training_run_summary(failed_run), error=failed_run.error_message), 500


@main.get("/api/models")
def list_models():
    """List model versions and recent training runs without customer-level data."""
    models = ModelVersion.query.order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc()).all()
    runs = TrainingRun.query.order_by(TrainingRun.started_at.desc(), TrainingRun.id.desc()).limit(100).all()
    return jsonify(models=[_model_summary(model) for model in models], training_runs=[_training_run_summary(run) for run in runs])


@main.get("/api/models/<int:model_id>")
def get_model(model_id: int):
    model = db.session.get(ModelVersion, model_id)
    if model is None:
        return jsonify(error="Model version not found."), 404
    return jsonify(
        model=_model_summary(model),
        preprocessing_config=model.preprocessing_config,
        feature_kinds=model.feature_kinds,
        split_metadata=model.split_metadata,
        target_mapping={"column": model.target_column, "positive_class": model.positive_class, "negative_class": model.negative_class},
    )


@main.get("/api/models/<int:model_id>/metrics")
def model_metrics(model_id: int):
    model = db.session.get(ModelVersion, model_id)
    if model is None:
        return jsonify(error="Model version not found."), 404
    return jsonify(
        model_id=model.id,
        algorithm=model.algorithm,
        validation_metrics=model.validation_metrics,
        test_metrics=model.test_metrics,
        split_metadata=model.split_metadata,
        selection_procedure=model.selection_procedure,
    )


def _mark_prediction_failed(run_id: int, message: str):
    db.session.rollback()
    failed_run = db.session.get(PredictionRun, run_id)
    failed_run.status = "failed"
    failed_run.error_message = message
    failed_run.completed_at = datetime.now(timezone.utc)
    db.session.commit()
    return failed_run


@main.post("/api/predictions")
def create_predictions():
    """Score a configured dataset using one compatible saved model version."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send prediction options as a JSON object."), 400
    dataset_id = payload.get("dataset_id")
    model_id = payload.get("model_id")
    if isinstance(dataset_id, bool) or not isinstance(dataset_id, int) or isinstance(model_id, bool) or not isinstance(model_id, int):
        return jsonify(error="Choose a dataset and saved model version."), 400
    dataset = db.session.get(Dataset, dataset_id)
    if dataset is None:
        return jsonify(error="Dataset not found."), 404
    model = db.session.get(ModelVersion, model_id)
    if model is None or model.status != "ready":
        return jsonify(error="A ready saved model version was not found."), 404

    def threshold_value(key: str, default: float) -> float:
        value = payload.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key.replace('_', ' ').title()} must be a number between 0 and 1.")
        number = float(value)
        if not np.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"{key.replace('_', ' ').title()} must be a number between 0 and 1.")
        return number

    try:
        high_threshold = threshold_value("high_risk_threshold", current_app.config["DEFAULT_HIGH_RISK_THRESHOLD"])
        medium_threshold = threshold_value("medium_risk_threshold", current_app.config["DEFAULT_MEDIUM_RISK_THRESHOLD"])
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if medium_threshold >= high_threshold:
        return jsonify(error="The medium-risk threshold must be lower than the high-risk threshold."), 400

    run = PredictionRun(
        dataset_id=dataset.id,
        model_version_id=model.id,
        status="running",
        high_risk_threshold=high_threshold,
        medium_risk_threshold=medium_threshold,
    )
    db.session.add(run)
    db.session.commit()
    run_id = run.id
    try:
        parsed = _load_dataset(dataset)
        pipeline = load_pipeline(current_app.config["MODEL_FOLDER"], model.artifact_filename)
        records, summary = score_dataset(
            parsed.frame,
            dataset.configuration,
            model,
            pipeline,
            high_threshold,
            medium_threshold,
        )
        for start in range(0, len(records), 1000):
            db.session.execute(insert(PredictionRecord), [
                {"prediction_run_id": run_id, **record} for record in records[start:start + 1000]
            ])
        run.status = "completed"
        run.row_count = len(records)
        run.summary = summary
        run.completed_at = datetime.now(timezone.utc)
        db.session.commit()
    except (PredictionValidationError, CSVValidationError, ModelArtifactError) as exc:
        failed_run = _mark_prediction_failed(run_id, str(exc))
        return jsonify(prediction_run=_prediction_run_summary(failed_run), error=str(exc)), 422
    except Exception as exc:
        message = f"Prediction failed ({type(exc).__name__}). Check the model and dataset configuration, then retry."
        failed_run = _mark_prediction_failed(run_id, message)
        return jsonify(prediction_run=_prediction_run_summary(failed_run), error=message), 500
    return jsonify(prediction_run=_prediction_run_summary(run)), 201


@main.get("/api/predictions/<int:prediction_id>")
def get_prediction_run(prediction_id: int):
    run = db.session.get(PredictionRun, prediction_id)
    if run is None:
        return jsonify(error="Prediction run not found."), 404
    return jsonify(prediction_run=_prediction_run_summary(run))


@main.get("/api/predictions/<int:prediction_id>/results")
def prediction_results(prediction_id: int):
    run = db.session.get(PredictionRun, prediction_id)
    if run is None:
        return jsonify(error="Prediction run not found."), 404
    if run.status != "completed":
        return jsonify(error=run.error_message or "Prediction results are not ready.", status=run.status), 409
    try:
        page = int(request.args.get("page", "1"))
        per_page = int(request.args.get("per_page", "50"))
    except ValueError:
        return jsonify(error="Page and page size must be integers."), 400
    if page < 1 or per_page < 1 or per_page > 100:
        return jsonify(error="Page must be positive and page size must be between 1 and 100."), 400

    query = PredictionRecord.query.filter_by(prediction_run_id=run.id)
    risk = request.args.get("risk", "").strip().lower()
    if risk:
        if risk not in {"high", "medium", "low", "unavailable"}:
            return jsonify(error="Risk filter must be high, medium, low, or unavailable."), 400
        query = query.filter(PredictionRecord.risk_category == risk)
    search = request.args.get("q", "").strip()
    if len(search) > 100:
        return jsonify(error="Search text is limited to 100 characters."), 400
    if search:
        query = query.filter(PredictionRecord.customer_id.ilike(f"%{search}%"))

    sort_key = request.args.get("sort", "row_number")
    sort_columns = {
        "row_number": PredictionRecord.row_number,
        "customer_id": PredictionRecord.customer_id,
        "churn_probability": PredictionRecord.churn_probability,
        "predicted_class": PredictionRecord.predicted_class,
        "risk_category": PredictionRecord.risk_category,
    }
    if sort_key not in sort_columns:
        return jsonify(error="Unsupported sort column."), 400
    direction = request.args.get("direction", "asc").lower()
    if direction not in {"asc", "desc"}:
        return jsonify(error="Sort direction must be asc or desc."), 400
    sort_column = sort_columns[sort_key]
    query = query.order_by(sort_column.desc() if direction == "desc" else sort_column.asc(), PredictionRecord.row_number.asc())
    total = query.count()
    pages = (total + per_page - 1) // per_page
    if pages and page > pages:
        page = pages
    elif not pages:
        page = 1
    result_page = query.offset((page - 1) * per_page).limit(per_page).all()
    return jsonify(
        prediction_run=_prediction_run_summary(run),
        page=page,
        per_page=per_page,
        total=total,
        pages=pages,
        actual_outcomes_included=bool(run.summary and run.summary.get("actual_outcomes_included")),
        results=[{
            "row_number": record.row_number,
            "customer_id": record.customer_id,
            "churn_probability": record.churn_probability,
            "predicted_class": record.predicted_class,
            "risk_category": record.risk_category,
            "actual_outcome": record.actual_outcome,
        } for record in result_page],
    )


def _csv_safe(value: str | None) -> str:
    """Neutralize spreadsheet-formula prefixes in untrusted textual fields."""
    if value is not None and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value or ""


@main.get("/api/predictions/<int:prediction_id>/export")
def export_prediction_results(prediction_id: int):
    """Stream all structured prediction rows to a spreadsheet-safe CSV export."""
    run = db.session.get(PredictionRun, prediction_id)
    if run is None:
        return jsonify(error="Prediction run not found."), 404
    if run.status != "completed":
        return jsonify(error="Only completed prediction runs can be exported."), 409

    def generate():
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["row_number", "customer_id", "churn_probability", "predicted_class", "risk_category", "actual_outcome"])
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        query = PredictionRecord.query.filter_by(prediction_run_id=run.id).order_by(PredictionRecord.row_number.asc()).yield_per(1000)
        for record in query:
            writer.writerow([
                record.row_number,
                _csv_safe(record.customer_id),
                "" if record.churn_probability is None else f"{record.churn_probability:.8f}",
                _csv_safe(record.predicted_class),
                record.risk_category,
                _csv_safe(record.actual_outcome),
            ])
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)

    return Response(
        stream_with_context(generate()),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=prediction_run_{run.id}.csv"},
    )


@main.get("/api/datasets")
def list_datasets():
    """List metadata only; customer rows are never returned by this endpoint."""
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc(), Dataset.id.desc()).all()
    return jsonify(datasets=[_dataset_summary(dataset) for dataset in datasets])


@main.post("/api/datasets/upload")
def upload_dataset():
    """Validate and persist a CSV using a generated filename."""
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify(error="Choose a CSV file to upload."), 400
    original_name = (uploaded.filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not original_name or not secure_filename(original_name):
        return jsonify(error="The uploaded file must have a valid filename."), 400
    if Path(original_name).suffix.casefold() != ".csv":
        return jsonify(error="Only files with a .csv extension are accepted."), 415

    limit = int(current_app.config["MAX_UPLOAD_SIZE"])
    content = uploaded.stream.read(limit + 1)
    if len(content) > limit:
        return jsonify(error=f"The file exceeds the {limit // (1024 * 1024)} MB upload limit."), 413
    try:
        parsed = parse_csv_bytes(content)
    except CSVValidationError as exc:
        return jsonify(error=str(exc)), 400

    root = Path(current_app.config["UPLOAD_FOLDER"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    storage_filename = f"{uuid4().hex}.csv"
    stored_path = root / storage_filename
    stored_path.write_bytes(content)
    report = quality_report(parsed.frame, parsed.header_changes)
    dataset = Dataset(
        original_filename=original_name[:255],
        storage_filename=storage_filename,
        row_count=report["row_count"],
        column_count=report["column_count"],
        validation_status="valid_with_findings" if report["missing_cells"] or report["duplicate_rows"] or report["high_cardinality_columns"] or parsed.header_changes else "valid",
        column_names=[column["name"] for column in report["columns"]],
        quality_summary=report,
        configuration=None,
    )
    try:
        db.session.add(dataset)
        db.session.commit()
    except Exception:
        db.session.rollback()
        stored_path.unlink(missing_ok=True)
        raise

    return jsonify(
        dataset=_dataset_summary(dataset),
        preview={"columns": dataset.column_names, "rows": safe_preview(parsed.frame)},
        quality=report,
        suggested_configuration=suggested_mapping(parsed.frame),
        encoding=parsed.encoding,
    ), 201


@main.get("/api/datasets/<int:dataset_id>/preview")
def preview_dataset(dataset_id: int):
    """Return a bounded first-15-row preview and stored quality summary."""
    dataset = db.session.get(Dataset, dataset_id)
    if dataset is None:
        return jsonify(error="Dataset not found."), 404
    try:
        parsed = _load_dataset(dataset)
    except CSVValidationError as exc:
        return jsonify(error=str(exc)), 422
    return jsonify(
        dataset=_dataset_summary(dataset),
        columns=dataset.column_names,
        rows=safe_preview(parsed.frame),
        preview_limit=15,
        suggested_configuration=suggested_mapping(parsed.frame),
    )


@main.get("/api/datasets/<int:dataset_id>/quality")
def dataset_quality(dataset_id: int):
    """Return structural quality findings and optionally target class choices."""
    dataset = db.session.get(Dataset, dataset_id)
    if dataset is None:
        return jsonify(error="Dataset not found."), 404
    try:
        parsed = _load_dataset(dataset)
        report = quality_report(parsed.frame, dataset.quality_summary.get("header_changes", []))
        target_column = request.args.get("target_column")
        if target_column:
            report["target_values"] = target_values(parsed.frame, target_column)
    except CSVValidationError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(quality=report)


@main.post("/api/datasets/<int:dataset_id>/configure")
def configure_dataset(dataset_id: int):
    """Validate and save explicit, dataset-specific column mappings."""
    dataset = db.session.get(Dataset, dataset_id)
    if dataset is None:
        return jsonify(error="Dataset not found."), 404
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the column configuration as a JSON object."), 400
    try:
        parsed = _load_dataset(dataset)
    except CSVValidationError as exc:
        return jsonify(error=str(exc)), 422

    columns = set(parsed.frame.columns)

    def optional_column(key: str) -> str | None:
        value = payload.get(key)
        if value in (None, ""):
            return None
        if not isinstance(value, str) or value not in columns:
            raise CSVValidationError(f"{key.replace('_', ' ').title()} must be one of this dataset's columns.")
        return value

    def column_list(key: str) -> list[str]:
        value = payload.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise CSVValidationError(f"{key.replace('_', ' ').title()} must be a list of column names.")
        if len(value) != len(set(value)) or any(item not in columns for item in value):
            raise CSVValidationError(f"{key.replace('_', ' ').title()} contains duplicate or unknown columns.")
        return value

    try:
        customer_id = optional_column("customer_id")
        target = optional_column("target")
        date_column = optional_column("date")
        revenue = optional_column("revenue")
        features = column_list("features")
        excluded = column_list("excluded")
        positive = payload.get("positive_class") or None
        negative = payload.get("negative_class") or None
        if positive is not None and not isinstance(positive, str):
            raise CSVValidationError("Positive churn class must be a text value from the selected target column.")
        if negative is not None and not isinstance(negative, str):
            raise CSVValidationError("Negative churn class must be a text value from the selected target column.")

        if target is None:
            if positive is not None or negative is not None:
                raise CSVValidationError("Select a target column before choosing positive and negative classes.")
        else:
            if not positive or not negative:
                raise CSVValidationError("Select both the positive churn class and the negative/non-churn class.")
            observed = set(target_values(parsed.frame, target, limit=len(parsed.frame))["values"])
            if positive == negative:
                raise CSVValidationError("Positive churn and negative/non-churn classes must be different.")
            if positive not in observed or negative not in observed:
                raise CSVValidationError("Both target classes must be values present in the selected target column.")

        if customer_id and customer_id in features:
            raise CSVValidationError("The customer identifier cannot also be selected as a model feature.")
        if target and target in features:
            raise CSVValidationError("The historical target cannot also be selected as a model feature.")
        if target and target in {customer_id, date_column, revenue}:
            raise CSVValidationError("The historical target must use a different column from the identifier, date, and revenue fields.")
        if date_column and date_column == revenue:
            raise CSVValidationError("Choose different columns for the date and revenue fields.")
        if set(features) & set(excluded):
            raise CSVValidationError("A column cannot be both a selected feature and excluded.")
        if target and target in excluded:
            raise CSVValidationError("The historical target cannot be excluded while it is configured as the target.")

        if target and features:
            dataset_type = "labeled_historical"
            capability = "Target classes and features are configured; this dataset may support supervised training after Phase 3 safeguards are added."
        elif target:
            dataset_type = "descriptive_only"
            capability = "A target is mapped, but no feature columns were selected, so supervised training is not currently possible."
        elif features:
            dataset_type = "unlabeled"
            capability = "No historical target is configured. Descriptive inspection is available; churn predictions require a compatible trained model."
        else:
            dataset_type = "descriptive_only"
            capability = "No feature columns were selected. This dataset cannot support supervised churn training as configured."

        dataset.configuration = {
            "customer_id": customer_id,
            "target": target,
            "positive_class": positive,
            "negative_class": negative,
            "features": features,
            "date": date_column,
            "revenue": revenue,
            "excluded": excluded,
            "dataset_type": dataset_type,
            "capability": capability,
        }
        db.session.commit()
    except CSVValidationError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        db.session.rollback()
        raise

    return jsonify(dataset=_dataset_summary(dataset)), 200


@main.delete("/api/datasets/<int:dataset_id>")
def delete_dataset(dataset_id: int):
    """Delete one dataset record and its generated file within upload storage."""
    dataset = db.session.get(Dataset, dataset_id)
    if dataset is None:
        return jsonify(error="Dataset not found."), 404
    try:
        stored_path = _dataset_path(dataset)
    except CSVValidationError as exc:
        return jsonify(error=str(exc)), 422
    artifact_names = [model.artifact_filename for model in dataset.model_versions]
    db.session.delete(dataset)
    db.session.commit()
    stored_path.unlink(missing_ok=True)
    artifact_cleanup_errors = []
    for artifact_name in artifact_names:
        try:
            remove_artifact(current_app.config["MODEL_FOLDER"], artifact_name)
        except (ModelArtifactError, OSError):
            artifact_cleanup_errors.append(artifact_name)
    return jsonify(deleted=True, id=dataset_id, artifact_cleanup_errors=artifact_cleanup_errors)


@main.app_errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    """Return a readable JSON response for requests exceeding Flask's limit."""
    return jsonify(error="The upload exceeds the configured request size limit."), 413
