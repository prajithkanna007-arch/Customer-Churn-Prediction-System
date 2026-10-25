"""Tests for app imports, CSV intake, quality reports, and mappings."""

import csv
from datetime import date, timedelta
from io import BytesIO, StringIO
from pathlib import Path
import sqlite3

import pytest
import pandas as pd

from churn_app import create_app
from churn_app.extensions import db
from churn_app.models import Dataset, ModelVersion, PredictionRecord, PredictionRun, TrainingRun, Workspace
from churn_app.services.ml_service import TrainingValidationError, train_baselines
from churn_app.services.model_storage import load_pipeline
from churn_app.services.model_storage import ModelArtifactError, artifact_path
from churn_app.services.prediction_service import _risk_category
from config import TestingConfig


@pytest.fixture()
def app(tmp_path):
    """Create an isolated app database and isolated upload directory."""
    app = create_app(TestingConfig)
    app.config.update(
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        MODEL_FOLDER=str(tmp_path / "models"),
        MAX_UPLOAD_SIZE=20 * 1024 * 1024,
        RANDOM_FOREST_N_ESTIMATORS=25,
        TRAINING_N_JOBS=1,
    )
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def test_factory_adds_dataset_table_without_resetting_existing_sqlite_data(tmp_path):
    database_path = tmp_path / "existing.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE workspace (id INTEGER PRIMARY KEY, name VARCHAR(120) NOT NULL, created_at DATETIME NOT NULL)")
    connection.execute("INSERT INTO workspace (id, name, created_at) VALUES (1, 'Existing workspace', '2025-01-01 00:00:00')")
    connection.commit()
    connection.close()

    config = type("TemporarySQLiteConfig", (TestingConfig,), {
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{database_path.as_posix()}",
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
    })
    app = create_app(config)
    with app.app_context():
        assert Workspace.query.one().name == "Existing workspace"
        assert Dataset.query.count() == 0
        db.session.remove()


def upload(client, content=b"account_key,active,monthly_value\nA-01,yes,12.5\nA-02,no,8\n", filename="customers.csv"):
    return client.post("/api/datasets/upload", data={"file": (BytesIO(content), filename)}, content_type="multipart/form-data")


def labeled_csv(row_count=60, duplicate_ids=False):
    """Build deterministic synthetic test records without using workspace data."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["person_ref", "usage_count", "plan_group", "outcome", "activity_date", "monthly_value"])
    start = date(2024, 1, 1)
    for index in range(row_count):
        positive = index % 2 == 0
        identifier = "DUPLICATE" if duplicate_ids and index < 2 else f"person-{index}"
        writer.writerow([
            identifier,
            "" if index % 17 == 0 else (index % 8 if positive else 20 + index % 8),
            "" if index % 13 == 0 else ("flex" if positive else "fixed"),
            "left" if positive else "stayed",
            (start + timedelta(days=index)).isoformat(),
            index * 3.5,
        ])
    return output.getvalue().encode("utf-8")


def create_labeled_dataset(client, content=None, filename="historical.csv"):
    """Upload and explicitly configure a disposable binary labeled dataset."""
    response = upload(client, content or labeled_csv(), filename)
    assert response.status_code == 201, response.get_json()
    dataset_id = response.get_json()["dataset"]["id"]
    configured = client.post(f"/api/datasets/{dataset_id}/configure", json={
        "customer_id": "person_ref",
        "target": "outcome",
        "positive_class": "left",
        "negative_class": "stayed",
        "features": ["usage_count", "plan_group"],
        "date": "activity_date",
        "revenue": "monthly_value",
        "excluded": [],
    })
    assert configured.status_code == 200, configured.get_json()
    return dataset_id


def create_unlabeled_dataset(client, rows=None, filename="to_score.csv", feature_order=None):
    """Upload and configure a separate disposable scoring dataset."""
    if rows is None:
        rows = [
            ["=2+2", 3, "new-plan", "extra-a"],
            ["member-2", 24, "fixed", "extra-b"],
            ["member-3", "", "", "extra-c"],
        ]
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["person_ref", "usage_count", "plan_group", "additional_column"])
    writer.writerows(rows)
    response = upload(client, output.getvalue().encode("utf-8"), filename)
    assert response.status_code == 201, response.get_json()
    dataset_id = response.get_json()["dataset"]["id"]
    configured = client.post(f"/api/datasets/{dataset_id}/configure", json={
        "customer_id": "person_ref",
        "features": feature_order or ["usage_count", "plan_group"],
        "excluded": ["additional_column"],
    })
    assert configured.status_code == 200, configured.get_json()
    return dataset_id


def test_health_endpoint_and_factory_routes(client, app):
    assert "main.upload_dataset" in app.view_functions
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


@pytest.mark.parametrize(
    "path,title",
    [
        ("/", "Dashboard"), ("/data", "Data Management"), ("/customers", "Customer Directory"),
        ("/prediction", "Churn Prediction"), ("/analysis", "Churn Analysis"),
        ("/retention", "Retention Center"), ("/reports", "Reports &amp; Model Performance"), ("/settings", "Settings"),
    ],
)
def test_page_routes_render(path, title, client):
    response = client.get(path)
    assert response.status_code == 200
    assert title.encode() in response.data


def test_data_management_page_has_functional_api_shell(client):
    response = client.get("/data")
    assert response.status_code == 200
    assert b"/api/datasets/upload" in response.data or b"Upload dataset" in response.data
    assert b"data_management.js" in response.data


def test_successful_csv_upload_persists_generated_file_and_metadata(client, app):
    response = upload(client, b"account_key,active,monthly_value\nA-01,yes,12.5\nA-02,no,8\n", "accounts.csv")
    assert response.status_code == 201
    body = response.get_json()
    assert body["dataset"]["original_filename"] == "accounts.csv"
    assert body["dataset"]["row_count"] == 2
    assert body["dataset"]["column_count"] == 3
    assert body["suggested_configuration"]["target"] is None
    assert body["preview"]["rows"][0]["account_key"] == "A-01"
    stored = list(Path(app.config["UPLOAD_FOLDER"]).glob("*.csv"))
    assert len(stored) == 1
    assert stored[0].name != "accounts.csv"
    with app.app_context():
        assert Dataset.query.count() == 1
    listing = client.get("/api/datasets")
    assert listing.status_code == 200
    assert len(listing.get_json()["datasets"]) == 1
    assert "rows" not in listing.get_json()["datasets"][0]


def test_same_original_filename_never_overwrites_existing_upload(client, app):
    first = upload(client, filename="same.csv").get_json()["dataset"]
    second = upload(client, filename="same.csv").get_json()["dataset"]
    assert first["id"] != second["id"]
    assert len(list(Path(app.config["UPLOAD_FOLDER"]).glob("*.csv"))) == 2


@pytest.mark.parametrize("content,message", [
    (b"", "empty"),
    (b"account,value\n", "no data rows"),
    (b"account,value\nA-01,one,two\n", "malformed"),
])
def test_empty_header_only_and_malformed_csv_rejected(client, content, message):
    response = upload(client, content)
    assert response.status_code == 400
    assert message in response.get_json()["error"].lower()


def test_invalid_file_type_rejected(client):
    response = upload(client, b"a,b\n1,2\n", "customers.txt")
    assert response.status_code == 415


def test_oversized_file_rejected(client, app):
    app.config["MAX_UPLOAD_SIZE"] = 1024
    response = upload(client, b"col\n" + b"x" * 1500)
    assert response.status_code == 413


def test_duplicate_and_blank_headers_are_normalized_and_reported(client):
    response = upload(client, b" name,name,\nfirst,second,third\n")
    assert response.status_code == 201
    body = response.get_json()
    assert body["preview"]["columns"] == ["name", "name__2", "column_3"]
    assert len(body["quality"]["header_changes"]) == 3


def test_utf16_csv_is_read_and_unreadable_encoding_has_clear_error(client):
    utf16 = "code,value\nA,1\n".encode("utf-16")
    accepted = upload(client, utf16)
    rejected = upload(client, b"code,value\nA,\x81\n")
    assert accepted.status_code == 201
    assert rejected.status_code == 400
    assert "encoding" in rejected.get_json()["error"].lower()


def test_preview_and_quality_report(client):
    content = b"record,segment\nA,alpha\nA,alpha\nB,\n"
    created = upload(client, content).get_json()["dataset"]
    preview = client.get(f"/api/datasets/{created['id']}/preview")
    quality = client.get(f"/api/datasets/{created['id']}/quality")
    assert preview.status_code == 200
    assert len(preview.get_json()["rows"]) == 3
    assert quality.status_code == 200
    assert quality.get_json()["quality"]["duplicate_rows"] == 1
    assert quality.get_json()["quality"]["missing_cells"] == 1


def test_preview_is_bounded_to_fifteen_rows(client):
    rows = "record,value\n" + "".join(f"R{i},{i}\n" for i in range(25))
    created = upload(client, rows.encode()).get_json()["dataset"]
    response = client.get(f"/api/datasets/{created['id']}/preview")
    assert response.status_code == 200
    assert response.get_json()["preview_limit"] == 15
    assert len(response.get_json()["rows"]) == 15


def test_high_cardinality_is_reported_as_a_finding(client):
    rows = "account,value\n" + "".join(f"account-{index},x\n" for index in range(30))
    created = upload(client, rows.encode()).get_json()["dataset"]
    response = client.get(f"/api/datasets/{created['id']}/quality")
    assert response.status_code == 200
    assert "account" in response.get_json()["quality"]["high_cardinality_columns"]


def test_quality_target_values_are_bounded_and_observed(client):
    created = upload(client).get_json()["dataset"]
    response = client.get(f"/api/datasets/{created['id']}/quality?target_column=active")
    assert response.status_code == 200
    assert set(response.get_json()["quality"]["target_values"]["values"]) == {"yes", "no"}


def test_configure_labeled_dataset_and_classification(client):
    created = upload(client).get_json()["dataset"]
    response = client.post(f"/api/datasets/{created['id']}/configure", json={
        "customer_id": "account_key", "target": "active", "positive_class": "yes", "negative_class": "no",
        "features": ["monthly_value"], "date": None, "revenue": "monthly_value", "excluded": [],
    })
    assert response.status_code == 200
    assert response.get_json()["dataset"]["dataset_type"] == "labeled_historical"
    assert response.get_json()["dataset"]["configuration"]["target"] == "active"


def test_unlabeled_dataset_is_distinguished_from_historical(client):
    created = upload(client).get_json()["dataset"]
    response = client.post(f"/api/datasets/{created['id']}/configure", json={
        "customer_id": "account_key", "features": ["active", "monthly_value"], "excluded": [],
    })
    assert response.status_code == 200
    assert response.get_json()["dataset"]["dataset_type"] == "unlabeled"
    assert "compatible trained model" in response.get_json()["dataset"]["configuration"]["capability"]


@pytest.mark.parametrize("target,positive,negative", [
    ("missing_target", "yes", "no"),
    ("active", "yes", "yes"),
    ("active", "yes", "unknown"),
    ("active", "", "no"),
])
def test_invalid_target_or_target_classes_rejected(client, target, positive, negative):
    created = upload(client).get_json()["dataset"]
    response = client.post(f"/api/datasets/{created['id']}/configure", json={
        "target": target, "positive_class": positive, "negative_class": negative,
        "features": ["monthly_value"], "excluded": [],
    })
    assert response.status_code == 400


def test_invalid_feature_overlap_and_unexpected_payload_rejected(client):
    created = upload(client).get_json()["dataset"]
    endpoint = f"/api/datasets/{created['id']}/configure"
    overlap = client.post(endpoint, json={"target": "active", "positive_class": "yes", "negative_class": "no", "features": ["active"], "excluded": []})
    malformed = client.post(endpoint, json=["unexpected"])
    assert overlap.status_code == 400
    assert malformed.status_code == 400


@pytest.mark.parametrize("suffix", ["preview", "quality", "configure"])
def test_nonexistent_dataset_ids_return_404(client, suffix):
    method = client.post if suffix == "configure" else client.get
    response = method(f"/api/datasets/9999/{suffix}", json={} if method == client.post else None)
    assert response.status_code == 404


def test_delete_removes_database_record_and_associated_file(client, app):
    created = upload(client).get_json()["dataset"]
    path = next(Path(app.config["UPLOAD_FOLDER"]).glob("*.csv"))
    response = client.delete(f"/api/datasets/{created['id']}")
    assert response.status_code == 200
    assert not path.exists()
    assert client.get(f"/api/datasets/{created['id']}/preview").status_code == 404


def test_upload_does_not_trust_client_filename_as_storage_path(client, app):
    response = upload(client, b"a,b\n1,2\n", "..\\..\\outside.csv")
    assert response.status_code == 201
    files = list(Path(app.config["UPLOAD_FOLDER"]).glob("*.csv"))
    assert len(files) == 1
    assert files[0].parent.resolve() == Path(app.config["UPLOAD_FOLDER"]).resolve()


def test_training_creates_both_baselines_persists_pipelines_and_reports_metrics(client, app):
    dataset_id = create_labeled_dataset(client)
    response = client.post(f"/api/datasets/{dataset_id}/train")
    assert response.status_code == 201, response.get_json()
    body = response.get_json()
    models = body["models"]
    assert {model["algorithm"] for model in models} == {"Logistic Regression", "Random Forest"}
    assert body["training_run"]["status"] == "completed"
    assert body["training_run"]["summary"]["split"]["strategy"].startswith("chronological")
    assert body["training_run"]["summary"]["split"]["dropped_missing_target_rows"] == 0

    for model in models:
        test_metrics = model["test_metrics"]
        validation_metrics = model["validation_metrics"]
        assert 0 <= test_metrics["accuracy"] <= 1
        assert 0 <= test_metrics["precision"] <= 1
        assert 0 <= test_metrics["recall"] <= 1
        assert 0 <= test_metrics["f1_score"] <= 1
        assert test_metrics["roc_auc"] is not None
        assert test_metrics["sample_count"] == 12
        assert validation_metrics["split"] == "validation"
        matrix = test_metrics["confusion_matrix"]
        assert matrix["labels"] == ["stayed", "left"]
        assert sum(matrix[key] for key in (
            "negative_as_negative", "negative_as_positive", "positive_as_negative", "positive_as_positive"
        )) == test_metrics["sample_count"]
        assert "test split is reserved" in model["selection_procedure"]
        with app.app_context():
            record = db.session.get(ModelVersion, model["id"])
            assert record is not None
            assert record.column_mapping["target"] == "outcome"
            assert record.column_mapping["features"] == ["usage_count", "plan_group"]
            artifact_path = Path(app.config["MODEL_FOLDER"], record.artifact_filename)
            assert artifact_path.is_file()
            loaded = load_pipeline(app.config["MODEL_FOLDER"], record.artifact_filename)
            assert hasattr(loaded, "predict_proba")

    detail = client.get(f"/api/models/{models[0]['id']}")
    metrics = client.get(f"/api/models/{models[0]['id']}/metrics")
    registry = client.get("/api/models")
    assert detail.status_code == metrics.status_code == registry.status_code == 200
    assert len(registry.get_json()["models"]) == 2
    assert metrics.get_json()["test_metrics"] == models[0]["test_metrics"]


def test_unlabeled_prediction_supports_unseen_categories_pagination_filters_and_export(client, app):
    historical_id = create_labeled_dataset(client)
    trained = client.post(f"/api/datasets/{historical_id}/train")
    assert trained.status_code == 201, trained.get_json()
    model_id = trained.get_json()["models"][0]["id"]
    scoring_id = create_unlabeled_dataset(client)

    response = client.post("/api/predictions", json={
        "dataset_id": scoring_id,
        "model_id": model_id,
        "high_risk_threshold": 0.8,
        "medium_risk_threshold": 0.3,
    })
    assert response.status_code == 201, response.get_json()
    run = response.get_json()["prediction_run"]
    assert run["summary"]["row_count"] == 3
    assert run["summary"]["actual_outcomes_included"] is False
    assert run["summary"]["probabilities_available"] is True
    assert sum(run["summary"]["risk_counts"].values()) == 3

    page = client.get(f"/api/predictions/{run['id']}/results?per_page=2&page=1&sort=churn_probability&direction=desc")
    assert page.status_code == 200
    assert page.get_json()["total"] == 3
    assert page.get_json()["pages"] == 2
    assert page.get_json()["actual_outcomes_included"] is False
    assert len(page.get_json()["results"]) == 2
    assert all(row["actual_outcome"] is None for row in page.get_json()["results"])
    probabilities = [row["churn_probability"] for row in page.get_json()["results"]]
    assert probabilities == sorted(probabilities, reverse=True)
    clamped = client.get(f"/api/predictions/{run['id']}/results?per_page=2&page=99")
    assert clamped.get_json()["page"] == 2

    risk = run["summary"]["risk_counts"]
    chosen_risk = next(key for key, value in risk.items() if value > 0 and key != "unavailable")
    filtered = client.get(f"/api/predictions/{run['id']}/results?risk={chosen_risk}&q=member")
    assert filtered.status_code == 200
    assert filtered.get_json()["total"] <= risk[chosen_risk]

    export = client.get(f"/api/predictions/{run['id']}/export")
    assert export.status_code == 200
    assert "text/csv" in export.content_type
    exported_text = export.get_data(as_text=True)
    assert "churn_probability" in exported_text
    assert "'=2+2" in exported_text
    assert len(exported_text.strip().splitlines()) == 4
    assert client.get(f"/api/predictions/{run['id']}").status_code == 200


def test_labeled_scoring_keeps_actual_outcome_separate_and_dataset_delete_cleans_artifacts(client, app):
    historical_id = create_labeled_dataset(client)
    trained = client.post(f"/api/datasets/{historical_id}/train")
    assert trained.status_code == 201
    model = trained.get_json()["models"][0]
    scoring = client.post("/api/predictions", json={"dataset_id": historical_id, "model_id": model["id"]})
    assert scoring.status_code == 201, scoring.get_json()
    run_id = scoring.get_json()["prediction_run"]["id"]
    results = client.get(f"/api/predictions/{run_id}/results?per_page=5")
    assert results.status_code == 200
    body = results.get_json()
    assert body["actual_outcomes_included"] is True
    assert body["results"][0]["actual_outcome"] in {"left", "stayed"}
    assert body["results"][0]["predicted_class"] in {"left", "stayed"}

    artifacts = list(Path(app.config["MODEL_FOLDER"]).glob("model_*.joblib"))
    assert len(artifacts) == 2
    deleted = client.delete(f"/api/datasets/{historical_id}")
    assert deleted.status_code == 200, deleted.get_json()
    assert all(not artifact.exists() for artifact in artifacts)
    with app.app_context():
        assert TrainingRun.query.count() == 0
        assert ModelVersion.query.count() == 0
        assert PredictionRun.query.count() == 0
        assert PredictionRecord.query.count() == 0


def test_prediction_rejects_incompatible_features_and_invalid_thresholds(client):
    dataset_id = create_labeled_dataset(client)
    trained = client.post(f"/api/datasets/{dataset_id}/train")
    model_id = trained.get_json()["models"][0]["id"]
    incompatible_id = create_unlabeled_dataset(client, feature_order=["plan_group", "usage_count"])
    incompatible = client.post("/api/predictions", json={"dataset_id": incompatible_id, "model_id": model_id})
    invalid = client.post("/api/predictions", json={
        "dataset_id": incompatible_id, "model_id": model_id,
        "high_risk_threshold": 0.4, "medium_risk_threshold": 0.4,
    })
    assert incompatible.status_code == 422
    assert "do not match" in incompatible.get_json()["error"]
    assert invalid.status_code == 400
    assert client.post("/api/predictions", json={"dataset_id": incompatible_id, "model_id": 9999}).status_code == 404
    assert client.post("/api/predictions", json={"dataset_id": 9999, "model_id": model_id}).status_code == 404


def test_prediction_rejects_feature_type_mismatch(client):
    dataset_id = create_labeled_dataset(client)
    trained = client.post(f"/api/datasets/{dataset_id}/train")
    model_id = trained.get_json()["models"][0]["id"]
    rows = [[f"person-{index}", "high", "unknown", f"extra-{index}"] for index in range(3)]
    scoring_id = create_unlabeled_dataset(client, rows=rows)
    response = client.post("/api/predictions", json={"dataset_id": scoring_id, "model_id": model_id})
    assert response.status_code == 422
    assert "must be numeric" in response.get_json()["error"]


@pytest.mark.parametrize("probability,high,medium,expected", [
    (0.70, 0.70, 0.40, "high"),
    (0.40, 0.70, 0.40, "medium"),
    (0.39, 0.70, 0.40, "low"),
    (None, 0.70, 0.40, "unavailable"),
])
def test_operational_risk_threshold_boundaries(probability, high, medium, expected):
    assert _risk_category(probability, high, medium) == expected


def test_model_artifact_loader_rejects_paths_outside_managed_directory(tmp_path):
    with pytest.raises(ModelArtifactError):
        artifact_path(str(tmp_path / "models"), "..\\outside.joblib")


def test_training_validation_errors_cover_missing_target_single_class_small_samples_and_duplicate_ids(app):
    settings = app.config
    with pytest.raises(TrainingValidationError, match="target column"):
        train_baselines(
            pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "label": ["yes", "no"] * 3}),
            {"target": "absent", "positive_class": "yes", "negative_class": "no", "features": ["feature"], "dataset_type": "labeled_historical"},
            settings,
        )
    with pytest.raises(TrainingValidationError, match="only one"):
        train_baselines(
            pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "label": ["yes"] * 6}),
            {"target": "label", "positive_class": "yes", "negative_class": "no", "features": ["feature"], "dataset_type": "labeled_historical"},
            settings,
        )
    with pytest.raises(TrainingValidationError, match="At least 3"):
        train_baselines(
            pd.DataFrame({"feature": [1, 2, 3, 4], "label": ["yes", "yes", "no", "no"]}),
            {"target": "label", "positive_class": "yes", "negative_class": "no", "features": ["feature"], "dataset_type": "labeled_historical"},
            settings,
        )
    with pytest.raises(TrainingValidationError, match="Duplicate customer identifiers"):
        train_baselines(
            pd.DataFrame({"identifier": ["x", "x", "z", "w", "v", "u"], "feature": [1, 2, 3, 4, 5, 6], "label": ["yes", "no"] * 3}),
            {"target": "label", "positive_class": "yes", "negative_class": "no", "features": ["feature"], "customer_id": "identifier", "dataset_type": "labeled_historical"},
            settings,
        )
    with pytest.raises(TrainingValidationError, match="binary target"):
        train_baselines(
            pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6, 7, 8, 9], "label": ["yes", "no", "maybe"] * 3}),
            {"target": "label", "positive_class": "yes", "negative_class": "no", "features": ["feature"], "dataset_type": "labeled_historical"},
            settings,
        )
    with pytest.raises(TrainingValidationError, match="distinct"):
        train_baselines(
            pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "label": ["yes", "no"] * 3}),
            {"target": "label", "positive_class": "yes", "negative_class": "yes", "features": ["feature"], "dataset_type": "labeled_historical"},
            settings,
        )


def test_training_drops_missing_target_rows_and_records_the_count(app):
    frame = pd.DataFrame({
        "feature": list(range(22)),
        "label": ["yes", "no"] * 10 + [None, None],
    })
    trained = train_baselines(
        frame,
        {"target": "label", "positive_class": "yes", "negative_class": "no", "features": ["feature"], "dataset_type": "labeled_historical"},
        app.config,
    )
    assert len(trained) == 2
    assert trained[0].split_metadata["dropped_missing_target_rows"] == 2
    assert trained[0].split_metadata["labeled_rows"] == 20


def test_train_api_records_failed_run_for_unlabeled_dataset(client, app):
    dataset_id = create_unlabeled_dataset(client)
    response = client.post(f"/api/datasets/{dataset_id}/train")
    assert response.status_code == 422
    assert "labeled historical" in response.get_json()["error"]
    with app.app_context():
        failed_run = TrainingRun.query.one()
        assert failed_run.status == "failed"
        assert failed_run.completed_at is not None
