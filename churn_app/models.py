"""Persistence models for workspaces and uploaded datasets."""

from datetime import datetime, timezone

from .extensions import db


class Workspace(db.Model):
    """A local application workspace, reserved for future configuration."""

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False, default="Default workspace")
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class Dataset(db.Model):
    """Metadata and user-approved column configuration for one uploaded CSV."""

    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    storage_filename = db.Column(db.String(80), nullable=False, unique=True)
    row_count = db.Column(db.Integer, nullable=False)
    column_count = db.Column(db.Integer, nullable=False)
    uploaded_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    validation_status = db.Column(db.String(32), nullable=False)
    column_names = db.Column(db.JSON, nullable=False)
    quality_summary = db.Column(db.JSON, nullable=False)
    configuration = db.Column(db.JSON, nullable=True)
    training_runs = db.relationship("TrainingRun", back_populates="dataset", cascade="all, delete-orphan")
    model_versions = db.relationship("ModelVersion", back_populates="dataset", cascade="all, delete-orphan")
    prediction_runs = db.relationship("PredictionRun", back_populates="dataset", cascade="all, delete-orphan")

    @property
    def dataset_type(self) -> str:
        """Return the configured analytical capability without claiming predictions."""
        if not self.configuration:
            return "unconfigured"
        return self.configuration.get("dataset_type", "unconfigured")


class TrainingRun(db.Model):
    """One requested baseline-model comparison, including failed attempts."""

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False, index=True)
    status = db.Column(db.String(24), nullable=False, default="running")
    selection_procedure = db.Column(db.String(255), nullable=False)
    summary = db.Column(db.JSON, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    started_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)

    dataset = db.relationship("Dataset", back_populates="training_runs")
    models = db.relationship("ModelVersion", back_populates="training_run", cascade="all, delete-orphan")


class ModelVersion(db.Model):
    """A fitted, application-managed model pipeline and its evaluation record."""

    id = db.Column(db.Integer, primary_key=True)
    training_run_id = db.Column(db.Integer, db.ForeignKey("training_run.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False, index=True)
    algorithm = db.Column(db.String(80), nullable=False)
    algorithm_version = db.Column(db.String(40), nullable=False)
    feature_columns = db.Column(db.JSON, nullable=False)
    feature_kinds = db.Column(db.JSON, nullable=False)
    column_mapping = db.Column(db.JSON, nullable=False)
    target_column = db.Column(db.String(255), nullable=False)
    positive_class = db.Column(db.String(255), nullable=False)
    negative_class = db.Column(db.String(255), nullable=False)
    date_feature = db.Column(db.String(255), nullable=True)
    preprocessing_config = db.Column(db.JSON, nullable=False)
    validation_metrics = db.Column(db.JSON, nullable=False)
    test_metrics = db.Column(db.JSON, nullable=False)
    split_metadata = db.Column(db.JSON, nullable=False)
    selection_procedure = db.Column(db.String(255), nullable=False)
    artifact_filename = db.Column(db.String(100), nullable=False, unique=True)
    status = db.Column(db.String(24), nullable=False, default="ready")
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    dataset = db.relationship("Dataset", back_populates="model_versions")
    training_run = db.relationship("TrainingRun", back_populates="models")
    prediction_runs = db.relationship("PredictionRun", back_populates="model_version", cascade="all, delete-orphan")


class PredictionRun(db.Model):
    """A persisted request to score one dataset with one saved model version."""

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False, index=True)
    model_version_id = db.Column(db.Integer, db.ForeignKey("model_version.id", ondelete="CASCADE"), nullable=False, index=True)
    status = db.Column(db.String(24), nullable=False, default="running")
    row_count = db.Column(db.Integer, nullable=False, default=0)
    high_risk_threshold = db.Column(db.Float, nullable=False)
    medium_risk_threshold = db.Column(db.Float, nullable=False)
    summary = db.Column(db.JSON, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)

    dataset = db.relationship("Dataset", back_populates="prediction_runs")
    model_version = db.relationship("ModelVersion", back_populates="prediction_runs")
    records = db.relationship("PredictionRecord", back_populates="prediction_run", cascade="all, delete-orphan", order_by="PredictionRecord.row_number")


class PredictionRecord(db.Model):
    """One customer-row prediction stored separately for bounded pagination."""

    id = db.Column(db.Integer, primary_key=True)
    prediction_run_id = db.Column(db.Integer, db.ForeignKey("prediction_run.id", ondelete="CASCADE"), nullable=False, index=True)
    row_number = db.Column(db.Integer, nullable=False)
    customer_id = db.Column(db.String(255), nullable=True, index=True)
    churn_probability = db.Column(db.Float, nullable=True, index=True)
    predicted_class = db.Column(db.String(255), nullable=False, index=True)
    risk_category = db.Column(db.String(24), nullable=False, index=True)
    actual_outcome = db.Column(db.String(255), nullable=True)

    prediction_run = db.relationship("PredictionRun", back_populates="records")

    __table_args__ = (db.UniqueConstraint("prediction_run_id", "row_number", name="uq_prediction_row"),)
