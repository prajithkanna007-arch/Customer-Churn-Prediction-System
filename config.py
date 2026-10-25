"""Application configuration."""

from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Defaults shared by local and test configurations."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'instance' / 'churn.sqlite3').as_posix()}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_UPLOAD_SIZE = 20 * 1024 * 1024
    # Allow multipart boundaries and form fields in addition to the file bytes.
    MAX_CONTENT_LENGTH = MAX_UPLOAD_SIZE + 256 * 1024
    UPLOAD_FOLDER = str(BASE_DIR / "instance" / "uploads")
    MODEL_FOLDER = str(BASE_DIR / "instance" / "models")
    TRAINING_RANDOM_STATE = 42
    TRAINING_TEST_FRACTION = 0.20
    TRAINING_VALIDATION_FRACTION = 0.20
    TRAINING_MIN_CLASS_SAMPLES = 3
    MAX_CATEGORIES_PER_FEATURE = 500
    RANDOM_FOREST_N_ESTIMATORS = 200
    TRAINING_N_JOBS = -1
    DEFAULT_HIGH_RISK_THRESHOLD = 0.70
    DEFAULT_MEDIUM_RISK_THRESHOLD = 0.40


class TestingConfig(Config):
    """Isolated configuration for automated tests."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
