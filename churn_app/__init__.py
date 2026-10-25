"""Flask application package."""

from pathlib import Path

from flask import Flask

from config import Config
from .extensions import db


def create_app(config_object: type = Config) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_object)

    if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite:///"):
        database_path = app.config["SQLALCHEMY_DATABASE_URI"].removeprefix("sqlite:///")
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    from . import models  # noqa: F401  Ensure model metadata is registered.
    from .routes import main

    app.register_blueprint(main)

    with app.app_context():
        db.create_all()

    return app
