"""
app/__init__.py
-----------------
Application factory. The model is loaded exactly once here, at app
creation time - never inside a request handler.
"""

import logging

from flask import Flask

from app.config import Config
from app.services.autoencoder_service import ModelLoadError, init_autoencoder_service


def create_app(config_class=Config) -> Flask:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    app = Flask(__name__)
    app.config.from_object(config_class)

    try:
        init_autoencoder_service(app.config["MODEL_PATH"])
    except ModelLoadError as exc:
        # We deliberately do not crash the whole process: the app will
        # still start (so health checks / static pages work), but
        # /api/predict will return a clean 503 until this is fixed.
        logger.error("Model failed to load at startup: %s", exc)

    from app.routes.prediction_routes import bp as prediction_bp

    app.register_blueprint(prediction_bp)

    return app
