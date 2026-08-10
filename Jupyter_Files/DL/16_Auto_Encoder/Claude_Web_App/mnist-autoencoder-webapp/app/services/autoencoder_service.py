"""
autoencoder_service.py
------------------------
Owns the trained model's lifecycle and inference pipeline. Isolated from
Flask routing entirely - routes call into this service and never touch
TensorFlow directly (Section 17/18 of the spec).

The model is loaded exactly once, at import/app-start time, never per
request, and this module never calls model.fit(...).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the trained model cannot be loaded at startup."""


class AutoencoderService:
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None

    def load(self) -> None:
        """
        Loads the Keras model once. No custom_objects are required -
        the notebook's model is a plain Sequential of Conv2D / MaxPool2D /
        UpSampling2D layers with no custom losses, metrics, or Lambda layers.
        """
        try:
            # Imported here (not at module import time) so that a missing/
            # broken TensorFlow install surfaces as a clean, logged error
            # instead of crashing the whole app at import time.
            from tensorflow import keras
        except ImportError as exc:
            logger.exception("TensorFlow/Keras is not installed.")
            raise ModelLoadError(
                "TensorFlow is not installed in this environment."
            ) from exc

        try:
            self._model = keras.models.load_model(self._model_path)
            logger.info("Model loaded successfully from %s", self._model_path)
        except (OSError, ValueError) as exc:
            logger.exception("Failed to load model from %s", self._model_path)
            raise ModelLoadError(
                f"Could not load model file at '{self._model_path}'."
            ) from exc

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """
        Runs inference only. model_input must already be shaped
        (1, 28, 28, 1) and normalized to [0, 1] - preprocessing is the
        caller's job (see image_utils.py), keeping this class a thin,
        testable wrapper around model.predict().
        """
        if self._model is None:
            raise ModelLoadError("Model is not loaded.")

        try:
            output = self._model.predict(model_input, verbose=0)
        except Exception as exc:  # noqa: BLE001 - convert any TF error to a clean one
            logger.exception("Prediction failed.")
            raise RuntimeError("Prediction failed.") from exc

        return output

    @property
    def input_shape(self):
        if self._model is None:
            return None
        return self._model.input_shape

    @property
    def output_shape(self):
        if self._model is None:
            return None
        return self._model.output_shape


# Module-level singleton, initialized once by the Flask app factory
# in app/__init__.py and reused across all requests.
autoencoder_service: AutoencoderService | None = None


def init_autoencoder_service(model_path: str) -> AutoencoderService:
    global autoencoder_service
    autoencoder_service = AutoencoderService(model_path)
    autoencoder_service.load()
    return autoencoder_service


def get_autoencoder_service() -> AutoencoderService:
    if autoencoder_service is None:
        raise ModelLoadError("Autoencoder service has not been initialized.")
    return autoencoder_service
