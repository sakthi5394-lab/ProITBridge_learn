"""
prediction_routes.py
-----------------------
HTTP layer only. Every route here validates input, delegates to the
service/util layers, and translates results/exceptions into clean JSON -
no TensorFlow calls and no image math happens directly in this file.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, render_template, request

from app.services.autoencoder_service import ModelLoadError, get_autoencoder_service
from app.utils.image_utils import (
    add_synthetic_noise,
    array_to_base64_png,
    compute_difference_map,
    compute_mse,
    from_model_output,
    image_to_normalized_array,
    load_grayscale_image,
    resize_to_model_input,
    to_model_input,
)
from app.utils.validation_utils import validate_upload

logger = logging.getLogger(__name__)

bp = Blueprint("prediction", __name__)


@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@bp.route("/api/predict", methods=["POST"])
def predict():
    file_storage = request.files.get("file")

    # --- 1. Validate the upload -----------------------------------------
    validation = validate_upload(file_storage)
    if not validation.ok:
        return jsonify({"success": False, "error": validation.error_message}), 400

    add_noise = request.form.get("add_noise", "false").lower() == "true"

    try:
        raw_bytes = file_storage.read()

        # --- 2. Preprocess -------------------------------------------------
        image = load_grayscale_image(raw_bytes)
        image = resize_to_model_input(image)
        clean_array = image_to_normalized_array(image)

        model_source_array = (
            add_synthetic_noise(clean_array) if add_noise else clean_array
        )
        model_input = to_model_input(model_source_array)
        logger.info("Image preprocessing completed (add_noise=%s)", add_noise)

        # --- 3. Predict ------------------------------------------------
        service = get_autoencoder_service()
        raw_output = service.predict(model_input)
        reconstructed_array = from_model_output(raw_output)
        logger.info("Prediction completed")

        # --- 4. Postprocess / metrics -----------------------------------
        mse = compute_mse(clean_array, reconstructed_array)
        difference_array = compute_difference_map(clean_array, reconstructed_array)
        logger.info("Reconstruction error calculated: %.6f", mse)

        response = {
            "success": True,
            "reconstruction_error": round(mse, 6),
            "original_image": array_to_base64_png(clean_array),
            "model_input_image": array_to_base64_png(model_source_array),
            "reconstructed_image": array_to_base64_png(reconstructed_array),
            "difference_image": array_to_base64_png(difference_array),
            "noise_applied": add_noise,
        }
        return jsonify(response), 200

    except ModelLoadError:
        logger.exception("Model unavailable during prediction request.")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Prediction service is temporarily unavailable. Please try again.",
                }
            ),
            503,
        )
    except (OSError, ValueError) as exc:
        logger.exception("Image processing failed: %s", exc)
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Unable to process this image. Please upload a clear "
                    "grayscale handwritten digit image.",
                }
            ),
            400,
        )
    except Exception:  # noqa: BLE001 - last-resort guard, never leak a traceback
        logger.exception("Unexpected error during prediction.")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Prediction service is temporarily unavailable. Please try again.",
                }
            ),
            500,
        )


@bp.route("/api/health", methods=["GET"])
def health():
    try:
        service = get_autoencoder_service()
    except ModelLoadError:
        return jsonify({"status": "unavailable", "model_loaded": False}), 503

    if not service.is_loaded:
        return jsonify({"status": "unavailable", "model_loaded": False}), 503

    return jsonify({"status": "ok", "model_loaded": True}), 200
