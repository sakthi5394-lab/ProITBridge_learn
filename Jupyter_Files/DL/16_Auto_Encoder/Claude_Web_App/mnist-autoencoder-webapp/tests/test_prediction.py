"""
test_prediction.py
--------------------
Model-loading and image-processing tests. These exercise the actual
trained model file, so they require TensorFlow + the real .h5 file to be
present under models/ - skip gracefully if either is unavailable (e.g. a
lightweight CI environment that only lints the Flask layer).

Run with: pytest tests/test_prediction.py -v
"""

import io

import numpy as np
import pytest
from PIL import Image

from app.config import Config
from app.utils.image_utils import (
    add_synthetic_noise,
    compute_difference_map,
    compute_mse,
    from_model_output,
    image_to_normalized_array,
    load_grayscale_image,
    resize_to_model_input,
    to_model_input,
)

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")


@pytest.fixture(scope="module")
def loaded_model():
    from tensorflow import keras

    try:
        return keras.models.load_model(Config.MODEL_PATH)
    except (OSError, ValueError):
        pytest.skip("Model file not found or failed to load.")


def _make_test_image_bytes():
    """A trivial synthetic RGB image standing in for an uploaded digit."""
    array = (np.random.rand(40, 40, 3) * 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# --------------------------- Model tests ---------------------------

def test_model_loads_successfully(loaded_model):
    assert loaded_model is not None


def test_model_input_shape(loaded_model):
    assert loaded_model.input_shape == (None, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1)


def test_model_prediction_output_shape(loaded_model):
    dummy_input = np.zeros((1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1), dtype="float32")
    output = loaded_model.predict(dummy_input, verbose=0)
    assert output.shape == (1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1)


# --------------------------- Image processing tests ---------------------------

def test_rgb_converts_to_grayscale():
    raw_bytes = _make_test_image_bytes()
    image = load_grayscale_image(raw_bytes)
    assert image.mode == "L"


def test_resize_to_model_dimensions():
    raw_bytes = _make_test_image_bytes()
    image = load_grayscale_image(raw_bytes)
    resized = resize_to_model_input(image)
    assert resized.size == (Config.IMAGE_SIZE, Config.IMAGE_SIZE)


def test_normalization_range():
    raw_bytes = _make_test_image_bytes()
    image = resize_to_model_input(load_grayscale_image(raw_bytes))
    array = image_to_normalized_array(image)
    assert array.min() >= 0.0
    assert array.max() <= 1.0
    assert array.dtype == np.float32


def test_noise_injection_stays_in_bounds():
    array = np.random.rand(Config.IMAGE_SIZE, Config.IMAGE_SIZE).astype("float32")
    noisy = add_synthetic_noise(array)
    assert noisy.min() >= 0.0
    assert noisy.max() <= 1.0
    assert noisy.shape == array.shape


def test_batch_dimension_added():
    array = np.random.rand(Config.IMAGE_SIZE, Config.IMAGE_SIZE).astype("float32")
    batched = to_model_input(array)
    assert batched.shape == (1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1)


def test_reconstruction_pipeline_end_to_end(loaded_model):
    raw_bytes = _make_test_image_bytes()
    image = resize_to_model_input(load_grayscale_image(raw_bytes))
    clean_array = image_to_normalized_array(image)
    model_input = to_model_input(clean_array)

    raw_output = loaded_model.predict(model_input, verbose=0)
    reconstructed = from_model_output(raw_output)

    assert reconstructed.shape == (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert reconstructed.min() >= 0.0
    assert reconstructed.max() <= 1.0

    mse = compute_mse(clean_array, reconstructed)
    assert mse >= 0.0

    diff = compute_difference_map(clean_array, reconstructed)
    assert diff.shape == clean_array.shape
