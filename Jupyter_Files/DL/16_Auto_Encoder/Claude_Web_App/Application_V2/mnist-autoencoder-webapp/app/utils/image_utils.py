"""
image_utils.py
---------------
All image transforms live here, and every transform mirrors what the
training notebook (De_Nosing_Image_AUtoEncoder.ipynb) actually did:

    1. Grayscale, resized to 28x28              (X_train.reshape(.., 28, 28, 1))
    2. Normalized to [0, 1] via /255.0           (X_train / 255.0)
    3. Optional synthetic noise + clip           (x + 0.3*N(0,1), np.clip(0,1))
       -> this matches X_train_Noise / X_train_Clip in the notebook.
       The model was trained noisy-in -> clean-out, so this step is what
       actually exercises the "denoising" behavior for a user-uploaded,
       already-clean digit photo.
    4. Batch + channel dims -> (1, 28, 28, 1)    (model.input_shape)

Nothing here is a guess: IMAGE_SIZE, NORMALIZATION, and the noise formula
are all read from config.py, which documents the notebook line each value
came from.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from app.config import Config


def load_grayscale_image(raw_bytes: bytes) -> Image.Image:
    """Open uploaded bytes and convert to single-channel grayscale ('L')."""
    image = Image.open(io.BytesIO(raw_bytes))
    return image.convert("L")


def resize_to_model_input(image: Image.Image) -> Image.Image:
    """Resize to the exact (width, height) the trained model expects."""
    size = (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    return image.resize(size, Image.LANCZOS)


def image_to_normalized_array(image: Image.Image) -> np.ndarray:
    """
    PIL Image (H, W) -> float32 array in [0, 1], shape (H, W).
    Mirrors: X_train = X_train.astype('float32') / 255.0
    """
    array = np.asarray(image).astype("float32") / 255.0
    return array


def add_synthetic_noise(array: np.ndarray, noise_factor: float = None) -> np.ndarray:
    """
    Adds Gaussian noise and clips to [0, 1], exactly matching the notebook:
        X_train_Noise = X_train + 0.3 * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
        X_train_Clip  = np.clip(a=X_train_Noise, a_min=0, a_max=1)
    Used only when the user opts in via the UI "add noise" toggle, so the
    denoising behavior is actually visible on a clean upload.
    """
    factor = Config.NOISE_FACTOR if noise_factor is None else noise_factor
    noisy = array + factor * np.random.normal(loc=0.0, scale=1.0, size=array.shape)
    return np.clip(noisy, 0.0, 1.0).astype("float32")


def to_model_input(array: np.ndarray) -> np.ndarray:
    """
    (H, W) float32 in [0,1] -> (1, H, W, 1) float32, matching
    model.input_shape == (None, 28, 28, 1).
    """
    batched = np.expand_dims(array, axis=0)   # batch dim
    batched = np.expand_dims(batched, axis=-1)  # channel dim
    return batched.astype("float32")


def from_model_output(output: np.ndarray) -> np.ndarray:
    """
    (1, H, W, 1) model output -> (H, W) float32 in [0,1].
    The model's final layer uses ReLU (not sigmoid), so output is not
    guaranteed to be bounded at 1.0 - we clip for correct, displayable
    reconstructions, without altering the model itself.
    """
    array = np.squeeze(output, axis=(0, -1))
    return np.clip(array, 0.0, 1.0).astype("float32")


def array_to_base64_png(array: np.ndarray) -> str:
    """
    float32 [0,1] (H, W) array -> base64-encoded PNG data URI, for
    embedding directly in the JSON API response / <img src="...">.
    """
    pixels = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")

    # Upscale for a crisper on-screen preview (28x28 is tiny). Nearest-neighbor
    # keeps pixel edges crisp rather than introducing blur into a diagnostic image.
    display_size = (Config.IMAGE_SIZE * 8, Config.IMAGE_SIZE * 8)
    image = image.resize(display_size, Image.NEAREST)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def compute_difference_map(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """abs(original - reconstructed), same shape, values in [0, 1]."""
    return np.abs(original - reconstructed).astype("float32")


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Mean Squared Error between original and reconstructed arrays.
    Matches the model's own training loss ("mean_squared_error"), so the
    number shown in the UI is directly comparable to training/eval metrics.
    """
    return float(np.mean(np.square(original - reconstructed)))
