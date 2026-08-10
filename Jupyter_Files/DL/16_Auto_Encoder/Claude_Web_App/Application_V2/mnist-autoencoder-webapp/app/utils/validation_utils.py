"""
validation_utils.py
--------------------
Validates uploaded files before they ever reach the model.

Responsibilities:
    * Extension / MIME whitelist checks
    * File size checks
    * Confirming the file is actually a readable image (not just named .png)

Nothing here touches TensorFlow or the model - this module only decides
whether a file is safe and well-formed enough to proceed to preprocessing.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg"}

# Keep this in sync with Flask's MAX_CONTENT_LENGTH in config.py.
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

# Sanity bounds on the *uploaded* image before we resize it down to 28x28.
# Anything absurdly small is probably not a real digit; anything absurdly
# large is probably not an MNIST-style crop and likely a mistaken upload.
MIN_IMAGE_DIMENSION = 8
MAX_IMAGE_DIMENSION = 4096


@dataclass
class ValidationResult:
    ok: bool
    error_message: str | None = None


def has_allowed_extension(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def has_allowed_mime_type(mimetype: str | None) -> bool:
    if not mimetype:
        # Some browsers/clients omit this; we don't hard-fail on it alone
        # since the real content check (can Pillow open it?) is authoritative.
        return True
    return mimetype in ALLOWED_MIME_TYPES


def validate_upload(file_storage) -> ValidationResult:
    """
    Runs the full validation pipeline against a Flask FileStorage object.
    Returns a ValidationResult; does not raise.
    """
    if file_storage is None or file_storage.filename == "":
        return ValidationResult(False, "Please select an image.")

    if not has_allowed_extension(file_storage.filename):
        return ValidationResult(
            False, "Unsupported image format. Please upload PNG, JPG, or JPEG."
        )

    if not has_allowed_mime_type(file_storage.mimetype):
        return ValidationResult(
            False, "Unsupported image format. Please upload PNG, JPG, or JPEG."
        )

    # Read bytes once, check size, then hand back a fresh buffer for Pillow.
    raw_bytes = file_storage.read()
    file_storage.seek(0)  # reset stream in case caller reads it again

    if len(raw_bytes) == 0:
        return ValidationResult(False, "The uploaded file is empty.")

    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        return ValidationResult(
            False,
            f"File is too large. Maximum allowed size is "
            f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()  # cheap structural check
    except (UnidentifiedImageError, OSError):
        return ValidationResult(
            False, "The uploaded file could not be processed."
        )

    # Re-open after verify() (verify() leaves the image unusable for further ops)
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        width, height = image.size
    except (UnidentifiedImageError, OSError):
        return ValidationResult(False, "The uploaded file could not be processed.")

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return ValidationResult(
            False,
            "Unable to process this image. Please upload a clear grayscale "
            "handwritten digit image.",
        )

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        return ValidationResult(
            False,
            "Unable to process this image. Please upload a clear grayscale "
            "handwritten digit image.",
        )

    return ValidationResult(True)
