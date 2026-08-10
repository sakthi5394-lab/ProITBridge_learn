"""
config.py
---------
Central configuration. Values that are inherent to the trained model
(IMAGE_SIZE, NORMALIZATION range, NOISE_FACTOR) are pinned here with a
comment pointing back to the exact notebook line they came from, so
nobody accidentally "tweaks" them out of sync with the actual model.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # --- Model -----------------------------------------------------------
    # De_Nosing_Image_AUtoEncoder.ipynb, cell 21: input_shape=(28,28,1)
    MODEL_PATH = os.environ.get(
        "MODEL_PATH", os.path.join(BASE_DIR, "models", "denoise_image.h5")
    )
    IMAGE_SIZE = 28          # notebook: X_train.reshape(60000, 28, 28, 1)
    IMAGE_CHANNELS = 1       # grayscale
    NOISE_FACTOR = 0.3       # notebook cell 14: 0.3 * np.random.normal(...)

    # --- Uploads -----------------------------------------------------------
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB, enforced by Flask itself
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

    # --- Flask -----------------------------------------------------------
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    JSON_SORT_KEYS = False
