import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

MODEL_PATH = os.path.join(BASE_DIR,
                          "model",
                          "CF_Keras_intel.h5")