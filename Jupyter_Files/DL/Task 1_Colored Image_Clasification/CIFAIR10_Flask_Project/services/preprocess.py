import numpy as np
from PIL import Image


def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize((32, 32))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img