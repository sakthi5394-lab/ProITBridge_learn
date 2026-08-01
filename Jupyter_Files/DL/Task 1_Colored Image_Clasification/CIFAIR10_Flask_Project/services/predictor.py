from keras.models import load_model 
from services.preprocess import preprocess_image
from model.labels import CLASS_NAMES


class Predictor:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, image_path):
        image = preprocess_image(image_path)
        prediction = self.model.predict(image)
        index = prediction.argmax()
        confidence = prediction.max()
        return CLASS_NAMES[index], confidence