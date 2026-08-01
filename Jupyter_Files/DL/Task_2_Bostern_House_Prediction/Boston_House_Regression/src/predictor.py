import pickle
import numpy as np
import keras.models

model = keras.models.load_model("models/bos_module.h5")
with open("models/scaler.pkl","rb") as f:
    scaler = pickle.load(f)


def predict_price(values):
    data = np.array(values).reshape(1,-1)
    data = scaler.transform(data)
    prediction = model.predict(data)

    return prediction[0][0]