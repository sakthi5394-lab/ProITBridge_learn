from pickle import dump
import keras

def save_module(scaller,bos_module):
    with open("models/scaler.pkl", "wb") as f:
        dump(scaller, f)

    bos_module.save("models/bos_module.h5")
