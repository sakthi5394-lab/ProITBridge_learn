from keras.models import Sequential
from keras.layers import Dense

def bos_module():
    bos_module = Sequential()
    bos_module.add(Dense( units = 64,activation="relu",input_shape = (13,))) ##Input Layer
    bos_module.add(Dense(32,activation= "relu"))
    bos_module.add(Dense(10,activation= "relu"))
    bos_module.add(Dense(1)) ## OuptLayer
    bos_module.compile(optimizer='Adam',loss="mean_squared_error",metrics = ["mean_absolute_error"])
    return  bos_module