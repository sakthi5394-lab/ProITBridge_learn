from keras.datasets import boston_housing
def Load_Boston_data():
    (X_train, y_train), (X_test, y_test) = boston_housing.load_data()
    return X_train, y_train,X_test, y_test