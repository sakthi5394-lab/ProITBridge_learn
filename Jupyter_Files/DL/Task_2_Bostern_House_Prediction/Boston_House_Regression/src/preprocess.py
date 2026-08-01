from sklearn.preprocessing import StandardScaler

def data_preprocess(X_train, X_test):
    scaller = StandardScaler()
    X_train = scaller.fit_transform(X_train)
    X_test = scaller.fit_transform(X_test)
    return scaller,X_train,X_test