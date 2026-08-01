import keras
from src import data_loader, model_build,preprocess,save_module

def Module_train():
    X_train, y_train,X_test, y_test = data_loader.Load_Boston_data()
    scaller,X_train,X_test = preprocess.data_preprocess(X_train, X_test)
    bos_module = model_build.bos_module()
    bos_module.fit(x=X_train,
                    y=y_train,
                    batch_size=20,
                    epochs=100,
                    verbose=1,
                    validation_split = 0.2)
    loss,mae = bos_module.evaluate(X_test,y_test)
    print("Total Loss : ",loss)
    print("Mae : ",mae)
    save_module.save_module(scaller,bos_module)
    return loss,mae