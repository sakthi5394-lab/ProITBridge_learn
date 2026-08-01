import keras

def Module_Evaluate(bos_module,X_test,y_test):
    loss,mae = bos_module.evaluate(X_test,y_test)
    return loss,mae