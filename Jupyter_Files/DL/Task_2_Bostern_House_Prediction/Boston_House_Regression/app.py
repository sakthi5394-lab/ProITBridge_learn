from flask import Flask, render_template, request
from src.predictor import predict_price
import keras

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    CRIM = float(request.form["CRIM"])
    ZN = float(request.form["ZN"])
    INDUS = float(request.form["INDUS"])
    CHAS = float(request.form["CHAS"])
    NOX = float(request.form["NOX"])
    RM = float(request.form["RM"])
    AGE = float(request.form["AGE"])
    DIS = float(request.form["DIS"])
    RAD = float(request.form["RAD"])
    TAX = float(request.form["TAX"])
    PTRATIO = float(request.form["PTRATIO"])
    B = float(request.form["B"])
    LSTAT = float(request.form["LSTAT"])

    input_data = [
        CRIM,
        ZN,
        INDUS,
        CHAS,
        NOX,
        RM,
        AGE,
        DIS,
        RAD,
        TAX,
        PTRATIO,
        B,
        LSTAT
    ]


    prediction = predict_price(input_data)

    return render_template(
        "result.html",
        prediction=round(prediction,2)
    )


if __name__ == "__main__":
    app.run(debug=True)