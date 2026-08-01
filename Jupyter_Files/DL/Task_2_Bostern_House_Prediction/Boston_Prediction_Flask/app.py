from flask import Flask, render_template, request
from src.predictor import predict_price
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    values = []

    for i in range(1,14):
        values.append(float(request.form[f"feature{i}"]))

    prediction = predict_price(values)

    return render_template(
        "result.html",
        prediction=round(prediction,2)
    )


if __name__ == "__main__":
    app.run(debug=True)