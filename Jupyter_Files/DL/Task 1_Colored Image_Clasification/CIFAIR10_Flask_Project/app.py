import os
from flask import Flask,render_template,request
from config import MODEL_PATH,UPLOAD_FOLDER
from services.predictor import Predictor
from utils.file_handler import save_file


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
predictor = Predictor(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No File"

    file = request.files["image"]
    filename, path = save_file(file,
                               app.config["UPLOAD_FOLDER"])

    label, confidence = predictor.predict(path)

    return render_template(
        "index.html",
        prediction=label,
        confidence=round(confidence * 100, 2),
        image=filename
    )


if __name__ == "__main__":

    app.run(debug=True)