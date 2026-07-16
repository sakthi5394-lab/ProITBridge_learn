import pickle
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, request, jsonify
from flask_cors import CORS  # Prevents browser block errors

app = Flask(__name__)
CORS(app)  # Allows your HTML file to safely talk to this backend

# 1. Load your pkl file (Replace 'your_model.pkl' with your actual file name)
with open('D:\MyLearning\Python_WS\LMS_ProITBridge\ML_Projects\Logestistic_regression\module\claim_module.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the HTML request
        request_data = request.json['data']
        
        # Convert data to a 2D array format that Scikit-Learn/models expect
        X = np.array([request_data])
        input_data = pd.DataFrame(X,columns =["CLMSEX","CLMINSUR","SEATBELT","CLMAGE","LOSS"])
        
        # 2. Make the prediction
        prediction = model.predict(input_data)
        print(prediction)
        
        # Return the prediction result as JSON
        return jsonify({'prediction': str(prediction[0])})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)