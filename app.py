from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
# Initialize Flask app
app = Flask(__name__)
         
CORS(app)   # Enable CORS for all routes
# Load saved models
try:
    heart_model = pickle.load(open("Models/heart.sav", "rb"))
    diabetes_model = pickle.load(open("Models/diabetes.sav", "rb"))
    liver_model = pickle.load(open("Models/liver_model.sav", "rb"))
    lung_cancer_model = pickle.load(open("Models/lung cancer.sav", "rb"))
except Exception as e:
    print(f"Error loading models: {str(e)}")

# Define route for the home page
@app.route("/")
def home():
    return "Welcome to the Multiple Disease Prediction API!"

# Heart Disease Prediction
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        data = request.json  # Input data in JSON format
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in the request data"}), 400
        features = np.array(data["features"]).reshape(1, -1)
        prediction = heart_model.predict(features)
        result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": f"Error processing the request: {str(e)}"}), 500

# Diabetes Prediction
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.json  # Input data in JSON format
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in the request data"}), 400
        features = np.array(data["features"]).reshape(1, -1)
        prediction = diabetes_model.predict(features)
        result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": f"Error processing the request: {str(e)}"}), 500

# Liver Disease Prediction
@app.route("/predict_liver", methods=["POST"])
def predict_liver():
    try:
        data = request.json  # Input data in JSON format
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in the request data"}), 400
        features = np.array(data["features"]).reshape(1, -1)
        prediction = liver_model.predict(features)
        result = "Positive for Liver Disease" if prediction[0] == 1 else "Negative for Liver Disease"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": f"Error processing the request: {str(e)}"}), 500

# Lung Cancer Prediction
@app.route("/predict_lung_cancer", methods=["POST"])
def predict_lung_cancer():
    try:
        data = request.json  # Input data in JSON format
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in the request data"}), 400
        features = np.array(data["features"]).reshape(1, -1)
        prediction = lung_cancer_model.predict(features)
        result = "Positive for Lung Cancer" if prediction[0] == 1 else "Negative for Lung Cancer"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": f"Error processing the request: {str(e)}"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
       