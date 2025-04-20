import os
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# Initialize Flask app
app = Flask(__name__)

# Load trained XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model("xgboost_model.json")

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Expected features
expected_features = ['qty', 'freight_price', 'comp_1', 'ps1', 'fp1', 
                     'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']

# HTML form template
form_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Kamal Price Prediction</title>
</head>
<body>
    <h2>Kamal Price Prediction Form</h2>
    <form method="POST" action="/predict">
        {% for field in fields %}
            <label for="{{ field }}">{{ field }}:</label>
            <input type="text" name="{{ field }}" required><br><br>
        {% endfor %}
        <input type="submit" value="Predict">
    </form>

    {% if prediction is not none %}
        <h3>Predicted Price: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

# Home route to display form (GET only)
@app.route("/predict", methods=["GET"])
def index():
    return render_template_string(form_template, fields=expected_features, prediction=None)

# Handle form submission (POST only)
@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        # Extract input values from form and convert to float
        input_values = [float(request.form[field]) for field in expected_features]

        # Create a DataFrame
        df = pd.DataFrame([input_values], columns=expected_features)

        # Scale input
        df_scaled = scaler.transform(df)

        # Convert to DMatrix
        dmatrix = xgb.DMatrix(df_scaled)

        # Make prediction
        prediction = xgb_model.predict(dmatrix)[0]

        # Render form again with prediction result
        return render_template_string(form_template, fields=expected_features, prediction=round(prediction, 2))

    except Exception as e:
        return f"Error: {e}"

# API endpoint for JSON-based prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        df = df[expected_features].fillna(0)

        df_scaled = scaler.transform(df)
        dmatrix = xgb.DMatrix(df_scaled)
        predictions = xgb_model.predict(dmatrix)

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
