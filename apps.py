import os
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# This is to initialize Flask app
app = Flask(__name__)

# This is to load the trained XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model("xgboost_model.json")  

# This is to load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return "Welcome to the Price Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input JSON data
        data = request.get_json()
        df = pd.DataFrame(data)

        # Ensure correct feature order
        expected_features = ['qty', 'freight_price', 'comp_1', 'ps1', 'fp1', 
                             'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']
        df = df[expected_features]

        # Handle missing values
        df = df.fillna(0)

        # Scale input data
        df_scaled = scaler.transform(df)

        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(df_scaled)

        # Make predictions
        predictions = xgb_model.predict(dmatrix)

        return jsonify({"predictions": predictions.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # Get port from environment variable (Render sets this dynamically)
    port = int(os.environ.get("PORT", 8080))
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=port)
