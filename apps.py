import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

apps = Flask(__name__)

# Load the trained XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model("xgboost_model.json")

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@apps.route('/')
def home():
    return "Welcome to the Price Prediction API!"

@apps.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input JSON data
        data = request.get_json()
        df = pd.DataFrame(data)

        # Ensure correct feature order
        expected_features = ['qty', 'freight_price', 'comp_1', 'ps1', 'fp1', 
                             'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']
        df = df[expected_features]

        # Handle missing values if any
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

if __name__ == '__main__':
    apps.run(host='0.0.0.0', port=8080)
