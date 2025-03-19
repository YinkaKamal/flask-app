# Import libraries
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r"C:\Users\usmank\OneDrive - Access-ARM Pensions\Documents\Nexford Assignments\Business Analytics Capstone\Final Project\Business analytics ready dataset (Final Project).csv"
data = pd.read_csv(file_path)

# Select features and target variable
features = ['qty', 'freight_price', 'comp_1', 'ps1', 'fp1', 
            'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']
target = 'unit_price'

X = data[features]
y = data[target]

# Handle missing values
X = X.fillna(X.mean())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Save the model properly
xgb_model.save_model("xgboost_model.json")

# Save the scaler separately
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Evaluate the model
y_pred = xgb_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
print("Model and scaler saved successfully!")
