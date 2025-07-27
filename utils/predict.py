import numpy as np
import pandas as pd
import pickle

# Load model only once
with open("models/grid_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the expected features used during training
FEATURES = ["Voltage", "Current", "Temperature", "Load"]

def make_prediction(df):
    try:
        # Drop the target column if it exists
        if "Fault" in df.columns:
            df = df.drop(columns=["Fault"])

        # Ensure only the features used in training are passed
        df = df[[col for col in FEATURES if col in df.columns]]

        # Reorder columns to match model training
        df = df.reindex(columns=FEATURES)

        # Make sure we are predicting with a valid model
        if not hasattr(model, "predict"):
            return "Model file is corrupted or not a valid ML model."

        # Make prediction
        predictions = model.predict(df)

        return predictions.tolist()

    except Exception as e:
        return f"Prediction error: {str(e)}"
