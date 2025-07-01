from fastapi import FastAPI
import mlflow.sklearn
import pickle
import sys
import os
from pathlib import Path

# Add src/ to sys.path to ensure imports work
sys.path.append(str(Path(__file__).parent.parent))

from api.pydantic_models import CustomerData, PredictionResponse
from data_processing import preprocess_data
import pandas as pd
from sklearn.impute import SimpleImputer

app = FastAPI()

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Load model and preprocessor
try:
    model = mlflow.sklearn.load_model("models:/gradient_boosting/1")
    with open('../models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Model or preprocessor not found. Ensure 'data_processing.py' and 'train.py' have been run successfully.")

# Initialize imputer (consistent with train.py)
imputer = SimpleImputer(strategy='median')

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    # Convert input data to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Preprocess data using the same pipeline as training
    processed_data, _, _ = preprocess_data(df, fit=False, preprocessor=preprocessor)
    
    # Apply imputer to match training preprocessing
    processed_data = imputer.transform(processed_data)
    
    # Make prediction
    probability = model.predict_proba(processed_data)[:, 1][0]
    
    return PredictionResponse(probability=probability)