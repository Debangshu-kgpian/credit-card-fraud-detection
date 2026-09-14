"""
api.py
FastAPI serving layer for the trained XGBoost fraud detection model.
Exposes a /predict endpoint: send one transaction's engineered features,
get back a fraud probability and a flag based on the cost-optimal threshold
found in evaluation.py.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from src import config

FRAUD_THRESHOLD = 0.07  # from evaluation.py's cost-based optimization

app = FastAPI(title="Fraud Detection API")

model_path = os.path.join(config.MODELS_DIR, "xgboost_fraud_model.joblib")
model = joblib.load(model_path)


class Transaction(BaseModel):
    features: Dict[str, float]


@app.get("/health")
def health():
    """Basic check that the server is up and the model loaded correctly."""
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    """Accepts one transaction's already-engineered feature values as a
    dict, runs it through the trained model, and returns a fraud
    probability plus a flag based on the cost-optimal threshold."""
    row = pd.DataFrame([transaction.features])
    row = row.reindex(columns=model.feature_names_in_, fill_value=0)

    probability = float(model.predict_proba(row)[:, 1][0])
    is_fraud = probability >= FRAUD_THRESHOLD

    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": bool(is_fraud),
        "threshold_used": FRAUD_THRESHOLD,
    }