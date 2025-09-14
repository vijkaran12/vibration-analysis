from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from typing import List

# ----- Config -----
FRONTEND_URL = "https://preview--vibe-sense-dash.lovable.app"
MODEL_PATH = "model/my1_model.pkl"

# ----- Load Model -----
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

EXPECTED_FEATURES = model.n_features_in_

# ----- Initialize FastAPI -----
app = FastAPI()

# ----- Enable CORS -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- In-memory storage -----
past_predictions = []

# ----- Schemas -----
class Features(BaseModel):
    data: List[float]

# ----- Helper function for safe confidence -----
def get_confidence(proba_array: np.ndarray) -> float:
    # If single-class model, return 1.0
    if proba_array.shape[1] == 1:
        return 1.0
    return float(proba_array.max())

# ----- Manual JSON Prediction -----
@app.post("/predict")
def predict(features: Features):
    if len(features.data) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, got {len(features.data)}"
        )

    X = np.array(features.data, dtype=float).reshape(1, -1)
    pred = str(model.predict(X)[0])
    confidence = get_confidence(model.predict_proba(X))

    record = {
        "input": features.data,
        "prediction": pred,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }
    past_predictions.append(record)
    return [record]  # top-level array for Lovable

# ----- CSV Prediction -----
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if df.shape[1] != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} columns, got {df.shape[1]}"
        )

    preds = model.predict(df)
    probas = model.predict_proba(df)

    predictions_list = []
    for i, row in df.iterrows():
        confidence = get_confidence(probas[i].reshape(1, -1))
        record = {
            "input": row.tolist(),
            "prediction": str(preds[i]),
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        predictions_list.append(record)
        past_predictions.append(record)

    return predictions_list  # top-level array

# ----- Past Predictions -----
@app.get("/past_predictions")
def get_past_predictions():
    return past_predictions  # top-level array
