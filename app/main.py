from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pickle
import pandas as pd
from typing import List

# ----- Config -----
FRONTEND_URL = "https://preview--vibe-sense-dash.lovable.app"
MODEL_PATH = "model/my1_model.pkl"

# ----- Load Model -----
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURE_NAMES = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
EXPECTED_FEATURES = len(FEATURE_NAMES)

# ----- Initialize FastAPI -----
app = FastAPI(
    title="Vibration Prediction API",
    version="1.0",
)

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

# ----- Root / Health -----
@app.get("/")
def root():
    return {"message": "Welcome to the Prediction API"}

@app.get("/health")
def health():
    return {"status": "Server is running"}

# ----- Manual JSON Prediction -----
@app.post("/predict")
def predict(features: Features):
    if len(features.data) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, got {len(features.data)}"
        )

    try:
        # Convert input to DataFrame with correct feature names
        X = pd.DataFrame([features.data], columns=FEATURE_NAMES)
        X = X.astype(float)

        pred = str(model.predict(X)[0])
        confidence = float(model.predict_proba(X).max())

        record = {
            "input": features.data,
            "prediction": pred,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }

        past_predictions.append(record)
        return [record]  # top-level list for Lovable

    except Exception as e:
        return [{"input": features.data, "prediction": "error", "confidence": 0.0, "timestamp": datetime.now().isoformat()}]

# ----- CSV Prediction -----
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if df.shape[1] != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {EXPECTED_FEATURES} columns, got {df.shape[1]}"
            )

        # Rename columns to match model
        df.columns = FEATURE_NAMES
        df = df.astype(float)

        preds = model.predict(df)
        confidences = model.predict_proba(df).max(axis=1)

        predictions_list = []
        for i, row in df.iterrows():
            record = {
                "input": row.tolist(),
                "prediction": str(preds[i]),
                "confidence": float(confidences[i]),
                "timestamp": datetime.now().isoformat()
            }
            predictions_list.append(record)
            past_predictions.append(record)

        return predictions_list  # top-level list for Lovable

    except Exception as e:
        return []

# ----- Past Predictions -----
@app.get("/past_predictions")
def get_past_predictions():
    return past_predictions  # top-level list for Lovable
