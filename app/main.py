from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from typing import List

# ----- Config -----
FRONTEND_URL = "https://preview--vibe-sense-dash.lovable.app"
MODEL_PATH = "model/my_model.pkl"

# ----- Load Model -----
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----- Initialize FastAPI -----
app = FastAPI(
    title="Vibration Prediction API",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
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

class CsvPrediction(BaseModel):
    input: List[float]
    prediction: str
    confidence: float
    timestamp: str

class CsvResponse(BaseModel):
    predictions: List[CsvPrediction]

class PastPredictionsResponse(BaseModel):
    history: List[CsvPrediction]

# ----- Root / Health -----
@app.get("/")
def root():
    return {"message": "Welcome to the Prediction API"}

@app.get("/health")
def health():
    return {"status": "Server is running"}

# ----- Manual JSON Prediction (Lovable-compatible) -----
@app.post("/predict", response_model=CsvResponse)
def predict(features: Features):
    try:
        X = np.array(features.data).reshape(1, -1)
        if X.shape[1] != model.n_features_in_:
            raise ValueError(f"Expected {model.n_features_in_} features, got {X.shape[1]}")

        pred = str(model.predict(X)[0])
        confidence = float(model.predict_proba(X).max())

        record = CsvPrediction(
            input=features.data,
            prediction=pred,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        past_predictions.append(record.dict())

        # Return as an array even for single prediction
        return {"predictions": [record]}
    except Exception as e:
        return {"predictions": []}

# ----- CSV Prediction -----
@app.post("/predict_csv", response_model=CsvResponse)
async def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if df.shape[1] != model.n_features_in_:
            raise ValueError(f"Expected {model.n_features_in_} columns, got {df.shape[1]}")

        preds = model.predict(df)
        confidences = model.predict_proba(df).max(axis=1)

        predictions_list = []
        for i, row in df.iterrows():
            record = CsvPrediction(
                input=row.tolist(),
                prediction=str(preds[i]),
                confidence=float(confidences[i]),
                timestamp=datetime.now().isoformat()
            )
            predictions_list.append(record)
            past_predictions.append(record.dict())

        return {"predictions": predictions_list}
    except Exception as e:
        return {"predictions": []}

# ----- Past Predictions -----
@app.get("/past_predictions", response_model=PastPredictionsResponse)
def get_past_predictions():
    return {"history": past_predictions}
