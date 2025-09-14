from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# ----- Load Model -----
with open("model/my_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----- Initialize FastAPI -----
app = FastAPI()

# ----- In-memory storage for past predictions -----
past_predictions = []

# ----- JSON Input Schema -----
class Features(BaseModel):
    data: list[float]   # Example: [0.23, 0.87, 0.12, 3.45]

# ----- Root / Health Endpoint -----
@app.get("/")
def main():
    return {"message": "Welcome to the Prediction API"}

@app.get("/health")
def health():
    return {"status": "Server is running"}

# ----- JSON Prediction Endpoint -----
@app.post("/predict")
def predict(features: Features):
    X = np.array(features.data).reshape(1, -1)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X).max()
    
    # Store prediction with timestamp
    record = {
        "type": "json",
        "input": features.data,
        "prediction": str(pred),
        "confidence": float(proba),
        "timestamp": datetime.now().isoformat()
    }
    past_predictions.append(record)
    
    return {"prediction": str(pred), "confidence": float(proba)}

# ----- CSV Prediction Endpoint -----
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    
    preds = model.predict(df)
    probas = model.predict_proba(df).max(axis=1)
    
    results = df.copy()
    results["prediction"] = preds
    results["confidence"] = probas
    results["timestamp"] = datetime.now().isoformat()
    
    # Store each row in past_predictions
    for _, row in results.iterrows():
        past_predictions.append({
            "type": "csv",
            "input": row[df.columns].tolist(),
            "prediction": row["prediction"],
            "confidence": row["confidence"],
            "timestamp": row["timestamp"]
        })
    
    return results.to_dict(orient="records")

# ----- GET Endpoint to Fetch Past Predictions -----
@app.get("/past_predictions")
def get_past_predictions():
    return past_predictions
