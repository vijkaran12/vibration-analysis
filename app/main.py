from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# ----- Config -----
frontend_url = "https://preview--vibe-sense-dash.lovable.app"
MODEL_PATH = "model/my_model.pkl"

# ----- Load Model -----
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----- Initialize FastAPI -----
app = FastAPI()

# ----- Enable CORS -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],  # allow your Lovable frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- In-memory storage for past predictions -----
past_predictions = []

# ----- JSON Input Schema -----
class Features(BaseModel):
    data: list[float]  # e.g., [0.23, 0.87, 0.12, 3.45]

# ----- Root / Health Endpoints -----
@app.get("/")
def root():
    return {"message": "Welcome to the Prediction API"}

@app.get("/health")
def health():
    return {"status": "Server is running"}

# ----- JSON Prediction Endpoint -----
@app.post("/predict")
def predict(features: Features):
    try:
        X = np.array(features.data).reshape(1, -1)
        if X.shape[1] != model.n_features_in_:
            return {"error": f"Expected {model.n_features_in_} features, got {X.shape[1]}"}
        
        pred = model.predict(X)[0]
        proba = float(model.predict_proba(X).max())  # convert to Python float

        # Store prediction
        record = {
            "type": "json",
            "input": features.data,
            "prediction": str(pred),
            "confidence": proba,
            "timestamp": datetime.now().isoformat()
        }
        past_predictions.append(record)

        # Wrap in top-level "result" key for Lovable
        return {"status": "success", "result": {"prediction": str(pred), "confidence": proba}}

    except Exception as e:
        return {"error": str(e)}


# ----- CSV Prediction Endpoint -----
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if df.shape[1] != model.n_features_in_:
            return {"error": f"Expected {model.n_features_in_} columns, got {df.shape[1]}"}
        
        preds = model.predict(df)
        probas = model.predict_proba(df).max(axis=1)
        results = df.copy()
        results["prediction"] = preds
        results["confidence"] = probas
        results["timestamp"] = datetime.now().isoformat()

        # Store past predictions
        for _, row in results.iterrows():
            past_predictions.append({
                "type": "csv",
                "input": row[df.columns].tolist(),
                "prediction": row["prediction"],
                "confidence": row["confidence"],
                "timestamp": row["timestamp"]
            })

        # Wrap in a top-level object for Lovable
        return {"status": "success", "predictions": results.to_dict(orient="records")}

    except Exception as e:
        return {"error": str(e)}
