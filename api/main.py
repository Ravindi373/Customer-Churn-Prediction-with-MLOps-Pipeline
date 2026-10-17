from pathlib import Path
from typing import Any, Dict, Optional

import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai


app = FastAPI(title="Churn Prediction API", version="1.0")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"

model = None
preprocessor = None


class PredictRequest(BaseModel):
    # send raw fields as key/value pairs
    features: Dict[str, Any]


@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessor: {PREPROCESSOR_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    # Create a 1-row DataFrame from raw input
    X_raw = pd.DataFrame([req.features])

    # Transform using saved preprocessor
    X = preprocessor.transform(X_raw)

    # Predict probability of churn = 1
    proba = float(model.predict_proba(X)[0][1])
    prediction = "Yes" if proba >= 0.5 else "No"
    
    response_data = {
        "churn_probability": round(proba, 4),
        "prediction": prediction,
    }

    # Bonus Task: LLM Retention Generator
    if prediction == "Yes":
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                client = genai.Client(api_key=api_key)
                
                customer_tenure = req.features.get("tenure", "unknown")
                customer_service = req.features.get("InternetService", "Standard")
                customer_charges = req.features.get("MonthlyCharges", "unknown")

                prompt = f"""
                You are an expert customer retention agent for a Telecom company.
                A customer is about to cancel their subscription. 
                They have been with us for {customer_tenure} months, pay ${customer_charges}/mo, and use {customer_service} internet.
                
                Task: Write a strictly 1-sentence personalized message offering a loyalty discount 
                or a free upgrade tailored to their current service to convince them to stay.
                Do not include any greetings or signatures, just the offer.
                """

                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                
                response_data["retention_incentive"] = response.text.strip()
            else:
                response_data["retention_incentive"] = "Gemini API key missing. Unable to generate incentive."
        except Exception as e:
            response_data["retention_incentive"] = f"LLM Generation Failed: {str(e)}"

    return response_data