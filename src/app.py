"""
app.py — FastAPI Churn Prediction Service
Endpoints: POST /predict | GET /health | GET /model-info
"""

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Paths ───────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
ART_DIR    = BASE_DIR / "artifacts"
DATA_DIR   = BASE_DIR / "data" / "processed"

# ─── Load artefacts at startup ───────────────────────────
print("Loading model artefacts...")

with open(ART_DIR / "model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(ART_DIR / "explainer.pkl", "rb") as f:
    EXPLAINER = pickle.load(f)

with open(ART_DIR / "features.json") as f:
    FEATURES: list[str] = json.load(f)

with open(DATA_DIR / "encoding_map.json") as f:
    ENCODING_MAP: dict = json.load(f)

with open(ART_DIR / "best_params.json") as f:
    BEST_PARAMS: dict = json.load(f)

MODEL_LOADED_AT = datetime.utcnow().isoformat() + "Z"
print(f"✅ Model ready | features: {len(FEATURES)} | loaded at: {MODEL_LOADED_AT}")

# ─── FastAPI app ─────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Prediction API",
    description="XGBoost + SHAP churn prediction service. Built for K21 Academy AI PM course.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic schemas ────────────────────────────────────
class CustomerRecord(BaseModel):
    gender: str = Field("Male", description="Male or Female")
    SeniorCitizen: int = Field(0, ge=0, le=1, description="1 if senior citizen, else 0")
    Partner: str = Field("No", description="Yes or No")
    Dependents: str = Field("No", description="Yes or No")
    tenure: int = Field(12, ge=0, le=72, description="Months with company")
    PhoneService: str = Field("Yes", description="Yes or No")
    MultipleLines: str = Field("No", description="Yes / No / No phone service")
    InternetService: str = Field("DSL", description="DSL / Fiber optic / No")
    OnlineSecurity: str = Field("No", description="Yes / No / No internet service")
    OnlineBackup: str = Field("No", description="Yes / No / No internet service")
    DeviceProtection: str = Field("No", description="Yes / No / No internet service")
    TechSupport: str = Field("No", description="Yes / No / No internet service")
    StreamingTV: str = Field("No", description="Yes / No / No internet service")
    StreamingMovies: str = Field("No", description="Yes / No / No internet service")
    Contract: str = Field("Month-to-month", description="Month-to-month / One year / Two year")
    PaperlessBilling: str = Field("Yes", description="Yes or No")
    PaymentMethod: str = Field("Electronic check", description="Payment method")
    MonthlyCharges: float = Field(65.0, ge=0.0, description="Monthly charge amount")
    TotalCharges: float = Field(780.0, ge=0.0, description="Total charges to date")

    model_config = {"json_schema_extra": {
        "example": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
            "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.5, "TotalCharges": 427.5,
        }
    }}


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: bool
    churn_label: str
    top_3_features: list[dict]
    all_feature_contributions: dict
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_version: str
    algorithm: str
    training_date: str
    n_features: int
    features: list[str]
    best_val_auc: float
    best_params: dict


# ─── Helper ──────────────────────────────────────────────
def preprocess_input(record: CustomerRecord) -> pd.DataFrame:
    """Convert raw CustomerRecord into the engineered feature DataFrame."""
    data = record.model_dump()

    # Encode binary columns
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        data[col] = 1 if data[col] == "Yes" else 0

    # Encode multi-category columns using saved label encoding
    multi_cat_cols = [
        "gender", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    for col in multi_cat_cols:
        if col in ENCODING_MAP:
            inv_map = {v: int(k) for k, v in ENCODING_MAP[col].items()}
            raw_val = str(data[col])
            data[col] = inv_map.get(raw_val, 0)

    # Engineered features
    tenure = data["tenure"]
    monthly = data["MonthlyCharges"]
    total = data["TotalCharges"]

    data["avg_monthly_spend"] = (total / tenure) if tenure > 0 else monthly
    data["tenure_band"] = min(int(tenure // 12), 3)
    data["charge_delta"] = monthly - data["avg_monthly_spend"]
    data["streaming_count"] = data["StreamingTV"] + data["StreamingMovies"]
    data["support_count"] = (
        data["OnlineSecurity"] + data["TechSupport"] +
        data["OnlineBackup"] + data["DeviceProtection"]
    )
    data["is_mtm_contract"] = 1 if data["Contract"] == 0 else 0

    row = pd.DataFrame([data])[FEATURES]
    return row


# ─── Routes ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    return ModelInfoResponse(
        model_version="1.0.0-optuna",
        algorithm="XGBoostClassifier",
        training_date=MODEL_LOADED_AT,
        n_features=len(FEATURES),
        features=FEATURES,
        best_val_auc=BEST_PARAMS.get("val_auc", 0.0),
        best_params=BEST_PARAMS.get("params", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerRecord):
    t0 = time.perf_counter()

    try:
        row = preprocess_input(customer)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering error: {e}")

    try:
        proba = float(MODEL.predict_proba(row)[0, 1])
        prediction = proba >= 0.5

        # SHAP contributions
        shap_vals = EXPLAINER.shap_values(row)[0]
        contributions = {
            feat: round(float(val), 6)
            for feat, val in zip(FEATURES, shap_vals)
        }
        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_3 = [
            {"feature": k, "contribution": v, "direction": "increases_churn" if v > 0 else "decreases_churn"}
            for k, v in sorted_contribs[:3]
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    latency = round((time.perf_counter() - t0) * 1000, 2)

    return PredictionResponse(
        churn_probability=round(proba, 4),
        prediction=prediction,
        churn_label="CHURN" if prediction else "RETAIN",
        top_3_features=top_3,
        all_feature_contributions=dict(sorted_contribs),
        latency_ms=latency,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)