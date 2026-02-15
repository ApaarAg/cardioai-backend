import pandas as pd
from app.services.model_loader import xgb_model
from app.config import MODEL_VERSION


def predict(data: dict, model_type: str = "xgboost"):

    df = pd.DataFrame([data])

    prediction = int(xgb_model.predict(df)[0])
    probability = float(xgb_model.predict_proba(df)[0][1])

    if probability < 0.3:
        risk = "Low"
    elif probability < 0.7:
        risk = "Moderate"
    else:
        risk = "High"

    return {
        "model": model_type,
        "probability": probability,
        "prediction": prediction,
        "risk_category": risk,
        "model_version": MODEL_VERSION,
    }
