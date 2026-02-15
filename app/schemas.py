from pydantic import BaseModel
from typing import Dict


class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class PredictionResponse(BaseModel):
    model: str
    probability: float
    prediction: int
    risk_category: str
    model_version: str


class ExplanationResponse(BaseModel):
    shap_values: Dict[str, float]
    base_value: float
