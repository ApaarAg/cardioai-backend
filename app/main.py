from fastapi import FastAPI, Query
import matplotlib
matplotlib.use("Agg")

from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PatientData, PredictionResponse, ExplanationResponse
from app.services.predictor import predict
from app.services.explainer import explain
from app.services.shap_plot import generate_waterfall_bytes
from app.services.lime_explainer import generate_lime
from app.services.shap_force import generate_shap_force_html
from app.config import APP_NAME, MODEL_VERSION


# ✅ Create app FIRST
app = FastAPI(
    title=APP_NAME,
    version=MODEL_VERSION,
    description="XGBoost-based Heart Disease Prediction API with SHAP Explainability",
)

# ✅ Then add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
def get_prediction(data: PatientData, model_type: str = Query(default="xgboost")):
    return predict(data.dict(), model_type)


@app.post("/explain", response_model=ExplanationResponse)
def get_explanation(data: PatientData):
    return explain(data.dict())


@app.post("/explain/shap-plot")
def shap_plot(data: PatientData):
    img_bytes = generate_waterfall_bytes(data.dict())
    return Response(content=img_bytes, media_type="image/png")


@app.post("/explain/lime-plot")
def lime_plot(data: PatientData):
    img_bytes = generate_lime(data.dict())
    return Response(content=img_bytes, media_type="image/png")


@app.post("/explain/shap-force", response_class=HTMLResponse)
def shap_force(data: PatientData):
    html = generate_shap_force_html(data.dict())
    return HTMLResponse(content=html)

@app.get("/")
def root():
    return {"message": "CardioAI Backend is running"}


