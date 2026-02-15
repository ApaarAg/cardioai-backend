import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "xgb_pipeline.pkl")

MODEL_VERSION = "1.0.0"
APP_NAME = "Clinical AI Prediction API"
