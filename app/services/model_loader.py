import pandas as pd
import joblib
import os
from app.config import MODEL_PATH

xgb_model = joblib.load(MODEL_PATH)

preprocessor = xgb_model.named_steps["preprocessor"]
classifier = xgb_model.named_steps["classifier"]

if hasattr(classifier, "calibrated_classifiers_"):
    fitted_xgb = classifier.calibrated_classifiers_[0].estimator
else:
    fitted_xgb = classifier

# IMPORTANT: Add feature names
original_feature_names = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]
