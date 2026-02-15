import shap
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import base64
from io import BytesIO
from app.services.model_loader import fitted_xgb, preprocessor

explainer = shap.TreeExplainer(fitted_xgb)

def generate_waterfall_bytes(data: dict):
    df = pd.DataFrame([data])
    transformed = preprocessor.transform(df)

    shap_values = explainer(transformed)

    shap.plots.waterfall(shap_values[0], show=False)

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close()

    return buffer.getvalue()
