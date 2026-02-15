import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from io import BytesIO
from lime.lime_tabular import LimeTabularExplainer
import joblib
from app.services.model_loader import xgb_model

# Load training distribution saved earlier
TRAINING_DATA = joblib.load(r"app\services\training_distribution.pkl")

FEATURE_NAMES = TRAINING_DATA.columns.tolist()

explainer = LimeTabularExplainer(
    training_data=TRAINING_DATA.values,
    feature_names=FEATURE_NAMES,
    class_names=["No Disease", "Disease"],
    mode="classification",
    discretize_continuous=False,
    random_state=42
)

def predict_fn(input_array):
    df = pd.DataFrame(input_array, columns=FEATURE_NAMES)
    return xgb_model.predict_proba(df)

def generate_lime(data: dict):

    input_df = pd.DataFrame([data])

    explanation = explainer.explain_instance(
        input_df.values[0],
        predict_fn,
        num_features=8
    )

    fig = explanation.as_pyplot_figure()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)

    return buffer.getvalue()
