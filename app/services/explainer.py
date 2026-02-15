import shap
import pandas as pd
from typing import Dict
from app.services.model_loader import fitted_xgb, preprocessor

explainer = shap.TreeExplainer(fitted_xgb)


def explain(data: Dict):

    df = pd.DataFrame([data])

    transformed = preprocessor.transform(df)

    shap_values = explainer.shap_values(transformed)

    feature_names = preprocessor.get_feature_names_out()

    shap_dict = dict(
        zip(feature_names, shap_values[0])
    )

    return {
        "shap_values": shap_dict,
        "base_value": float(explainer.expected_value),
    }
