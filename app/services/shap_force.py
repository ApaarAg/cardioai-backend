import shap
import pandas as pd
from app.services.model_loader import xgb_model

# Get internal XGB model (since calibrated)
calibrator = xgb_model.named_steps["classifier"]
fitted_xgb = calibrator.calibrated_classifiers_[0].estimator
explainer = shap.TreeExplainer(fitted_xgb)

preprocessor = xgb_model.named_steps["preprocessor"]


def generate_shap_force_html(data: dict):

    # 1️⃣ Convert input to DataFrame
    df = pd.DataFrame([data])

    # 2️⃣ Transform using pipeline preprocessor
    transformed = preprocessor.transform(df)

    # 3️⃣ Get correct feature names from ColumnTransformer
    feature_names = preprocessor.get_feature_names_out()

    # 4️⃣ Convert transformed array → DataFrame with names
    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names
    )

    # 5️⃣ OPTIONAL: Clean names (remove num__ / cat__)
    transformed_df.columns = [
        name.replace("num__", "").replace("cat__", "")
        for name in transformed_df.columns
    ]

    # 6️⃣ Compute SHAP values using DataFrame (NOT numpy array)
    shap_values = explainer.shap_values(transformed_df)

    base_value = explainer.expected_value

    # 7️⃣ Create force plot with proper feature names
    force_plot = shap.force_plot(
        base_value,
        shap_values[0],
        transformed_df.iloc[0],   # <-- VERY IMPORTANT
        matplotlib=False
    )

    # 8️⃣ Return standalone HTML
    html = f"""
    <html>
        <head>
            {shap.getjs()}
        </head>
        <body>
            {force_plot.html()}
        </body>
    </html>
    """

    return html
