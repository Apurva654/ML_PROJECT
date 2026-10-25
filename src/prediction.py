import os
import joblib
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def predict_single_instance(pipeline, model, input_dict: dict, original_columns: list):
    input_df = pd.DataFrame([input_dict])
    for col in original_columns:
        if col not in input_df.columns:
            input_df[col] = None
    transformed_input = pipeline.transform(input_df[original_columns])
    return float(model.predict(transformed_input)[0])