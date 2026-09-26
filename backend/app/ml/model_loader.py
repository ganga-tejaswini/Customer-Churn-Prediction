"""
model_loader.py
----------------
Loads best_model.pkl and preprocessor.pkl ONCE when the backend starts,
rather than on every request (loading a pickle per request would be slow
and is unnecessary — the model doesn't change between requests).
"""

import joblib
import json
import pandas as pd

from app.config import settings

_model = None
_preprocessor = None
_metrics = None


def load_artifacts():
    global _model, _preprocessor, _metrics

    _model = joblib.load(settings.MODEL_PATH)
    _preprocessor = joblib.load(settings.PREPROCESSOR_PATH)

    with open(settings.METRICS_PATH) as f:
        _metrics = json.load(f)

    return _model, _preprocessor, _metrics


def get_model():
    if _model is None:
        load_artifacts()
    return _model


def get_preprocessor():
    if _preprocessor is None:
        load_artifacts()
    return _preprocessor


def get_metrics():
    if _metrics is None:
        load_artifacts()
    return _metrics


def encode_customer(customer_dict: dict) -> pd.DataFrame:
    """
    Takes a raw customer dict (matching CustomerFeatures schema) and
    returns a single-row DataFrame encoded exactly like training data,
    using the SAME fitted preprocessor from training — this consistency
    is why preprocessor.pkl travels with best_model.pkl.
    """
    preprocessor = get_preprocessor()
    df = pd.DataFrame([customer_dict])

    for col in preprocessor["categorical_cols"]:
        le = preprocessor["encoders"][col]
        df[col] = df[col].astype(str).apply(
            lambda v: v if v in le.classes_ else le.classes_[0]
        )
        df[col] = le.transform(df[col])

    df[preprocessor["numeric_cols"]] = preprocessor["scaler"].transform(
        df[preprocessor["numeric_cols"]]
    )

    return df[preprocessor["feature_order"]]