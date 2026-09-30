"""
model_loader.py (backend/app/ml/)
----------------------------------
Same as before, PLUS: computes engagement_score and service_utilization
for a single incoming customer, matching exactly how preprocess.py
computes them during training. If these don't match, predictions will
be wrong even though nothing crashes — so keep this logic in sync with
ml/src/preprocess.py's add_engineered_features() if you ever change it.
"""

import joblib
import json
import pandas as pd

from app.config import settings

_model = None
_preprocessor = None
_metrics = None

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# NOTE: these min/max values come from the training data. For a real
# deployment, save them alongside preprocessor.pkl instead of hardcoding —
# left as plain constants here so the fix is easy to see and swap in.
TENURE_MIN, TENURE_MAX = 0, 72
TOTAL_CHARGES_MIN, TOTAL_CHARGES_MAX = 0, 8684.8


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


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    tenure_n = (df["tenure"] - TENURE_MIN) / (TENURE_MAX - TENURE_MIN + 1e-9)
    charges_n = (df["TotalCharges"] - TOTAL_CHARGES_MIN) / (TOTAL_CHARGES_MAX - TOTAL_CHARGES_MIN + 1e-9)
    df["engagement_score"] = (tenure_n + charges_n) / 2

    def count_services(row):
        return sum(
            1 for col in SERVICE_COLS
            if str(row[col]).strip().lower() not in ("no", "no phone service", "no internet service")
        )

    df["service_utilization"] = df[SERVICE_COLS].apply(count_services, axis=1)
    return df


def encode_customer(customer_dict: dict) -> pd.DataFrame:
    preprocessor = get_preprocessor()
    df = pd.DataFrame([customer_dict])
    df = _add_engineered_features(df)

    for col in preprocessor["categorical_cols"]:
        le = preprocessor["encoders"][col]
        df[col] = df[col].astype(str).apply(lambda v: v if v in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    df[preprocessor["numeric_cols"]] = preprocessor["scaler"].transform(df[preprocessor["numeric_cols"]])
    return df[preprocessor["feature_order"]]