"""
preprocess.py
--------------
Cleans, encodes, and splits the Telco Customer Churn dataset.
Used by train.py and, later, by the backend at prediction time
(the SAME preprocessor object must be reused, not rebuilt, so that
a live prediction request is encoded exactly like training data was).

Expected raw file: ml/data/raw/telco_churn.csv
(Kaggle "Telco Customer Churn" by IBM — WA_Fn-UseC_-Telco-Customer-Churn.csv)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_churn.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessor.pkl")

TARGET_COL = "Churn"
ID_COL = "customerID"

# Columns that are categorical in the raw Telco dataset
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# SeniorCitizen is already 0/1 in the raw file — treat as numeric, no encoding needed
BINARY_NUMERIC_COLS = ["SeniorCitizen"]


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is stored as string in the raw file and has blank values
    # for customers with 0 tenure — coerce to numeric and fill with 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop the customer ID — it's an identifier, not a feature
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # Normalize target to binary 0/1
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    return df


def build_preprocessor(df: pd.DataFrame):
    """
    Fits a LabelEncoder per categorical column and a StandardScaler on
    numeric columns. Returns a dict bundling everything needed to
    transform new, unseen data the same way (used again at prediction time).
    """
    encoders = {}
    df_encoded = df.copy()

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    df_encoded[NUMERIC_COLS] = scaler.fit_transform(df_encoded[NUMERIC_COLS])

    preprocessor = {
        "encoders": encoders,
        "scaler": scaler,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "binary_numeric_cols": BINARY_NUMERIC_COLS,
        "feature_order": CATEGORICAL_COLS + NUMERIC_COLS + BINARY_NUMERIC_COLS,
    }

    return preprocessor, df_encoded


def transform_with_preprocessor(df: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    """
    Applies an ALREADY-FITTED preprocessor to new data (e.g. a single
    customer submitted through the API). Unseen categorical values fall
    back to the encoder's first known class rather than crashing.
    """
    df = df.copy()

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


def split_and_balance(df_encoded: pd.DataFrame):
    """
    Stratified train/test split, then SMOTE applied to the TRAINING
    split only (never touch the test set — that would leak information
    and make evaluation metrics meaningless).
    """
    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    return X_train_bal, X_test, y_train_bal, y_test


def run_preprocessing_pipeline():
    """Full pipeline: load -> clean -> encode/scale -> split -> save preprocessor."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)

    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    preprocessor, df_encoded = build_preprocessor(df_clean)

    X_train, X_test, y_train, y_test = split_and_balance(df_encoded)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    print(f"Preprocessing complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    run_preprocessing_pipeline()