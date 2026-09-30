"""
preprocess.py
--------------
Cleans, encodes, and splits the Telco Customer Churn dataset.
Used by train.py and, later, by the backend at prediction time
(the SAME preprocessor object must be reused, not rebuilt, so that
a live prediction request is encoded exactly like training data was).

UPDATED: this project's actual dataset is the *extended* IBM Telco
Churn file (columns like "Senior Citizen", "Tenure Months", plus geo
columns and Churn Score/CLTV/Churn Reason). This version:
  1. Reads the .xlsx file directly (no CSV conversion needed).
  2. Renames columns to the internal names the rest of the app uses
     (gender, SeniorCitizen, tenure, PhoneService, ... Churn).
  3. DROPS "Churn Score", "CLTV", and "Churn Reason" — these leak the
     answer (Churn Reason literally states why the customer left;
     Churn Score is someone else's pre-computed prediction). Training
     on them would make the model meaningless.
  4. Drops geo/ID columns (CustomerID, Count, Country, State, City,
     Zip Code, Lat Long, Latitude, Longitude, Churn Value) since the
     rest of the app (backend schema, frontend form) is built around
     the standard 19-feature set, not location data.

Expected raw file: dataset/Telco_churn.xlsx.xlsx (top level of the repo,
sibling to ml/, backend/, frontend/)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset", "Telco_churn.xlsx.xlsx"
)
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessor.pkl")

TARGET_COL = "Churn"

# Maps the actual column names in this dataset -> the internal names
# used everywhere else in the app (backend schemas, frontend form).
COLUMN_RENAME_MAP = {
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Contract": "Contract",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Label": "Churn",
}

# Columns that must never reach the model — leakage or irrelevant identifiers/geo
DROP_COLS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Value", "Churn Score", "CLTV", "Churn Reason",
]

# Columns that are categorical after renaming (Senior Citizen is Yes/No
# in THIS dataset, unlike the classic version where it's already 0/1 —
# so it's treated as categorical here, not numeric)
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # TotalCharges can have blanks for customers with 0 tenure
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

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
        "feature_order": CATEGORICAL_COLS + NUMERIC_COLS,
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