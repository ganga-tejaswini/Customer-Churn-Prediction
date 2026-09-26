"""
schemas.py
----------
Pydantic models for request validation and response shaping.
CustomerFeatures mirrors the raw Telco dataset columns (minus target/ID)
so the frontend form maps directly onto this.
"""

from pydantic import BaseModel
from typing import Optional


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class TopFactor(BaseModel):
    feature: str
    importance: float
    value: float


class PredictionResponse(BaseModel):
    churn_prediction: bool
    churn_probability: float
    risk_level: str
    model_used: str
    top_factors: list[TopFactor]


class ModelMetricOut(BaseModel):
    model_name: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    roc_auc: Optional[float]
    is_best: bool


class DashboardStats(BaseModel):
    total_predictions: int
    churn_rate: float
    risk_distribution: dict
    predictions_over_time: list[dict]