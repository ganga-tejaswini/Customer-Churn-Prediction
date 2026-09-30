"""
routers/predict.py
-------------------
POST /predict — takes one customer's features, returns a churn
prediction, probability, risk level, and the top factors behind it.
Also logs the result to Neon so it shows up in the dashboard.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Prediction
from app.schemas import CustomerFeatures, PredictionResponse
from app.ml.model_loader import get_model, get_preprocessor, encode_customer
from app.ml.explainer import get_top_factors, get_risk_level

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures, db: AsyncSession = Depends(get_db)):
    model = get_model()
    preprocessor = get_preprocessor()

    encoded_df = encode_customer(customer.model_dump())
    probability = float(model.predict_proba(encoded_df)[0][1])
    prediction = probability >= 0.5
    risk_level = get_risk_level(probability)

    top_factors = get_top_factors(
        model, encoded_df.values, preprocessor["feature_order"]
    )

    # Log to Neon for the history dashboard
    record = Prediction(
        input_features=customer.model_dump(),
        churn_prediction=prediction,
        churn_probability=probability,
        risk_level=risk_level,
        model_used=type(model).__name__,
        top_factors=top_factors,
        source="single",
    )
    db.add(record)
    await db.commit()

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_level=risk_level,
        model_used=type(model).__name__,
        top_factors=top_factors,
    )