"""
routers/batch.py
-----------------
POST /predict/batch — upload a CSV of many customers, get back a
ranked list (highest churn risk first). Closer to how a retention
team would actually use this day to day than one-at-a-time entry.
"""

import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Prediction
from app.ml.model_loader import get_model, get_preprocessor
from app.ml.explainer import get_risk_level

router = APIRouter()


@router.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    model = get_model()
    preprocessor = get_preprocessor()

    df_encoded = df.copy()
    for col in preprocessor["categorical_cols"]:
        le = preprocessor["encoders"][col]
        df_encoded[col] = df_encoded[col].astype(str).apply(
            lambda v: v if v in le.classes_ else le.classes_[0]
        )
        df_encoded[col] = le.transform(df_encoded[col])

    df_encoded[preprocessor["numeric_cols"]] = preprocessor["scaler"].transform(
        df_encoded[preprocessor["numeric_cols"]]
    )

    X = df_encoded[preprocessor["feature_order"]]
    probabilities = model.predict_proba(X)[:, 1]

    results = []
    records_to_log = []
    for i, prob in enumerate(probabilities):
        risk = get_risk_level(float(prob))
        row_result = {
            "customerID": df.iloc[i].get("customerID", f"row_{i}"),
            "churn_probability": round(float(prob), 4),
            "risk_level": risk,
        }
        results.append(row_result)

        records_to_log.append(Prediction(
            input_features=df.iloc[i].to_dict(),
            churn_prediction=bool(prob >= 0.5),
            churn_probability=float(prob),
            risk_level=risk,
            model_used=type(model).__name__,
            top_factors=None,
            source="batch",
        ))

    db.add_all(records_to_log)
    await db.commit()

    # Highest risk first — this is the order a retention team actually wants
    results.sort(key=lambda r: r["churn_probability"], reverse=True)

    return {"count": len(results), "results": results}