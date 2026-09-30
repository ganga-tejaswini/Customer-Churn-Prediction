"""
routers/models_compare.py
--------------------------
GET /models — serves the 4 models' metrics from ml/models/metrics.json
(written by train.py), for the frontend's model comparison page.
"""

from fastapi import APIRouter

from app.ml.model_loader import get_metrics

router = APIRouter()


@router.get("/models")
async def get_model_comparison():
    metrics = get_metrics()
    return metrics