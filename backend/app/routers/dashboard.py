"""
routers/dashboard.py
---------------------
GET /dashboard — aggregate stats over everything logged in Neon so far.
This is what makes the database actually earn its place instead of
sitting unused.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from collections import Counter

from app.db import get_db
from app.models import Prediction
from app.schemas import DashboardStats

router = APIRouter()


@router.get("/dashboard", response_model=DashboardStats)
async def dashboard(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Prediction))
    predictions = result.scalars().all()

    total = len(predictions)
    churned = sum(1 for p in predictions if p.churn_prediction)
    churn_rate = round(churned / total, 4) if total else 0.0

    risk_counts = Counter(p.risk_level for p in predictions)

    # Group by day for a simple trend line
    by_day = Counter(p.created_at.date().isoformat() for p in predictions if p.created_at)
    trend = [{"date": day, "count": count} for day, count in sorted(by_day.items())]

    return DashboardStats(
        total_predictions=total,
        churn_rate=churn_rate,
        risk_distribution=dict(risk_counts),
        predictions_over_time=trend,
    )