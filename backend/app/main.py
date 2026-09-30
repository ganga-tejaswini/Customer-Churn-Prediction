"""
main.py
-------
App entrypoint. Mounts all routers, sets up CORS so the React frontend
can call this API, and loads the ML model once at startup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.ml.model_loader import load_artifacts
from app.routers import predict, batch, dashboard, models_compare

app = FastAPI(title="Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    # Loads best_model.pkl, preprocessor.pkl, metrics.json into memory once,
    # instead of reading them from disk on every request.
    load_artifacts()


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(predict.router, tags=["predict"])
app.include_router(batch.router, tags=["batch"])
app.include_router(dashboard.router, tags=["dashboard"])
app.include_router(models_compare.router, tags=["models"])