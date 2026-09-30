"""
config.py
---------
Loads environment variables once. Every other module imports `settings`
from here instead of calling os.getenv directly, so there's one place
to check when something's misconfigured.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/churn_db"
    )
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173"
    ).split(",")

    MODEL_PATH: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "ml", "models", "best_model.pkl"
    )
    PREPROCESSOR_PATH: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "ml", "models", "preprocessor.pkl"
    )
    METRICS_PATH: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "ml", "models", "metrics.json"
    )


settings = Settings()