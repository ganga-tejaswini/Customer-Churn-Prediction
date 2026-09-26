"""
explainer.py
------------
Backend-side copy of the explanation logic in ml/src/explain.py —
kept separate so the backend never needs to import from the ml/ folder
directly (they stay decoupled, per the architecture).
"""

import numpy as np


def get_top_factors(model, encoded_row, feature_names: list, top_n: int = 4) -> list:
    if not hasattr(model, "feature_importances_"):
        return []

    importances = model.feature_importances_
    row = np.asarray(encoded_row).flatten()

    signal = importances * np.abs(row)
    top_idx = np.argsort(signal)[::-1][:top_n]

    return [
        {
            "feature": feature_names[i],
            "importance": round(float(importances[i]), 4),
            "value": round(float(row[i]), 4),
        }
        for i in top_idx
    ]


def get_risk_level(probability: float) -> str:
    if probability >= 0.66:
        return "High"
    elif probability >= 0.33:
        return "Medium"
    return "Low"