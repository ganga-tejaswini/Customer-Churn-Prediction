"""
explain.py
----------
Gives the top features driving a single prediction — this is what
powers the "why is this customer at risk" explanation in the frontend,
one of the things that makes the app feel like a real product instead
of a bare predict() wrapper.

Uses the model's built-in feature_importances_ combined with how far
each feature value sits from the training mean, as a lightweight
per-customer explanation. (SHAP gives more precise per-row explanations
but is heavier to install/run — swap this out for SHAP later if you
want more rigor and have time before submission.)
"""

import numpy as np


def get_top_features(model, X_row, feature_names, top_n: int = 4) -> list:
    """
    X_row: a single already-preprocessed row (1D array-like), matching
    the column order the model was trained on.
    """
    if not hasattr(model, "feature_importances_"):
        return []

    importances = model.feature_importances_
    row = np.asarray(X_row).flatten()

    # Weight global importance by how "extreme" this customer's value is
    # (a scaled numeric feature far from 0 stands out more)
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