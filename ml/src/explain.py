"""
explain.py
----------
UPGRADED to use real SHAP (SHapley Additive exPlanations), matching
the paper's core contribution — global explanations (which features
matter most overall) and local explanations (why THIS customer's
prediction came out the way it did).

You'll need to add `shap>=0.44` to ml/requirements.txt.
"""

import shap
import numpy as np
import pandas as pd


def get_shap_explainer(model):
    """Tree-based models (all 4 of yours) use the fast TreeExplainer."""
    return shap.TreeExplainer(model)


def get_global_explanation(model, X_sample: pd.DataFrame, save_path: str = None):
    """
    Global explanation: which features matter most ACROSS ALL customers.
    X_sample should be a few hundred rows from your training set — you
    don't need the whole dataset, SHAP on the full set is slow.

    If save_path is given (e.g. "ml/models/shap_beeswarm.png"), saves
    the beeswarm plot as an image — useful to drop straight into your
    report or PPT.
    """
    explainer = get_shap_explainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Some models return a list (one array per class) — take the churn class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        zip(X_sample.columns, mean_abs_shap), key=lambda x: x[1], reverse=True
    )

    if save_path:
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    return [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in ranked]


def get_local_explanation(model, X_row: pd.DataFrame, top_n: int = 4) -> list:
    """
    Local explanation: why did THIS specific customer get this prediction?
    Returns the top features pushing this one prediction toward or away
    from churn — this is what the backend calls per prediction.
    """
    explainer = get_shap_explainer(model)
    shap_values = explainer.shap_values(X_row)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    row_values = shap_values[0] if shap_values.ndim > 1 else shap_values
    feature_names = X_row.columns.tolist()

    ranked_idx = np.argsort(np.abs(row_values))[::-1][:top_n]

    return [
        {
            "feature": feature_names[i],
            "shap_value": round(float(row_values[i]), 4),
            "direction": "increases churn risk" if row_values[i] > 0 else "decreases churn risk",
        }
        for i in ranked_idx
    ]