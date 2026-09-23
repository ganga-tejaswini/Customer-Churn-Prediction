"""
evaluate.py
-----------
Computes evaluation metrics for a fitted model. Kept separate from
train.py so the backend or a notebook can re-score a model without
re-running the whole training pipeline.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    # Some models (e.g. AdaBoost) expose predict_proba too — used for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
    }