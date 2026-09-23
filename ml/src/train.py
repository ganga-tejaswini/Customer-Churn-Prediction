"""
train.py
--------
Trains all 4 ensemble classifiers named in the project abstract
(Random Forest, AdaBoost, Gradient Boosting, XGBoost), evaluates each
with cross-validation, picks the best by F1 score on the churn class,
and saves the winner + metrics for every model.

Run after preprocess.py:
    python train.py
"""

import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from preprocess import run_preprocessing_pipeline
from evaluate import evaluate_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

RANDOM_STATE = 42


def get_model_candidates():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            eval_metric="logloss", random_state=RANDOM_STATE
        ),
    }


def train_and_compare():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor = run_preprocessing_pipeline()

    candidates = get_model_candidates()
    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
        test_metrics = evaluate_model(model, X_test, y_test)
        test_metrics["cv_f1_mean"] = round(float(cv_f1), 4)

        results[name] = test_metrics
        print(f"  {name}: {test_metrics}")

    # Pick the winner by F1 on the churn class (accuracy is misleading
    # on an imbalanced dataset like this one)
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted_models[best_name]

    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']})")

    joblib.dump(best_model, BEST_MODEL_PATH)

    output = {
        "best_model": best_name,
        "models": results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Best model saved to {BEST_MODEL_PATH}")
    print(f"All metrics saved to {METRICS_PATH}")

    return best_name, results


if __name__ == "__main__":
    train_and_compare()