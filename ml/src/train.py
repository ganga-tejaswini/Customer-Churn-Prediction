"""
train.py
--------
Same 4-model comparison as before, but now each model's hyperparameters
are tuned with GridSearchCV instead of being fixed guesses — this
mirrors the paper's approach (their Table 2) and gives you a real
answer to "how did you choose these hyperparameters?" in your viva.

Run after preprocess.py:
    python train.py

NOTE: grid search takes longer than the old fixed-parameter version
(several minutes instead of under one). That's expected — it's
actually searching, not just running once.
"""

import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier

from preprocess import run_preprocessing_pipeline
from evaluate import evaluate_model
from statistical_test import run_wilcoxon_tests

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
STATS_PATH = os.path.join(MODELS_DIR, "statistical_test.json")

RANDOM_STATE = 42

# Smaller, targeted grids — a full grid like the paper's would take hours
# on a laptop. This covers the parameters that matter most for each model.
PARAM_GRIDS = {
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {"n_estimators": [100, 200], "max_depth": [8, 12, None]},
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=RANDOM_STATE),
        "params": {"n_estimators": [100, 200, 300], "learning_rate": [0.1, 0.5, 1.0]},
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    },
    "XGBoost": {
        "estimator": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    },
}


def train_and_compare():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor = run_preprocessing_pipeline()

    results = {}
    fitted_models = {}
    cv_fold_scores = {}  # needed for the Wilcoxon test

    for name, config in PARAM_GRIDS.items():
        print(f"Tuning {name}...")
        search = GridSearchCV(
            config["estimator"], config["params"], scoring="f1", cv=5, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        fitted_models[name] = best_model

        print(f"  Best params: {search.best_params_}")

        fold_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring="f1")
        cv_fold_scores[name] = fold_scores.tolist()

        test_metrics = evaluate_model(best_model, X_test, y_test)
        test_metrics["best_params"] = search.best_params_
        test_metrics["cv_f1_mean"] = round(float(fold_scores.mean()), 4)
        test_metrics["cv_f1_std"] = round(float(fold_scores.std()), 4)

        results[name] = test_metrics
        print(f"  {name}: F1={test_metrics['f1']}, ROC-AUC={test_metrics['roc_auc']}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']})")

    joblib.dump(best_model, BEST_MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "models": results}, f, indent=2)

    # Wilcoxon signed-rank test: is the best model SIGNIFICANTLY better
    # than each other model, or could the difference be chance?
    stats_results = run_wilcoxon_tests(cv_fold_scores, best_name)
    with open(STATS_PATH, "w") as f:
        json.dump(stats_results, f, indent=2)

    print(f"\nWilcoxon test results (comparing {best_name} to each other model):")
    for model_name, r in stats_results.items():
        print(f"  {model_name}: p-value={r['p_value']} ({'significant' if r['significant'] else 'not significant'})")

    print(f"\nBest model saved to {BEST_MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Statistical test results saved to {STATS_PATH}")

    return best_name, results, stats_results


if __name__ == "__main__":
    train_and_compare()