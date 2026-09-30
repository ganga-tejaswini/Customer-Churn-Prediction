"""
statistical_test.py
--------------------
NEW FILE. Save this into ml/src/ alongside your other files.

Implements the same validation the paper used: the Wilcoxon signed-rank
test, applied to each model's per-fold F1 scores from cross-validation.
This answers "is my best model ACTUALLY better, or did it just get
lucky on this split?" — a p-value below 0.05 means the difference is
statistically significant, not chance.
"""

from scipy.stats import wilcoxon


def run_wilcoxon_tests(cv_fold_scores: dict, best_model_name: str) -> dict:
    """
    cv_fold_scores: {model_name: [fold1_f1, fold2_f1, ...]} — same number
    of folds for every model, produced by cross_val_score with the same cv.
    """
    best_scores = cv_fold_scores[best_model_name]
    results = {}

    for name, scores in cv_fold_scores.items():
        if name == best_model_name:
            continue

        try:
            stat, p_value = wilcoxon(best_scores, scores)
        except ValueError:
            # Happens if the two score lists are identical on every fold
            stat, p_value = 0.0, 1.0

        results[name] = {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 5),
            "significant": bool(p_value < 0.05),
        }

    return results