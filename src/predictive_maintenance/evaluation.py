import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true,
    probabilities,
    *,
    threshold: float = 0.5,
    cost_missed_failure: float = 10_000,
    cost_intervention: float = 1_000,
) -> dict:
    """Evaluate binary scores on a benchmark containing both classes."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("Inputs must be non-empty and have equal lengths.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only 0 or 1.")

    if np.unique(y_true).size != 2:
        raise ValueError("Both classes are required for this benchmark.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    costs = np.array([cost_missed_failure, cost_intervention])
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Costs must be finite and non-negative.")

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    business_cost = (
        cost_missed_failure * fn
        + cost_intervention * (tp + fp)
    )

    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "interventions": int(tp + fp),
        "business_cost_eur": float(business_cost),
    }