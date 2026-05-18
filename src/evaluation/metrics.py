"""Medical standard evaluation metrics for clinical prediction models.

Calculates:
- AUROC (Area Under Receiver Operating Characteristic Curve)
- AUPRC (Area Under Precision-Recall Curve, critical for imbalanced EHR cohorts)
- Sensitivity at fixed Specificity (e.g., 90% and 95% specificity thresholds)
- Brier Score (measure of probabilistic calibration)
- Standard classification metrics: F1, Precision, Specificity, Balanced Accuracy
"""

from typing import Any, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def sensitivity_at_fixed_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, target_specificity: float = 0.90
) -> Tuple[float, float]:
    """Finds the operating threshold that achieves >= target_specificity and returns sensitivity.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for class 1.
        target_specificity: Desired minimum specificity (e.g., 0.90 for 90%).

    Returns:
        Tuple of (achieved_sensitivity, operating_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr

    # Filter points where specificity >= target_specificity
    valid_indices = np.where(specificity >= target_specificity)[0]
    if len(valid_indices) == 0:
        # If unable to reach target specificity, return lowest FPR point
        idx = np.argmin(fpr)
    else:
        # Choose the threshold that maximizes TPR among those satisfying specificity
        idx = valid_indices[np.argmax(tpr[valid_indices])]

    achieved_sens = float(tpr[idx])
    threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return achieved_sens, threshold


def evaluate_clinical_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    default_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes a comprehensive suite of clinical performance metrics.

    Args:
        y_true: Ground truth binary array.
        y_prob: Predicted continuous probability array.
        default_threshold: Decision threshold for discrete classification.

    Returns:
        Dictionary of medical metrics and operational thresholds.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= default_threshold).astype(int)

    # Core discrimination metrics
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # Fixed specificity operating points (Clinical rule-out / rule-in)
    sens_at_90_spec, thresh_90 = sensitivity_at_fixed_specificity(y_true, y_prob, 0.90)
    sens_at_95_spec, thresh_95 = sensitivity_at_fixed_specificity(y_true, y_prob, 0.95)

    # Standard metrics at default threshold
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    # Optimal F1 threshold search
    precisions, recalls, f_thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_f1_idx])
    best_f1_thresh = float(f_thresholds[best_f1_idx]) if best_f1_idx < len(f_thresholds) else 0.5

    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "sensitivity_at_90_spec": sens_at_90_spec,
        "threshold_at_90_spec": thresh_90,
        "sensitivity_at_95_spec": sens_at_95_spec,
        "threshold_at_95_spec": thresh_95,
        "default_threshold": default_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_thresh,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
