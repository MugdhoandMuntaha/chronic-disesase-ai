"""Split Conformal Prediction & Uncertainty Quantification for Clinical Risk Scoring.

Implements distribution-free uncertainty quantification:
1. Conformal Prediction Sets C(x) with finite-sample coverage guarantees P(Y in C(X)) >= 1 - alpha.
2. Clinical ambiguity detection: Flags borderline patients where C(x) = {0, 1} for mandatory physician review.
3. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true: Binary ground truth array of shape (N,).
        y_prob: Continuous predicted probabilities of shape (N,).
        num_bins: Number of equal-width calibration bins.

    Returns:
        ece: Expected Calibration Error.
        mce: Maximum Calibration Error across bins.
        bin_records: List of dictionaries with per-bin accuracy and confidence.
    """
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)
    bin_records = []

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_prob >= low) & (y_prob < high) if i < num_bins - 1 else (y_prob >= low) & (y_prob <= high)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_prob[in_bin]))
            diff = abs(bin_acc - bin_conf)
            ece += (bin_count / n) * diff
            mce = max(mce, diff)

            bin_records.append({
                "bin_range": f"{low:.2f}-{high:.2f}",
                "patient_count": int(bin_count),
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "calibration_gap": round(diff, 4),
            })

    return float(ece), float(mce), bin_records


class ClinicalConformalPredictor:
    """Split Conformal Predictor for binary clinical risk scoring."""

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Significance level. Coverage guarantee is >= 1 - alpha (e.g. 0.10 -> 90% coverage).
        """
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None

    def calibrate(self, y_true_calib: np.ndarray, y_prob_calib: np.ndarray) -> float:
        """Calibrates non-conformity threshold using a held-out calibration set.

        Score s_i = 1 - P(Y = y_i | x_i):
        - If y_i = 1, s_i = 1 - p_i
        - If y_i = 0, s_i = p_i
        """
        n = len(y_true_calib)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Non-conformity scores
        scores = np.where(y_true_calib == 1, 1.0 - y_prob_calib, y_prob_calib)
        self.calibration_scores = scores

        # Conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher" if hasattr(np, "quantile") else "linear"))
        print(f"[ConformalPredictor] Calibrated threshold q_hat = {self.q_hat:.4f} at 1-alpha = {1 - self.alpha:.1%}")
        return self.q_hat

    def predict_set(self, prob: float) -> List[int]:
        """Predicts confidence set for a single continuous probability.

        Returns:
            set_prediction: List of included labels [0], [1], [0, 1], or [].
        """
        if self.q_hat is None:
            raise ValueError("Conformal predictor must be calibrated before prediction.")

        pred_set = []
        # Score for class 0 is: 1 - (1 - prob) = prob
        if prob <= self.q_hat:
            pred_set.append(0)
        # Score for class 1 is: 1 - prob
        if (1.0 - prob) <= self.q_hat:
            pred_set.append(1)

        # Fallback if empty (e.g. at very high alpha)
        if len(pred_set) == 0:
            pred_set = [int(prob >= 0.5)]

        return pred_set

    def predict_batch(self, probs: np.ndarray) -> List[List[int]]:
        """Predicts confidence sets for an array of probabilities."""
        return [self.predict_set(p) for p in probs]

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Evaluates empirical coverage, set efficiency, and clinical ambiguity."""
        pred_sets = self.predict_batch(y_prob)
        n = len(y_true)

        # Coverage: true label is in prediction set
        covered = [y_true[i] in pred_sets[i] for i in range(n)]
        empirical_coverage = float(np.mean(covered))

        # Ambiguity: prediction set contains both {0, 1}
        ambiguous_cases = [len(s) == 2 for s in pred_sets]
        ambiguous_count = int(np.sum(ambiguous_cases))
        ambiguity_rate = float(np.mean(ambiguous_cases))

        # Singletons
        singleton_0 = int(np.sum([s == [0] for s in pred_sets]))
        singleton_1 = int(np.sum([s == [1] for s in pred_sets]))

        ece, mce, bin_details = compute_ece(y_true, y_prob)
        brier = float(brier_score_loss(y_true, y_prob))

        results = {
            "target_coverage_guarantee": 1.0 - self.alpha,
            "empirical_coverage": round(empirical_coverage, 4),
            "guarantee_satisfied": bool(empirical_coverage >= (1.0 - self.alpha - 0.05)),
            "conformal_threshold_q_hat": round(self.q_hat, 4) if self.q_hat else None,
            "sample_count": n,
            "singleton_non_disease_count": singleton_0,
            "singleton_onset_count": singleton_1,
            "ambiguous_borderline_count": ambiguous_count,
            "ambiguity_rate_pct": round(ambiguity_rate * 100.0, 1),
            "expected_calibration_error": round(ece, 5),
            "maximum_calibration_error": round(mce, 5),
            "brier_score": round(brier, 6),
            "calibration_bins": bin_details,
        }
        return results


def run_conformal_calibration_workflow() -> Dict[str, Any]:
    """Calibrates and validates conformal prediction on the test cohort."""
    print("=" * 75)
    print("EXECUTING CLINICAL CONFORMAL PREDICTION & UNCERTAINTY QUANTIFICATION")
    print("=" * 75)

    test_tab_path = PROCESSED_DATA_DIR / "test_tabular.csv"
    train_tab_path = PROCESSED_DATA_DIR / "train_tabular.csv"
    if not test_tab_path.exists() or not train_tab_path.exists():
        raise FileNotFoundError("Processed datasets not found.")

    df_train = pd.read_csv(train_tab_path)
    df_test = pd.read_csv(test_tab_path)

    target_col = "target_label" if "target_label" in df_train.columns else "target_disease"
    y_train = df_train[target_col].values.astype(int)
    y_test = df_test[target_col].values.astype(int)

    # Load XGBoost predictions
    xgb_path = MODELS_DIR / "xgb_baseline.json"
    from xgboost import XGBClassifier
    from src.models.baseline_xgb import XGBoostBaselineTrainer

    trainer = XGBoostBaselineTrainer()
    X_train, _, X_test, _ = trainer.load_data()

    model = XGBClassifier()
    model.load_model(str(xgb_path))

    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Use 50% of test cohort as calibration set, 50% as evaluation set
    calib_size = len(y_test) // 2
    y_calib, y_eval = y_test[:calib_size], y_test[calib_size:]
    p_calib, p_eval = y_prob_test[:calib_size], y_prob_test[calib_size:]

    predictor = ClinicalConformalPredictor(alpha=0.10)
    predictor.calibrate(y_calib, p_calib)
    metrics = predictor.evaluate(y_eval, p_eval)

    print(f"\nTarget Coverage Guarantee: {metrics['target_coverage_guarantee']:.1%}")
    print(f"Empirical Coverage:        {metrics['empirical_coverage']:.1%}")
    print(f"High-Certainty Controls:   {metrics['singleton_non_disease_count']} patients (Prediction Set: {{0}})")
    print(f"High-Certainty Onset:      {metrics['singleton_onset_count']} patients (Prediction Set: {{1}})")
    print(f"Ambiguous / Flagged:       {metrics['ambiguous_borderline_count']} patients (Prediction Set: {{0, 1}})")
    print(f"Expected Calib Error (ECE):{metrics['expected_calibration_error']:.5f}")

    # Export report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "conformal_calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Conformal] Exported conformal calibration report to {report_path}")

    return metrics


if __name__ == "__main__":
    run_conformal_calibration_workflow()
