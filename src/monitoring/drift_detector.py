"""Clinical Population Distribution Shift and Covariate Drift Detection.

Implements medical ML safety monitoring using:
1. Population Stability Index (PSI):
   PSI = sum((P_b - Q_b) * ln(P_b / Q_b))
   - PSI < 0.10: Stable distribution (No action needed)
   - 0.10 <= PSI < 0.25: Moderate covariate drift (Monitor closely)
   - PSI >= 0.25: Significant clinical distribution shift (Trigger retraining alert)
2. Two-Sample Kolmogorov-Smirnov (KS) Test (scipy.stats.ks_2samp):
   Detects distribution shape changes in laboratory biomarkers (FPG, HbA1c, BP, eGFR).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculates Population Stability Index (PSI) between reference and current populations.

    Args:
        expected: Reference/baseline array (e.g. training set values).
        actual: Incoming monitored array (e.g. inference batch).
        num_bins: Number of quantile bins to discretize values.
        epsilon: Small smoothing constant to avoid log(0) or division by zero.

    Returns:
        psi_value: Total Population Stability Index.
        bin_details: Breakdown of expected vs actual proportions per bin.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0, []

    # Compute quantile bin edges on expected reference data
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    # Ensure unique strictly increasing edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0, []

    # Adjust lowest and highest edge to contain all observations
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Bin counts
    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    # Proportions
    exp_pct = (exp_counts / len(expected)) + epsilon
    act_pct = (act_counts / len(actual)) + epsilon

    # Re-normalize with epsilon
    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)

    # PSI calculation
    psi_bins = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    total_psi = float(np.sum(psi_bins))

    bin_details = []
    for b_idx in range(len(exp_counts)):
        bin_details.append({
            "bin_index": b_idx,
            "expected_proportion": round(float(exp_pct[b_idx]), 4),
            "actual_proportion": round(float(act_pct[b_idx]), 4),
            "psi_contribution": round(float(psi_bins[b_idx]), 6),
        })

    return total_psi, bin_details


class ClinicalDriftDetector:
    """Monitors clinical distribution drift across vitals, labs, and demographics."""

    def __init__(
        self,
        monitored_features: Optional[List[str]] = None,
        reference_path: Union[str, Path] = PROCESSED_DATA_DIR / "train_tabular.csv",
    ):
        self.reference_path = Path(reference_path)
        self.monitored_features = monitored_features or [
            "age",
            "baseline_bmi",
            "fasting_glucose_latest",
            "fasting_glucose_mean",
            "hba1c_latest",
            "hba1c_mean",
            "systolic_bp_latest",
            "systolic_bp_mean",
            "diastolic_bp_latest",
            "diastolic_bp_mean",
            "egfr_latest",
            "triglycerides_latest",
        ]
        self.reference_df: Optional[pd.DataFrame] = None
        self._load_reference_data()

    def _load_reference_data(self) -> None:
        """Loads baseline reference training data."""
        if self.reference_path.exists():
            self.reference_df = pd.read_csv(self.reference_path)
        else:
            print(f"[DriftDetector] Warning: Reference file {self.reference_path} not found.")

    def evaluate_cohort(
        self,
        current_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Evaluates drift across all monitored clinical variables for a new patient cohort."""
        if self.reference_df is None:
            raise ValueError("Reference training cohort is not loaded.")

        feature_reports: List[Dict[str, Any]] = []
        max_psi = 0.0
        drifted_features = []

        for feat in self.monitored_features:
            if feat not in self.reference_df.columns or feat not in current_df.columns:
                continue

            ref_vals = self.reference_df[feat].dropna().values.astype(float)
            curr_vals = current_df[feat].dropna().values.astype(float)

            if len(curr_vals) < 5:
                continue

            # 1. PSI Calculation
            psi_val, bin_details = calculate_psi(ref_vals, curr_vals, num_bins=10)

            # 2. Two-Sample Kolmogorov-Smirnov Test
            ks_res = stats.ks_2samp(ref_vals, curr_vals)
            ks_stat = float(ks_res.statistic)
            ks_pvalue = float(ks_res.pvalue)

            # Triage drift status
            if psi_val < 0.10:
                tier = "Stable"
                color = "#10b981"  # Green
            elif psi_val < 0.25:
                tier = "Moderate Drift"
                color = "#f59e0b"  # Amber
                drifted_features.append(feat)
            else:
                tier = "Significant Drift"
                color = "#ef4444"  # Red
                drifted_features.append(feat)

            if psi_val > max_psi:
                max_psi = psi_val

            feature_reports.append({
                "feature": feat,
                "psi": round(psi_val, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pvalue, 6),
                "is_statistically_significant": bool(ks_pvalue < 0.05),
                "drift_tier": tier,
                "status_color": color,
                "ref_mean": round(float(np.mean(ref_vals)), 2),
                "curr_mean": round(float(np.mean(curr_vals)), 2),
                "ref_std": round(float(np.std(ref_vals)), 2),
                "curr_std": round(float(np.std(curr_vals)), 2),
            })

        # Overall cohort safety status
        if max_psi < 0.10:
            cohort_status = "Healthy / Stable Population"
            safety_code = "GREEN"
        elif max_psi < 0.25:
            cohort_status = "Moderate Population Shift Detected"
            safety_code = "AMBER"
        else:
            cohort_status = "Critical Clinical Distribution Shift - Retraining Alert"
            safety_code = "RED"

        report = {
            "cohort_safety_code": safety_code,
            "cohort_status": cohort_status,
            "max_psi": round(max_psi, 4),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "monitored_samples": len(current_df),
            "features": feature_reports,
        }

        return report

    def export_report(self, report: Dict[str, Any], filename: str = "clinical_drift_report.json") -> Path:
        """Exports JSON drift summary report."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_DIR / filename
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[DriftDetector] Drift evaluation report exported to: {save_path}")
        return save_path


def run_drift_monitoring_check() -> Dict[str, Any]:
    """Runs a demonstration monitoring check comparing test cohort against training baseline."""
    print("=" * 75)
    print("CLINICAL COHORT DRIFT & POPULATION STABILITY MONITORING")
    print("=" * 75)

    test_path = PROCESSED_DATA_DIR / "test_tabular.csv"
    if not test_path.exists():
        raise FileNotFoundError("test_tabular.csv not found. Run preprocessing first.")

    df_test = pd.read_csv(test_path)
    detector = ClinicalDriftDetector()
    report = detector.evaluate_cohort(df_test)
    detector.export_report(report)

    print(f"\nCohort Safety Code: {report['cohort_safety_code']}")
    print(f"Cohort Status:      {report['cohort_status']}")
    print(f"Max PSI:            {report['max_psi']:.4f}")
    print(f"Drifted Biomarkers: {report['drifted_features_count']}")

    print("\n--- Feature-Level Stability Analysis ---")
    print(f"{'Feature':<28} | {'PSI':<8} | {'KS Stat':<8} | {'p-value':<10} | {'Status':<16}")
    print("-" * 75)
    for f in report["features"]:
        print(
            f"{f['feature']:<28} | {f['psi']:<8.4f} | {f['ks_statistic']:<8.4f} | "
            f"{f['ks_pvalue']:<10.4f} | {f['drift_tier']:<16}"
        )

    return report


if __name__ == "__main__":
    run_drift_monitoring_check()
