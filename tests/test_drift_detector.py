"""Automated Unit Tests for Clinical Covariate Drift & PSI Monitoring."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import ClinicalDriftDetector, calculate_psi

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_psi_identical_distributions():
    """Verifies that identical distributions produce PSI close to zero."""
    np.random.seed(42)
    ref = np.random.normal(100, 15, 1000)
    curr = np.random.normal(100, 15, 1000)

    psi, _ = calculate_psi(ref, curr, num_bins=10)
    assert psi < 0.05, f"Expected PSI < 0.05 for identical distributions, got {psi}"


def test_psi_shifted_distribution():
    """Verifies that significantly shifted distributions produce large PSI (> 0.25)."""
    np.random.seed(42)
    ref = np.random.normal(100, 15, 1000)
    curr = np.random.normal(140, 15, 1000)  # Severe distribution shift (+40 mg/dL)

    psi, _ = calculate_psi(ref, curr, num_bins=10)
    assert psi > 0.25, f"Expected PSI > 0.25 for shifted distribution, got {psi}"


def test_drift_detector_cohort_evaluation():
    """Verifies that ClinicalDriftDetector evaluates dataframe cohorts with proper schema."""
    detector = ClinicalDriftDetector()
    test_path = PROJECT_ROOT / "data" / "processed" / "test_tabular.csv"
    assert test_path.exists()

    df_test = pd.read_csv(test_path)
    report = detector.evaluate_cohort(df_test)

    assert "cohort_safety_code" in report
    assert report["cohort_safety_code"] in ["GREEN", "AMBER", "RED"]
    assert "max_psi" in report
    assert "features" in report
    assert len(report["features"]) > 0

    first_feat = report["features"][0]
    assert "feature" in first_feat
    assert "psi" in first_feat
    assert "ks_statistic" in first_feat
    assert "ks_pvalue" in first_feat
    assert "drift_tier" in first_feat
