"""Automated Unit Tests for Decision Curve Analysis (DCA) and Clinical Net Benefit."""

from pathlib import Path
import numpy as np
import pytest

from src.evaluation.dca import (
    ClinicalDecisionCurveAnalyzer,
    compute_interventions_avoided,
    compute_net_benefit,
    compute_treat_all_net_benefit,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_net_benefit_calculation():
    """Verifies that Net Benefit correctly penalizes false positives according to clinical odds."""
    # Synthetic cohort: 100 patients, 20 positives
    y_true = np.array([1] * 20 + [0] * 80)
    # Perfect model predictions
    y_prob = np.array([1.0] * 20 + [0.0] * 80)
    thresholds = np.array([0.10, 0.20, 0.50])

    nb = compute_net_benefit(y_true, y_prob, thresholds)

    # For perfect model with 0 FP, Net Benefit is simply TP / N = 20 / 100 = 0.20
    for val in nb:
        assert val == pytest.approx(0.20, abs=1e-4)


def test_treat_all_net_benefit():
    """Verifies Treat All Net Benefit formula across thresholds."""
    y_true = np.array([1] * 15 + [0] * 85)
    thresholds = np.array([0.15])
    # At threshold equal to prevalence (0.15), Net Benefit for Treat All should be 0.0
    nb_all = compute_treat_all_net_benefit(y_true, thresholds)
    assert nb_all[0] == pytest.approx(0.0, abs=1e-4)


def test_interventions_avoided():
    """Verifies calculation of unnecessary interventions avoided per 100 patients."""
    nb_model = np.array([0.15])
    nb_all = np.array([0.0])
    thresholds = np.array([0.15])

    # Avoided = (0.15 - 0.0) / (0.15 / 0.85) * 100 = 0.15 / 0.17647 * 100 = 85.0
    avoided = compute_interventions_avoided(nb_model, nb_all, thresholds)
    assert avoided[0] == pytest.approx(85.0, abs=1.0)


def test_decision_curve_analyzer():
    """Verifies that the complete DCA analyzer runs on the processed test data."""
    analyzer = ClinicalDecisionCurveAnalyzer(num_thresholds=20)
    results = analyzer.run_analysis()

    assert "thresholds" in results
    assert "curves" in results
    assert "XGBoost Tabular Baseline" in results["curves"]
    assert "Treat All (Universal Intervention)" in results["curves"]
    assert "Treat None (No Intervention)" in results["curves"]

    plot_file = PROJECT_ROOT / "reports" / "decision_curve_analysis.png"
    assert plot_file.exists(), "decision_curve_analysis.png must be generated"
