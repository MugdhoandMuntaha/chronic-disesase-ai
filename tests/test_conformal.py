"""Unit tests for Split Conformal Prediction and Uncertainty Quantification."""

import numpy as np
import pytest

from src.evaluation.conformal import ClinicalConformalPredictor, compute_ece


def test_compute_ece_perfect_calibration():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.2, 0.8, 0.9, 0.9])
    ece, mce, bins = compute_ece(y_true, y_prob, num_bins=5)

    assert isinstance(ece, float)
    assert isinstance(mce, float)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert len(bins) > 0


def test_conformal_calibration_threshold():
    np.random.seed(42)
    y_calib = np.random.binomial(1, 0.25, size=200)
    # Simulate well-calibrated probabilities
    y_prob_calib = np.where(y_calib == 1, np.random.uniform(0.6, 0.99, size=200), np.random.uniform(0.01, 0.4, size=200))

    predictor = ClinicalConformalPredictor(alpha=0.10)
    q_hat = predictor.calibrate(y_calib, y_prob_calib)

    assert predictor.q_hat is not None
    assert 0.0 < q_hat < 1.0


def test_conformal_prediction_set_generation():
    predictor = ClinicalConformalPredictor(alpha=0.10)
    predictor.q_hat = 0.20

    # Low probability: should return [0]
    set_low = predictor.predict_set(0.05)
    assert set_low == [0]

    # High probability: should return [1]
    set_high = predictor.predict_set(0.95)
    assert set_high == [1]

    # Borderline ambiguous probability: should contain {0, 1}
    # For q_hat = 0.20:
    # class 0 included if prob <= 0.20
    # class 1 included if (1 - prob) <= 0.20 -> prob >= 0.80
    # Let's test with higher q_hat = 0.60
    predictor.q_hat = 0.60
    set_ambiguous = predictor.predict_set(0.50)
    assert 0 in set_ambiguous and 1 in set_ambiguous


def test_conformal_empirical_coverage():
    np.random.seed(42)
    n = 300
    y_true = np.random.binomial(1, 0.3, size=n)
    y_prob = np.where(y_true == 1, np.random.uniform(0.55, 0.95, size=n), np.random.uniform(0.05, 0.45, size=n))

    # Split calibration and test
    y_cal, y_eval = y_true[:150], y_true[150:]
    p_cal, p_eval = y_prob[:150], y_prob[150:]

    predictor = ClinicalConformalPredictor(alpha=0.10)
    predictor.calibrate(y_cal, p_cal)
    results = predictor.evaluate(y_eval, p_eval)

    assert results["empirical_coverage"] >= 0.85
    assert results["guarantee_satisfied"] is True
    assert "singleton_non_disease_count" in results
    assert "singleton_onset_count" in results
