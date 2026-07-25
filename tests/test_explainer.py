"""Unit tests for Step 5: Clinical Model Explainability with SHAP (src/explainability/explainer.py)."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from src.explainability.explainer import ClinicalSHAPExplainer


@pytest.fixture(scope="module")
def explainer() -> ClinicalSHAPExplainer:
    """Fixture providing initialized ClinicalSHAPExplainer."""
    exp = ClinicalSHAPExplainer()
    exp.init_explainer()
    return exp


def test_explainer_initialization(explainer: ClinicalSHAPExplainer) -> None:
    """Verifies that TreeExplainer initializes and computes SHAP matrices for the test cohort."""
    assert explainer.model is not None
    assert explainer.X_test is not None
    assert explainer.shap_values is not None

    # Verify SHAP matrix shape matches test instances and features
    assert explainer.shap_values.shape == (200, len(explainer.feature_names))
    assert np.isfinite(explainer.base_value)


def test_global_feature_importance(explainer: ClinicalSHAPExplainer) -> None:
    """Verifies calculation of cohort-level mean absolute SHAP feature ranking."""
    top_10 = explainer.get_global_feature_importance(top_k=10)

    assert len(top_10) == 10
    assert "Feature" in top_10.columns
    assert "Mean_Abs_SHAP" in top_10.columns

    # Verify ranking is descending
    shap_vals = top_10["Mean_Abs_SHAP"].values
    assert (np.diff(shap_vals) <= 1e-6).all(), "Feature importance must be sorted descending"
    assert (shap_vals >= 0.0).all(), "Mean absolute SHAP values must be non-negative"


def test_patient_local_explanation(explainer: ClinicalSHAPExplainer) -> None:
    """Verifies local patient risk breakdown into risk drivers and protective factors."""
    explanation = explainer.explain_patient(patient_idx=0, top_k=5)

    assert explanation["patient_index"] == 0
    assert 0.0 <= explanation["predicted_probability"] <= 1.0
    assert "top_risk_drivers" in explanation
    assert "top_protective_factors" in explanation

    # Check structure of risk driver items
    for item in explanation["top_risk_drivers"]:
        assert "feature" in item
        assert "observed_value" in item
        assert "shap_value" in item
        assert item["shap_value"] > 0, "Risk drivers must have positive SHAP impact"

    # Check structure of protective factor items
    for item in explanation["top_protective_factors"]:
        assert "feature" in item
        assert "observed_value" in item
        assert "shap_value" in item
        assert item["shap_value"] < 0, "Protective factors must have negative SHAP impact"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
