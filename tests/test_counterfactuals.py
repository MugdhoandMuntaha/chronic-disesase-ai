"""Unit tests for Actionable Clinical Recourse and Counterfactual Optimization."""

import pandas as pd
import pytest

from src.explainability.counterfactuals import (
    CLINICAL_LEVERS,
    ClinicalCounterfactualExplainer,
)


@pytest.fixture
def explainer():
    return ClinicalCounterfactualExplainer()


def test_explainer_initialization(explainer):
    assert explainer.model is not None
    assert len(explainer.feature_names) > 50


def test_low_risk_patient_no_recourse_needed(explainer):
    # Construct a synthetic low-risk control profile
    patient = {f: 0.0 for f in explainer.feature_names}
    patient["age"] = 35.0
    patient["is_male"] = 0
    patient["baseline_bmi"] = 22.0
    patient["fasting_glucose_latest"] = 82.0
    patient["fasting_glucose_mean"] = 82.0
    patient["hba1c_latest"] = 5.0
    patient["hba1c_mean"] = 5.0
    patient["systolic_bp_latest"] = 112.0
    patient["systolic_bp_mean"] = 112.0

    result = explainer.generate_recourse(patient, target_risk=0.20)
    assert result["recourse_needed"] is False
    assert result["target_achieved"] is True
    assert result["actions_count"] == 0


def test_high_risk_patient_recourse_and_immutability(explainer):
    # Load high-risk patient from test cohort
    df_test = pd.read_csv("data/processed/test_tabular.csv")
    pt = df_test[df_test["patient_id"] == "PT_000019"].iloc[0]

    result = explainer.generate_recourse(pt, target_risk=0.20)

    assert result["recourse_needed"] is True
    assert result["actions_count"] > 0
    assert result["counterfactual_risk"] < result["initial_risk"]
    assert result["absolute_risk_reduction"] > 0.50

    # Verify immutable demographic features are strictly unchanged
    immutable_keys = ["age", "is_male", "family_history_diabetes"]
    for k in immutable_keys:
        if k in pt:
            assert pt[k] == result["counterfactual_features"][k], f"Immutable feature {k} was altered!"

    # Verify physiological safety floors are respected
    for action in result["recommended_actions"]:
        assert action["target_value"] < action["baseline_value"], "Target value must represent a reduction"
        assert action["required_reduction"] > 0.0
        # Check against floors
        if action["lever_name"] == "fasting_glucose":
            assert action["target_value"] >= 85.0
        elif action["lever_name"] == "systolic_bp":
            assert action["target_value"] >= 115.0
        elif action["lever_name"] == "hba1c":
            assert action["target_value"] >= 5.2
