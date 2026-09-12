"""Unit tests for Step 6: FastAPI Clinical Inference and Explainability Microservice (src/api/main.py)."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing TestClient with lifespan context execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient) -> None:
    """Verifies API root status and documentation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["docs_url"] == "/docs"


def test_health_check(client: TestClient) -> None:
    """Verifies that models are loaded and API reports healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["xgboost_tabular_model_loaded"] is True
    assert data["pytorch_sequence_gru_loaded"] is True
    assert data["shap_explainer_ready"] is True


def test_predict_tabular_endpoint(client: TestClient) -> None:
    """Verifies tabular prediction on an escalating risk patient."""
    payload = {
        "patient_id": "PT_TEST_001",
        "age": 62.0,
        "is_male": 1,
        "baseline_bmi": 34.2,
        "smoking_numeric": 1,
        "family_history_diabetes": 1,
        "fasting_glucose_latest": 138.0,
        "fasting_glucose_delta": 24.0,
        "hba1c_latest": 6.6,
        "hba1c_delta": 0.8,
        "systolic_bp_latest": 145.0,
        "diastolic_bp_latest": 90.0,
        "egfr_latest": 72.0,
        "triglycerides_latest": 235.0,
        "has_I10": 1,
        "has_R73.03": 1,
        "has_E78.5": 1,
    }

    response = client.post("/predict/tabular", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert res["patient_id"] == "PT_TEST_001"
    assert 0.0 <= res["predicted_risk_probability"] <= 1.0
    assert res["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk (Imminent Onset)"]
    assert len(res["clinical_recommendation"]) > 0
    assert len(res["actionable_next_steps"]) > 0


def test_predict_sequence_endpoint(client: TestClient) -> None:
    """Verifies sequence GRU prediction across longitudinal encounters."""
    payload = {
        "patient_id": "PT_SEQ_001",
        "age": 55.0,
        "is_male": 0,
        "baseline_bmi": 28.5,
        "smoking_numeric": 0,
        "family_history_diabetes": 1,
        "visits": [
            {
                "days_to_index": -360,
                "encounter_type": "outpatient",
                "systolic_bp": 128.0,
                "diastolic_bp": 80.0,
                "heart_rate": 72.0,
                "fasting_glucose": 102.0,
                "hba1c": 5.7,
            },
            {
                "days_to_index": -180,
                "encounter_type": "outpatient",
                "systolic_bp": 134.0,
                "diastolic_bp": 84.0,
                "heart_rate": 74.0,
                "fasting_glucose": 115.0,
                "hba1c": 6.0,
            },
            {
                "days_to_index": -30,
                "encounter_type": "outpatient",
                "systolic_bp": 142.0,
                "diastolic_bp": 88.0,
                "heart_rate": 78.0,
                "fasting_glucose": 130.0,
                "hba1c": 6.4,
            },
        ],
    }

    response = client.post("/predict/sequence", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert res["patient_id"] == "PT_SEQ_001"
    assert 0.0 <= res["predicted_risk_probability"] <= 1.0
    assert res["model_version"] == "PyTorch-Sequence-GRU-v1.0"


def test_explain_endpoint(client: TestClient) -> None:
    """Verifies SHAP explanation endpoint decomposing positive and protective factors."""
    payload = {
        "patient_id": "PT_TEST_EXPLAIN",
        "age": 45.0,
        "is_male": 1,
        "baseline_bmi": 24.0,
        "smoking_numeric": 0,
        "family_history_diabetes": 0,
        "fasting_glucose_latest": 88.0,
        "fasting_glucose_delta": 2.0,
        "hba1c_latest": 5.1,
        "hba1c_delta": 0.0,
        "systolic_bp_latest": 118.0,
        "diastolic_bp_latest": 74.0,
    }

    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert res["patient_id"] == "PT_TEST_EXPLAIN"
    assert "cohort_base_value" in res
    assert "top_risk_drivers" in res
    assert "top_protective_factors" in res


def test_metrics_endpoint(client: TestClient) -> None:
    """Verifies medical benchmark retrieval for models."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "models" in data
    assert "xgboost_tabular" in data["models"]
    assert "pytorch_sequence_gru" in data["models"]


def test_predict_retain_endpoint(client: TestClient) -> None:
    """Verifies that the RETAIN prediction endpoint returns probabilities and visit attributions."""
    payload = {
        "patient_id": "PT_RETAIN_001",
        "age": 62.0,
        "is_male": 1,
        "baseline_bmi": 32.5,
        "smoking_numeric": 1,
        "family_history_diabetes": 1,
        "visits": [
            {
                "days_to_index": -360,
                "encounter_type": "outpatient",
                "systolic_bp": 130.0,
                "diastolic_bp": 82.0,
                "fasting_glucose": 105.0,
                "hba1c": 5.8,
            },
            {
                "days_to_index": -180,
                "encounter_type": "outpatient",
                "systolic_bp": 138.0,
                "diastolic_bp": 86.0,
                "fasting_glucose": 118.0,
                "hba1c": 6.2,
            },
            {
                "days_to_index": -30,
                "encounter_type": "outpatient",
                "systolic_bp": 145.0,
                "diastolic_bp": 90.0,
                "fasting_glucose": 135.0,
                "hba1c": 6.7,
            },
        ],
    }

    response = client.post("/predict/retain", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert res["patient_id"] == "PT_RETAIN_001"
    assert 0.0 <= res["predicted_risk_probability"] <= 1.0
    assert res["model_version"] == "PyTorch-RETAIN-v1.0"
    assert "visit_attributions" in res
    assert len(res["visit_attributions"]) == 3
    assert "visit_attention_alpha" in res["visit_attributions"][0]


def test_predict_conformal_endpoint(client: TestClient) -> None:
    """Verifies conformal prediction set and uncertainty quantification endpoint."""
    payload = {
        "patient_id": "PT_CONF_001",
        "age": 58.0,
        "is_male": 1,
        "baseline_bmi": 32.0,
        "smoking_numeric": 1,
        "family_history_diabetes": 1,
        "fasting_glucose_latest": 128.0,
        "hba1c_latest": 6.4,
        "systolic_bp_latest": 140.0,
        "diastolic_bp_latest": 88.0,
    }
    response = client.post("/predict/conformal", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert res["patient_id"] == "PT_CONF_001"
    assert res["target_coverage_guarantee"] == 0.90
    assert isinstance(res["prediction_set"], list)
    assert len(res["prediction_set"]) >= 1
    assert isinstance(res["is_ambiguous"], bool)
    assert "clinical_directive" in res


def test_counterfactual_recourse_endpoint(client: TestClient) -> None:
    """Verifies actionable clinical recourse endpoint."""
    payload = {
        "patient_id": "PT_000019",
        "target_risk": 0.20,
    }
    response = client.post("/explain/counterfactual", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert res["patient_id"] == "PT_000019"
    assert res["target_risk"] == 0.20
    assert res["recourse_needed"] is True
    assert "recommended_actions" in res
    assert len(res["recommended_actions"]) > 0
    assert res["absolute_risk_reduction"] > 0.0


def test_fairness_audit_endpoint(client: TestClient) -> None:
    """Verifies demographic fairness and bias audit endpoint."""
    response = client.get("/fairness")
    assert response.status_code == 200
    res = response.json()

    assert "audit_title" in res
    assert "biological_sex_audit" in res
    assert "age_bracket_audit" in res
    assert "regulatory_summary" in res
    assert "disparity_analysis" in res["biological_sex_audit"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

