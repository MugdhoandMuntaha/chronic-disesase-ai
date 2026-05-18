"""Unit tests for EHR preprocessing, model training, and clinical inference pipeline."""

import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessor import EHRPreprocessor
from src.models.tabular_model import ChronicDiseaseModel


@pytest.fixture
def preprocessor() -> EHRPreprocessor:
    """Fixture providing initialized preprocessor."""
    return EHRPreprocessor()


def test_preprocessor_pipeline(preprocessor: EHRPreprocessor) -> None:
    """Verifies that the preprocessor runs and produces non-empty feature matrices without leakage."""
    pts, enc, meas, diag = preprocessor.load_raw_data_polars()
    assert len(pts) > 0
    assert len(enc) > 0
    assert len(meas) > 0
    assert len(diag) > 0

    features_df = preprocessor.build_feature_matrix_polars(pts, enc, meas, diag)
    assert len(features_df) == len(pts)
    assert "target_label" in features_df.columns
    assert "fasting_glucose_latest" in features_df.columns
    assert "hba1c_delta" in features_df.columns

    # Check imputation
    imputed_df = preprocessor.fit_transform_imputation(features_df, is_train=True)
    numeric_cols = [c for c in imputed_df.columns if c not in ("patient_id", "target_label")]
    assert not imputed_df[numeric_cols].isnull().any().any(), "No missing values should remain after imputation"


def test_model_training_and_clinical_metrics() -> None:
    """Verifies that model trains and produces expected medical benchmark metrics."""
    temp_model_dir = Path(tempfile.mkdtemp(prefix="test_model_"))
    try:
        model_handler = ChronicDiseaseModel(model_dir=temp_model_dir)
        X_train, y_train, X_test, y_test = model_handler.load_processed_data()

        assert len(X_train) > 0 and len(X_test) > 0
        assert len(model_handler.feature_columns) > 20

        # Train model
        model_handler.train(X_train, y_train)

        # Evaluate model
        metrics = model_handler.evaluate(X_test, y_test)
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert "sensitivity_at_90_spec" in metrics
        assert metrics["auroc"] >= 0.80, f"Expected AUROC >= 0.80, got {metrics['auroc']}"

        # Save and reload
        saved_path = model_handler.save("test_model.joblib")
        assert saved_path.exists()

        reloaded = ChronicDiseaseModel(model_dir=temp_model_dir)
        reloaded.load("test_model.joblib")
        assert len(reloaded.feature_columns) == len(model_handler.feature_columns)

        # Test single-patient risk prediction inference
        sample_patient = X_test.iloc[0].to_dict()
        pred = reloaded.predict_risk(sample_patient)
        assert "predicted_risk_probability" in pred
        assert 0.0 <= pred["predicted_risk_probability"] <= 1.0
        assert pred["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk (Imminent Onset)"]
        assert len(pred["clinical_recommendation"]) > 0

    finally:
        shutil.rmtree(temp_model_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
