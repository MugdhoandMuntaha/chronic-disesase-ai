"""Unit tests for Step 3: Tabular Baseline Model with XGBoost (src/models/baseline_xgb.py)."""

import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from xgboost import XGBClassifier

from src.models.baseline_xgb import XGBoostBaselineTrainer


@pytest.fixture(scope="module")
def trainer() -> XGBoostBaselineTrainer:
    """Fixture providing initialized XGBoostBaselineTrainer."""
    return XGBoostBaselineTrainer(random_state=42)


def test_load_data(trainer: XGBoostBaselineTrainer) -> None:
    """Verifies data loading, column exclusion, and feature matrix separation."""
    X_train, y_train, X_test, y_test = trainer.load_data()

    assert X_train.shape[0] == 800
    assert X_test.shape[0] == 200
    assert len(trainer.feature_names) == X_train.shape[1]

    # Verify ID and target columns are completely excluded from features
    assert "patient_id" not in X_train.columns
    assert "target_disease" not in X_train.columns
    assert "target_label" not in X_train.columns

    # Verify target labels are binary
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_scale_pos_weight(trainer: XGBoostBaselineTrainer) -> None:
    """Verifies automatic class imbalance ratio calculation."""
    _, y_train, _, _ = trainer.load_data()
    weight = trainer.calculate_scale_pos_weight(y_train)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    expected_weight = neg_count / pos_count

    assert np.isclose(weight, expected_weight, rtol=1e-3)
    assert weight > 1.0, "Weight should be greater than 1 given ~15% positive prevalence"


def test_model_training_and_evaluation(trainer: XGBoostBaselineTrainer) -> None:
    """Verifies training execution and core clinical evaluation metrics."""
    X_train, y_train, X_test, y_test = trainer.load_data()
    model = trainer.train(X_train, y_train, X_test, y_test)

    assert model is not None
    metrics = trainer.evaluate(X_test, y_test)

    # Verify metric keys
    required_metrics = ["auroc", "auprc", "sensitivity", "specificity", "brier_score"]
    for m in required_metrics:
        assert m in metrics
        assert 0.0 <= metrics[m] <= 1.0

    # Medical threshold checks
    assert metrics["auroc"] >= 0.85
    assert metrics["auprc"] >= 0.80
    assert metrics["sensitivity"] >= 0.80
    assert metrics["specificity"] >= 0.80


def test_save_and_reload_model(trainer: XGBoostBaselineTrainer) -> None:
    """Verifies native JSON model artifact persistence and reloading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer.models_dir = Path(tmp_dir)
        save_path = trainer.save_model("test_xgb.json")

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        # Reload with raw XGBClassifier
        reloaded_model = XGBClassifier()
        reloaded_model.load_model(str(save_path))

        X_train, _, X_test, _ = trainer.load_data()
        preds = reloaded_model.predict_proba(X_test)[:, 1]
        assert len(preds) == len(X_test)
        assert ((preds >= 0.0) & (preds <= 1.0)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
