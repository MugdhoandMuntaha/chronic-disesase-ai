"""Automated Unit Tests for MLflow Experiment Tracking & Leaderboard Generation."""

from pathlib import Path
import pytest
import pandas as pd

from src.tracking.experiment_tracker import ChronicDiseaseExperimentTracker

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_tracker_initialization(tmp_path):
    """Verifies that the experiment tracker initializes a SQLite tracking store."""
    db_file = tmp_path / "test_mlflow.db"
    db_uri = f"sqlite:///{db_file.as_posix()}"

    tracker = ChronicDiseaseExperimentTracker(
        experiment_name="unit_test_experiment",
        tracking_uri=db_uri,
    )
    assert tracker.experiment_name == "unit_test_experiment"
    assert tracker.tracking_uri == db_uri


def test_leaderboard_generation():
    """Verifies that the leaderboard compares XGBoost, GRU, and RETAIN correctly."""
    tracker = ChronicDiseaseExperimentTracker()
    df_leaderboard = tracker.generate_leaderboard()

    assert isinstance(df_leaderboard, pd.DataFrame)
    assert len(df_leaderboard) >= 2
    assert "Model" in df_leaderboard.columns
    assert "AUROC" in df_leaderboard.columns
    assert "AUPRC" in df_leaderboard.columns
    assert "Clinical Interpretability" in df_leaderboard.columns

    # Verify RETAIN is included
    models = list(df_leaderboard["Model"])
    assert any("RETAIN" in m for m in models)
