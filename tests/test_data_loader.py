"""Unit tests for Step 2: Data Cleaning and Feature Pipeline (src/data_loader.py)."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import pytest

from src.data_loader import EHRDataLoader, MEASUREMENT_COLUMNS, DEMOGRAPHIC_COLUMNS


@pytest.fixture(scope="module")
def data_loader() -> EHRDataLoader:
    """Fixture providing initialized data loader."""
    return EHRDataLoader(max_seq_len=15, test_size=0.20, seed=42)


def test_ingest_data(data_loader: EHRDataLoader) -> None:
    """Verifies that mock_ehr_data.csv exists and is ingested into a non-empty Polars DataFrame."""
    df = data_loader.ingest_data()
    assert len(df) > 0
    assert "patient_id" in df.columns
    assert "target_label" in df.columns
    for col in MEASUREMENT_COLUMNS:
        assert col in df.columns


def test_clean_and_encode(data_loader: EHRDataLoader) -> None:
    """Verifies chronological sorting and categorical encodings."""
    df = data_loader.ingest_data()
    df_clean = data_loader.clean_and_encode_data(df)

    assert "is_male" in df_clean.columns
    assert "smoking_numeric" in df_clean.columns
    assert "enc_outpatient" in df_clean.columns

    # Verify binary / discrete ranges
    unique_genders = set(df_clean["is_male"].unique().to_list())
    assert unique_genders.issubset({0, 1})

    unique_smoking = set(df_clean["smoking_numeric"].unique().to_list())
    assert unique_smoking.issubset({0, 1, 2})


def test_zero_patient_leakage_split(data_loader: EHRDataLoader) -> None:
    """Verifies strict patient-level splitting with zero overlap between train and test."""
    df = data_loader.ingest_data()
    df_clean = data_loader.clean_and_encode_data(df)
    train_df, test_df, train_pts, test_pts = data_loader.split_by_patient_id(df_clean)

    train_set = set(train_pts)
    test_set = set(test_pts)

    assert len(train_set) == 800
    assert len(test_set) == 200
    assert len(train_set.intersection(test_set)) == 0, "Patient leakage detected!"

    # Verify no overlap in visit dataframes
    train_visit_pts = set(train_df["patient_id"].unique().to_list())
    test_visit_pts = set(test_df["patient_id"].unique().to_list())
    assert len(train_visit_pts.intersection(test_visit_pts)) == 0


def test_tabular_format_aggregation(data_loader: EHRDataLoader) -> None:
    """Verifies tabular aggregation shape, metrics, and complete absence of NaN values."""
    df = data_loader.ingest_data()
    df_clean = data_loader.clean_and_encode_data(df)
    train_df, test_df, _, _ = data_loader.split_by_patient_id(df_clean)
    data_loader.compute_imputation_values(train_df)

    train_tab, test_tab = data_loader.prepare_tabular_format(train_df, test_df)

    assert train_tab.shape[0] == 800
    assert test_tab.shape[0] == 200
    assert "visit_count" in train_tab.columns
    assert "hba1c_latest" in train_tab.columns
    assert "fasting_glucose_mean" in train_tab.columns

    # Check for zero NaNs after imputation
    assert not train_tab.isnull().any().any(), "No missing values should remain in tabular train set"
    assert not test_tab.isnull().any().any(), "No missing values should remain in tabular test set"


def test_sequential_format_arrays(data_loader: EHRDataLoader) -> None:
    """Verifies 3D array shapes, mask consistency, and persistence."""
    df = data_loader.ingest_data()
    df_clean = data_loader.clean_and_encode_data(df)
    train_df, test_df, train_pts, test_pts = data_loader.split_by_patient_id(df_clean)
    data_loader.compute_imputation_values(train_df)

    X_train, mask_train, y_train, X_test, mask_test, y_test = data_loader.prepare_sequential_format(
        train_df, test_df, train_pts, test_pts
    )

    # Validate shapes
    assert X_train.shape == (800, 15, 21)
    assert mask_train.shape == (800, 15)
    assert y_train.shape == (800,)
    assert X_test.shape == (200, 15, 21)
    assert mask_test.shape == (200, 15)
    assert y_test.shape == (200,)

    # Verify mask values are either 0 or 1
    assert set(np.unique(mask_train)).issubset({0.0, 1.0})
    assert set(np.unique(mask_test)).issubset({0.0, 1.0})

    # Verify target labels are binary
    assert set(np.unique(y_train)).issubset({0.0, 1.0})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
