"""Unit tests for synthetic EHR generator module."""

import csv
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.data.generator import SyntheticEHRGenerator





@pytest.fixture
def generator() -> SyntheticEHRGenerator:
    """Fixture providing a small deterministic generator instance."""
    return SyntheticEHRGenerator(num_patients=100, prevalence=0.20, seed=123)


def test_patient_cohort_generation(generator: SyntheticEHRGenerator) -> None:
    """Verifies patient generation size, columns, and target label bounds."""
    patients = generator.generate_patient_cohort()

    assert len(patients) == 100
    expected_keys = {
        "patient_id",
        "age",
        "gender",
        "ethnicity",
        "smoking_status",
        "baseline_bmi",
        "family_history_diabetes",
        "index_date",
        "target_label",
    }
    for pt in patients:
        assert expected_keys.issubset(pt.keys())
        assert 18 <= pt["age"] <= 89
        assert 15.0 <= pt["baseline_bmi"] <= 60.0
        assert pt["target_label"] in (0, 1)

    positives = sum(p["target_label"] for p in patients)
    # Prevalence should be in a reasonable clinical range (10% - 35% on sample of 100)
    assert 10 <= positives <= 35


def test_longitudinal_records_integrity(generator: SyntheticEHRGenerator) -> None:
    """Verifies relational integrity across encounters, measurements, and diagnoses."""
    patients = generator.generate_patient_cohort()
    encounters, measurements, diagnoses = generator.generate_longitudinal_records(patients)

    patient_ids = {p["patient_id"] for p in patients}
    assert len(encounters) > len(patients), "There should be multiple encounters per patient"
    assert len(measurements) == len(encounters), "Each encounter should have a measurement record"

    encounter_ids = set()
    for enc in encounters:
        assert enc["patient_id"] in patient_ids, "Encounter patient_id must exist in patients"
        assert enc["days_to_index"] <= 0, "Observation days must precede index date (no future leakage)"
        encounter_ids.add(enc["encounter_id"])

    for meas in measurements:
        assert meas["patient_id"] in patient_ids
        assert meas["encounter_id"] in encounter_ids
        # Validate physiological bounds on observed values
        if meas["fasting_glucose"] is not None:
            assert 40.0 <= meas["fasting_glucose"] <= 400.0
        if meas["hba1c"] is not None:
            assert 3.5 <= meas["hba1c"] <= 16.0
        if meas["systolic_bp"] is not None:
            assert 60.0 <= meas["systolic_bp"] <= 240.0
        if meas["egfr"] is not None:
            assert 5.0 <= meas["egfr"] <= 140.0

    for diag in diagnoses:
        assert diag["patient_id"] in patient_ids
        assert diag["encounter_id"] in encounter_ids
        assert len(diag["icd10_code"]) > 0


def test_clinical_trajectory_differentiation(generator: SyntheticEHRGenerator) -> None:
    """Verifies that positive cases display higher average glucose & HbA1c than controls."""
    patients = generator.generate_patient_cohort()
    encounters, measurements, diagnoses = generator.generate_longitudinal_records(patients)

    pos_patient_ids = {p["patient_id"] for p in patients if p["target_label"] == 1}
    neg_patient_ids = {p["patient_id"] for p in patients if p["target_label"] == 0}

    pos_glucose = [m["fasting_glucose"] for m in measurements if m["patient_id"] in pos_patient_ids and m["fasting_glucose"] is not None]
    neg_glucose = [m["fasting_glucose"] for m in measurements if m["patient_id"] in neg_patient_ids and m["fasting_glucose"] is not None]

    assert len(pos_glucose) > 0 and len(neg_glucose) > 0
    mean_pos_glucose = sum(pos_glucose) / len(pos_glucose)
    mean_neg_glucose = sum(neg_glucose) / len(neg_glucose)

    # Positive patients must have higher mean fasting glucose
    assert mean_pos_glucose > mean_neg_glucose, (
        f"Positive glucose ({mean_pos_glucose:.1f}) should exceed negative ({mean_neg_glucose:.1f})"
    )


def test_save_cohort_to_disk(generator: SyntheticEHRGenerator) -> None:
    """Verifies saving tables to disk creates valid files with expected row counts."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_ehr_"))
    try:
        saved = generator.run(output_dir=temp_dir, save_format="csv")
        assert len(saved["csv"]) == 4

        for csv_file in saved["csv"]:
            assert csv_file.exists()
            assert csv_file.stat().st_size > 0
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
                assert len(header) > 0
                assert len(rows) > 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v"])
    else:
        print("Running tests in standalone mode...")
        gen = SyntheticEHRGenerator(num_patients=100, prevalence=0.20, seed=123)
        print("  -> Running test_patient_cohort_generation...")
        test_patient_cohort_generation(gen)
        print("  -> Running test_longitudinal_records_integrity...")
        test_longitudinal_records_integrity(gen)
        print("  -> Running test_clinical_trajectory_differentiation...")
        test_clinical_trajectory_differentiation(gen)
        print("  -> Running test_save_cohort_to_disk...")
        test_save_cohort_to_disk(gen)
        print("All generator tests PASSED successfully!")

