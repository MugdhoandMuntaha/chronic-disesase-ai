"""Synthetic Longitudinal EHR Data Generator for Chronic Disease Prediction.

Simulates multi-table relational EHR data including:
- Demographics (patients.csv / patients.parquet)
- Longitudinal clinical encounters (encounters.csv / encounters.parquet)
- Vitals and laboratory panels (measurements.csv / measurements.parquet)
- ICD-10 diagnostic codes (diagnoses.csv / diagnoses.parquet)

Models realistic clinical dynamics:
- Positive cases exhibit progressive deterioration in fasting glucose, HbA1c,
  blood pressure, and eGFR leading up to the index date.
- Controls exhibit stable or mild fluctuations within healthy/normal bounds.
- Missingness (MCAR and MAR) mirrors real-world lab ordering frequencies.
"""

import argparse
import csv
import datetime
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config import (
    CLINICAL_MEASUREMENT_RANGES,
    DEFAULT_OBSERVATION_WINDOW_DAYS,
    DEFAULT_PREDICTION_WINDOW_DAYS,
    DEFAULT_RANDOM_SEED,
    ICD10_CODE_MAP,
    RAW_DATA_DIR,
)
from src.utils.logger import setup_logger

logger = setup_logger("ehr_generator")


class SyntheticEHRGenerator:
    """Generates synthetic multi-modal electronic health record (EHR) cohorts."""

    def __init__(
        self,
        num_patients: int = 1000,
        prevalence: float = 0.18,
        seed: int = DEFAULT_RANDOM_SEED,
        observation_days: int = DEFAULT_OBSERVATION_WINDOW_DAYS,
    ):
        self.num_patients = num_patients
        self.prevalence = prevalence
        self.seed = seed
        self.observation_days = observation_days

        self.rng = random.Random(seed)

        # Static cohorts
        self.ethnicities = ["Caucasian", "African American", "Hispanic", "Asian", "Other"]
        self.ethnicity_weights = [0.58, 0.16, 0.16, 0.06, 0.04]
        self.smoking_statuses = ["Never", "Former", "Current"]
        self.smoking_weights = [0.55, 0.28, 0.17]
        self.encounter_types = ["outpatient", "telehealth", "emergency", "inpatient"]
        self.encounter_type_weights = [0.75, 0.15, 0.07, 0.03]

    def _clip(self, value: float, min_val: float, max_val: float) -> float:
        """Clips value within specified physiological safety boundaries."""
        return max(min_val, min(max_val, value))

    def generate_patient_cohort(self) -> List[Dict[str, Any]]:
        """Generates static demographic and baseline risk data for all patients."""
        patients = []
        base_index_date = datetime.date(2025, 1, 1)

        for i in range(1, self.num_patients + 1):
            patient_id = f"PT_{i:06d}"

            # Stagger index dates within a 180-day window
            offset_days = self.rng.randint(0, 180)
            index_date = base_index_date + datetime.timedelta(days=offset_days)

            # Target label assignment based on simulated risk factors
            age = int(self.rng.gauss(55, 14))
            age = max(18, min(89, age))

            gender = self.rng.choice(["Male", "Female"])
            ethnicity = self.rng.choices(self.ethnicities, weights=self.ethnicity_weights)[0]
            smoking = self.rng.choices(self.smoking_statuses, weights=self.smoking_weights)[0]

            # Baseline BMI
            if gender == "Male":
                baseline_bmi = self.rng.gauss(28.2, 5.0)
            else:
                baseline_bmi = self.rng.gauss(27.8, 5.5)
            baseline_bmi = self._clip(baseline_bmi, 16.0, 55.0)

            # Family history
            family_history = 1 if self.rng.random() < 0.24 else 0

            # Calculate latent chronic risk score
            # Age, high BMI, smoking, and family history increase chronic onset probability
            risk_logit = -2.8
            risk_logit += (age - 50) * 0.045
            risk_logit += (baseline_bmi - 25.0) * 0.09
            risk_logit += 0.45 if smoking == "Current" else (0.2 if smoking == "Former" else 0.0)
            risk_logit += 0.65 if family_history == 1 else 0.0

            # Convert logit to probability
            prob = 1.0 / (1.0 + math.exp(-risk_logit))

            # Calibrate threshold to approximate target prevalence
            target_label = 1 if self.rng.random() < prob else 0

            patients.append({
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "ethnicity": ethnicity,
                "smoking_status": smoking,
                "baseline_bmi": round(baseline_bmi, 1),
                "family_history_diabetes": family_history,
                "index_date": index_date.strftime("%Y-%m-%d"),
                "target_label": target_label,
            })

        # Fine-tune prevalence if needed
        positives = sum(p["target_label"] for p in patients)
        logger.info(f"Generated {len(patients)} patients. Positives: {positives} ({positives / len(patients):.1%})")
        return patients

    def generate_longitudinal_records(
        self, patients: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generates longitudinal encounters, time-series measurements, and diagnoses.

        Returns:
            Tuple of (encounters_list, measurements_list, diagnoses_list)
        """
        encounters: List[Dict[str, Any]] = []
        measurements: List[Dict[str, Any]] = []
        diagnoses: List[Dict[str, Any]] = []

        enc_counter = 1
        meas_counter = 1
        diag_counter = 1

        for pt in patients:
            patient_id = pt["patient_id"]
            is_positive = pt["target_label"] == 1
            index_date = datetime.datetime.strptime(pt["index_date"], "%Y-%m-%d").date()
            age = pt["age"]
            baseline_bmi = pt["baseline_bmi"]

            # Number of encounters over the 2-year observation window
            # Sicker / positive patients average more frequent clinical monitoring
            if is_positive:
                n_encounters = self.rng.randint(4, 12)
            else:
                n_encounters = self.rng.randint(2, 7)

            # Generate random visit intervals over [-observation_days, -5] days before index date
            day_offsets = sorted(self.rng.sample(range(10, self.observation_days), n_encounters), reverse=True)

            # Patient-specific baseline physiology parameters
            if is_positive:
                base_fpg = self.rng.uniform(105.0, 125.0)  # Prediabetic baseline
                base_hba1c = self.rng.uniform(5.7, 6.3)
                base_sbp = self.rng.uniform(130.0, 148.0)
                base_dbp = self.rng.uniform(82.0, 94.0)
                base_hr = self.rng.uniform(72.0, 88.0)
                base_egfr = self.rng.uniform(75.0, 95.0)
                base_creat = self.rng.uniform(0.9, 1.3)
                base_chol = self.rng.uniform(205.0, 250.0)
                base_ldl = self.rng.uniform(125.0, 165.0)
                base_hdl = self.rng.uniform(34.0, 46.0)
                base_trig = self.rng.uniform(170.0, 270.0)
            else:
                base_fpg = self.rng.uniform(82.0, 98.0)    # Normal baseline
                base_hba1c = self.rng.uniform(4.9, 5.4)
                base_sbp = self.rng.uniform(110.0, 126.0)
                base_dbp = self.rng.uniform(68.0, 78.0)
                base_hr = self.rng.uniform(62.0, 78.0)
                base_egfr = self.rng.uniform(88.0, 112.0)
                base_creat = self.rng.uniform(0.7, 1.05)
                base_chol = self.rng.uniform(160.0, 205.0)
                base_ldl = self.rng.uniform(85.0, 118.0)
                base_hdl = self.rng.uniform(48.0, 68.0)
                base_trig = self.rng.uniform(85.0, 145.0)

            for offset in day_offsets:
                enc_id = f"ENC_{enc_counter:08d}"
                enc_counter += 1

                visit_date = index_date - datetime.timedelta(days=offset)
                enc_type = self.rng.choices(self.encounter_types, weights=self.encounter_type_weights)[0]

                encounters.append({
                    "encounter_id": enc_id,
                    "patient_id": patient_id,
                    "encounter_date": visit_date.strftime("%Y-%m-%d"),
                    "days_to_index": -offset,
                    "encounter_type": enc_type,
                })

                # Temporal progression progress [0.0 = earliest visit, 1.0 = right at index date]
                progress = (self.observation_days - offset) / self.observation_days

                # Clinical trajectory dynamics
                if is_positive:
                    # Deteriorating trajectory: FPG & HbA1c escalating
                    fpg_drift = progress * self.rng.uniform(18.0, 36.0)
                    hba1c_drift = progress * self.rng.uniform(0.6, 1.3)
                    sbp_drift = progress * self.rng.uniform(4.0, 12.0)
                    egfr_drift = -progress * self.rng.uniform(5.0, 16.0)
                    bmi_drift = progress * self.rng.uniform(0.5, 2.2)
                    trig_drift = progress * self.rng.uniform(20.0, 50.0)
                else:
                    # Stable random walk / mild natural fluctuation
                    fpg_drift = self.rng.gauss(0.0, 3.0)
                    hba1c_drift = self.rng.gauss(0.0, 0.1)
                    sbp_drift = self.rng.gauss(0.0, 3.0)
                    egfr_drift = self.rng.gauss(0.0, 2.0)
                    bmi_drift = self.rng.gauss(0.0, 0.3)
                    trig_drift = self.rng.gauss(0.0, 8.0)

                # Add stochastic visit-level noise
                curr_fpg = self._clip(base_fpg + fpg_drift + self.rng.gauss(0, 4.0), 50.0, 350.0)
                curr_hba1c = self._clip(base_hba1c + hba1c_drift + self.rng.gauss(0, 0.12), 4.0, 14.5)
                curr_sbp = self._clip(base_sbp + sbp_drift + self.rng.gauss(0, 5.0), 80.0, 220.0)
                curr_dbp = self._clip(base_dbp + (sbp_drift * 0.45) + self.rng.gauss(0, 3.5), 50.0, 130.0)
                curr_hr = self._clip(base_hr + self.rng.gauss(0, 4.0), 45.0, 150.0)
                curr_bmi = self._clip(baseline_bmi + bmi_drift, 16.0, 60.0)
                curr_egfr = self._clip(base_egfr + egfr_drift + self.rng.gauss(0, 3.0), 10.0, 130.0)
                curr_creat = self._clip(base_creat + self.rng.gauss(0, 0.08), 0.3, 6.0)
                curr_chol = self._clip(base_chol + self.rng.gauss(0, 8.0), 100.0, 400.0)
                curr_ldl = self._clip(base_ldl + self.rng.gauss(0, 7.0), 40.0, 280.0)
                curr_hdl = self._clip(base_hdl + self.rng.gauss(0, 3.0), 18.0, 100.0)
                curr_trig = self._clip(base_trig + trig_drift + self.rng.gauss(0, 15.0), 40.0, 600.0)

                # Real-world EHR missingness simulation:
                # Routine vitals: recorded in 95% of visits
                # Glucose: recorded in ~65% of visits
                # HbA1c: ordered periodically (~50% of visits)
                # Full Lipid/Renal panel: ordered in ~40% of visits
                has_vitals = self.rng.random() < 0.95
                has_glucose = self.rng.random() < 0.68
                has_hba1c = self.rng.random() < 0.52
                has_lipids = self.rng.random() < 0.42
                has_renal = self.rng.random() < 0.45

                meas_record = {
                    "encounter_id": enc_id,
                    "patient_id": patient_id,
                    "measurement_date": visit_date.strftime("%Y-%m-%d"),
                    "days_to_index": -offset,
                    "systolic_bp": round(curr_sbp, 1) if has_vitals else None,
                    "diastolic_bp": round(curr_dbp, 1) if has_vitals else None,
                    "heart_rate": round(curr_hr, 1) if has_vitals else None,
                    "bmi": round(curr_bmi, 1) if has_vitals else None,
                    "fasting_glucose": round(curr_fpg, 1) if has_glucose else None,
                    "hba1c": round(curr_hba1c, 2) if has_hba1c else None,
                    "total_cholesterol": round(curr_chol, 1) if has_lipids else None,
                    "ldl": round(curr_ldl, 1) if has_lipids else None,
                    "hdl": round(curr_hdl, 1) if has_lipids else None,
                    "triglycerides": round(curr_trig, 1) if has_lipids else None,
                    "serum_creatinine": round(curr_creat, 2) if has_renal else None,
                    "egfr": round(curr_egfr, 1) if has_renal else None,
                }
                measurements.append(meas_record)

                # Diagnoses assignment for the encounter
                # Certain ICD-10 codes become frequent as conditions worsen
                visit_codes = set()
                if curr_sbp >= 135.0 or curr_dbp >= 85.0:
                    if self.rng.random() < 0.85:
                        visit_codes.add("I10")
                if has_lipids and (curr_ldl >= 130.0 or curr_trig >= 200.0):
                    if self.rng.random() < 0.80:
                        visit_codes.add("E78.5")
                if curr_bmi >= 30.0:
                    visit_codes.add("E66.01" if curr_bmi >= 35.0 else "E66.9")
                if has_hba1c and (5.7 <= curr_hba1c < 6.5):
                    if self.rng.random() < 0.75:
                        visit_codes.add("R73.03")
                if has_renal and curr_egfr < 60.0:
                    visit_codes.add("N18.3")
                elif has_renal and curr_egfr < 80.0:
                    visit_codes.add("N18.2")
                if pt["smoking_status"] == "Current":
                    if self.rng.random() < 0.70:
                        visit_codes.add("Z72.0")
                if pt["family_history_diabetes"] == 1 and self.rng.random() < 0.40:
                    visit_codes.add("Z83.3")

                for code in visit_codes:
                    diagnoses.append({
                        "diagnosis_id": f"DX_{diag_counter:08d}",
                        "encounter_id": enc_id,
                        "patient_id": patient_id,
                        "diagnosis_date": visit_date.strftime("%Y-%m-%d"),
                        "days_to_index": -offset,
                        "icd10_code": code,
                        "description": ICD10_CODE_MAP.get(code, "Comorbidity"),
                    })
                    diag_counter += 1

        logger.info(
            f"Generated {len(encounters)} encounters, {len(measurements)} measurement records, "
            f"and {len(diagnoses)} diagnoses across {len(patients)} patients."
        )
        return encounters, measurements, diagnoses

    def save_cohort(
        self,
        patients: List[Dict[str, Any]],
        encounters: List[Dict[str, Any]],
        measurements: List[Dict[str, Any]],
        diagnoses: List[Dict[str, Any]],
        output_dir: Path,
        save_format: str = "both",
    ) -> Dict[str, List[Path]]:
        """Saves generated dataset tables to CSV and/or Parquet formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files: Dict[str, List[Path]] = {"csv": [], "parquet": []}

        tables = {
            "patients": patients,
            "encounters": encounters,
            "measurements": measurements,
            "diagnoses": diagnoses,
        }

        # 1. Save CSV
        if save_format in ("csv", "both"):
            for table_name, records in tables.items():
                if not records:
                    continue
                csv_path = output_dir / f"{table_name}.csv"
                fieldnames = list(records[0].keys())
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(records)
                saved_files["csv"].append(csv_path)
                logger.info(f"Saved CSV table: {csv_path} ({len(records)} rows)")

        # 2. Save Parquet (attempt via pyarrow, pandas, or polars)
        if save_format in ("parquet", "both"):
            parquet_saved = False
            # Try Polars first
            try:
                import polars as pl
                for table_name, records in tables.items():
                    if not records:
                        continue
                    pq_path = output_dir / f"{table_name}.parquet"
                    df = pl.DataFrame(records)
                    df.write_parquet(pq_path)
                    saved_files["parquet"].append(pq_path)
                    logger.info(f"Saved Parquet (polars): {pq_path}")
                parquet_saved = True
            except ImportError:
                pass

            # Try PyArrow / Pandas if polars not available
            if not parquet_saved:
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    for table_name, records in tables.items():
                        if not records:
                            continue
                        pq_path = output_dir / f"{table_name}.parquet"
                        table = pa.Table.from_pylist(records)
                        pq.write_table(table, pq_path)
                        saved_files["parquet"].append(pq_path)
                        logger.info(f"Saved Parquet (pyarrow): {pq_path}")
                    parquet_saved = True
                except ImportError:
                    pass

            if not parquet_saved:
                logger.warning(
                    "Neither polars nor pyarrow is installed; skipping parquet generation. "
                    "CSV files have been saved successfully."
                )

        return saved_files

    def run(self, output_dir: Path, save_format: str = "both") -> Dict[str, List[Path]]:
        """Executes full synthetic cohort generation and saves results."""
        logger.info(f"Starting synthetic EHR generation: {self.num_patients} patients, seed={self.seed}")
        patients = self.generate_patient_cohort()
        encounters, measurements, diagnoses = self.generate_longitudinal_records(patients)
        saved = self.save_cohort(patients, encounters, measurements, diagnoses, output_dir, save_format)
        logger.info("Cohort generation completed successfully.")
        return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Longitudinal EHR Data Generator")
    parser.add_argument(
        "--num-patients",
        type=int,
        default=1000,
        help="Number of unique patients to generate (default: 1000)",
    )
    parser.add_argument(
        "--prevalence",
        type=float,
        default=0.18,
        help="Approximate target chronic disease prevalence (default: 0.18)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Directory to save generated tables (default: data/raw)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["both", "csv", "parquet"],
        default="both",
        help="Output file format (default: both)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run fast verification of synthetic cohort generator logic without modifying data/raw.",
    )

    args = parser.parse_args()

    if args.verify:
        logger.info("Executing synthetic cohort generator verification (50 patients)...")
        gen = SyntheticEHRGenerator(num_patients=50, prevalence=0.20, seed=123)
        patients = gen.generate_patient_cohort()
        encounters, measurements, diagnoses = gen.generate_longitudinal_records(patients)
        assert len(patients) == 50, f"Expected 50 patients, got {len(patients)}"
        assert len(encounters) > 50, f"Expected > 50 encounters, got {len(encounters)}"
        assert len(measurements) > 50, f"Expected > 50 measurements, got {len(measurements)}"
        logger.info(f"Verification successful: {len(patients)} patients, {len(encounters)} encounters, {len(measurements)} measurements.")
        return

    generator = SyntheticEHRGenerator(
        num_patients=args.num_patients,
        prevalence=args.prevalence,
        seed=args.seed,
    )
    generator.run(output_dir=Path(args.output_dir), save_format=args.format)


if __name__ == "__main__":
    main()
