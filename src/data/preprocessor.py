"""High-Performance EHR Feature Engineering Pipeline powered by Polars and Pandas.

Extracts static demographics, longitudinal visit dynamics, time-series vital/lab summary
statistics and trajectory slopes, and ICD-10 comorbidity features without data leakage.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from src.utils.config import (
    DEFAULT_RANDOM_SEED,
    ICD10_CODE_MAP,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.utils.logger import setup_logger

logger = setup_logger("ehr_preprocessor")

MEASUREMENT_COLUMNS = [
    "fasting_glucose",
    "hba1c",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "bmi",
    "total_cholesterol",
    "ldl",
    "hdl",
    "triglycerides",
    "serum_creatinine",
    "egfr",
]

TRACKED_ICD_CODES = [
    "I10",
    "E78.5",
    "R73.03",
    "E66.01",
    "E66.9",
    "N18.2",
    "N18.3",
    "Z72.0",
    "Z83.3",
]


class EHRPreprocessor:
    """Transforms raw relational EHR tables into a patient-level feature matrix for modeling."""

    def __init__(self, raw_data_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.feature_columns: List[str] = []
        self.imputation_values: Dict[str, float] = {}

    def load_raw_data_polars(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Loads raw EHR tables using Polars for ultra-fast columnar performance."""
        if (self.raw_data_dir / "patients.parquet").exists():
            logger.info("Loading Parquet tables via Polars...")
            patients = pl.read_parquet(self.raw_data_dir / "patients.parquet")
            encounters = pl.read_parquet(self.raw_data_dir / "encounters.parquet")
            measurements = pl.read_parquet(self.raw_data_dir / "measurements.parquet")
            diagnoses = pl.read_parquet(self.raw_data_dir / "diagnoses.parquet")
        else:
            logger.info("Loading CSV tables via Polars...")
            patients = pl.read_csv(self.raw_data_dir / "patients.csv")
            encounters = pl.read_csv(self.raw_data_dir / "encounters.csv")
            measurements = pl.read_csv(self.raw_data_dir / "measurements.csv")
            diagnoses = pl.read_csv(self.raw_data_dir / "diagnoses.csv")

        return patients, encounters, measurements, diagnoses

    def build_feature_matrix_polars(
        self,
        patients: pl.DataFrame,
        encounters: pl.DataFrame,
        measurements: pl.DataFrame,
        diagnoses: pl.DataFrame,
    ) -> pd.DataFrame:
        """Leverages vectorized Polars aggregations to construct leakage-free feature matrix."""
        logger.info("Extracting static demographics...")
        df_demo = patients.with_columns([
            (pl.col("gender") == "Male").cast(pl.Int32).alias("is_male"),
            pl.col("smoking_status").replace_strict({
                "Never": 0, "Former": 1, "Current": 2
            }, default=0).cast(pl.Int32).alias("smoking_numeric"),

            (pl.col("ethnicity") == "Caucasian").cast(pl.Int32).alias("ethnicity_Caucasian"),
            (pl.col("ethnicity") == "African American").cast(pl.Int32).alias("ethnicity_African_American"),
            (pl.col("ethnicity") == "Hispanic").cast(pl.Int32).alias("ethnicity_Hispanic"),
            (pl.col("ethnicity") == "Asian").cast(pl.Int32).alias("ethnicity_Asian"),
            (pl.col("ethnicity") == "Other").cast(pl.Int32).alias("ethnicity_Other"),
        ])

        demo_cols = [
            "patient_id", "age", "is_male", "baseline_bmi",
            "family_history_diabetes", "smoking_numeric",
            "ethnicity_Caucasian", "ethnicity_African_American",
            "ethnicity_Hispanic", "ethnicity_Asian", "ethnicity_Other",
        ]
        if "target_label" in df_demo.columns:
            demo_cols.append("target_label")
        df_demo = df_demo.select(demo_cols)

        logger.info("Computing encounter dynamics (strict observation window)...")
        enc_valid = encounters.filter(pl.col("days_to_index") <= 0)
        df_enc = enc_valid.group_by("patient_id").agg([
            pl.count("encounter_id").cast(pl.Int32).alias("encounter_count"),
            pl.col("days_to_index").max().abs().cast(pl.Float64).alias("days_since_last_encounter"),
            (pl.col("days_to_index").max() - pl.col("days_to_index").min()).abs().cast(pl.Float64).alias("observation_span_days"),
            (pl.col("encounter_type") == "emergency").sum().cast(pl.Int32).alias("emergency_encounter_count"),
        ])

        logger.info("Aggregating longitudinal measurements & trajectory slopes...")
        meas_valid = measurements.filter(pl.col("days_to_index") <= 0).sort(["patient_id", "days_to_index"])

        meas_aggs = []
        for col in MEASUREMENT_COLUMNS:
            if col in meas_valid.columns:
                valid_col = pl.col(col).drop_nulls()
                meas_aggs.extend([
                    valid_col.last().alias(f"{col}_latest"),
                    valid_col.mean().alias(f"{col}_mean"),
                    valid_col.min().alias(f"{col}_min"),
                    valid_col.max().alias(f"{col}_max"),
                    valid_col.std().alias(f"{col}_std"),
                    (valid_col.last() - valid_col.first()).alias(f"{col}_delta"),
                ])

        df_meas = meas_valid.group_by("patient_id").agg(meas_aggs)

        logger.info("Extracting ICD-10 diagnostic comorbidity indicators...")
        diag_valid = diagnoses.filter(pl.col("days_to_index") <= 0)

        diag_aggs = []
        for code in TRACKED_ICD_CODES:
            diag_aggs.append(
                (pl.col("icd10_code") == code).any().cast(pl.Int32).alias(f"has_{code}")
            )
        diag_aggs.append(pl.col("icd10_code").n_unique().cast(pl.Int32).alias("total_comorbidities"))
        df_diag = diag_valid.group_by("patient_id").agg(diag_aggs)

        # Merge all tables in Polars
        merged = df_demo.join(df_enc, on="patient_id", how="left")
        merged = merged.join(df_meas, on="patient_id", how="left")
        merged = merged.join(df_diag, on="patient_id", how="left")

        # Fill default encounter counts
        merged = merged.with_columns([
            pl.col("encounter_count").fill_null(0),
            pl.col("emergency_encounter_count").fill_null(0),
            pl.col("days_since_last_encounter").fill_null(365.0),
            pl.col("observation_span_days").fill_null(0.0),
            pl.col("total_comorbidities").fill_null(0),
        ])

        for code in TRACKED_ICD_CODES:
            merged = merged.with_columns(pl.col(f"has_{code}").fill_null(0))

        # Convert to Pandas for scikit-learn & downstream modeling
        pdf = merged.to_pandas()
        logger.info(f"Feature matrix assembled successfully: {pdf.shape[0]} patients, {pdf.shape[1]} columns.")
        return pdf

    def fit_transform_imputation(self, df_features: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Applies median imputation on numeric features to handle EHR missingness."""
        df = df_features.copy()
        numeric_cols = [c for c in df.columns if c not in ("patient_id", "target_label")]

        if is_train:
            self.imputation_values = {}
            for col in numeric_cols:
                med = float(df[col].median())
                if np.isnan(med):
                    med = 0.0
                self.imputation_values[col] = med
                df[col] = df[col].fillna(med)
        else:
            for col in numeric_cols:
                val = self.imputation_values.get(col, 0.0)
                df[col] = df[col].fillna(val)

        return df

    def run_pipeline(
        self, test_size: float = 0.20, seed: int = DEFAULT_RANDOM_SEED
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Runs the complete preprocessing and train/test split pipeline."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        pts, enc, meas, diag = self.load_raw_data_polars()
        full_df = self.build_feature_matrix_polars(pts, enc, meas, diag)

        train_df, test_df = train_test_split(
            full_df,
            test_size=test_size,
            random_state=seed,
            stratify=full_df["target_label"] if "target_label" in full_df.columns else None,
        )

        train_df = self.fit_transform_imputation(train_df, is_train=True)
        test_df = self.fit_transform_imputation(test_df, is_train=False)

        self.feature_columns = [c for c in train_df.columns if c not in ("patient_id", "target_label")]

        # Save processed files using CSV and Parquet
        train_df.to_csv(self.processed_dir / "train_features.csv", index=False)
        test_df.to_csv(self.processed_dir / "test_features.csv", index=False)

        try:
            pl.from_pandas(train_df).write_parquet(self.processed_dir / "train_features.parquet")
            pl.from_pandas(test_df).write_parquet(self.processed_dir / "test_features.parquet")
        except Exception:
            pass

        # Save feature metadata
        metadata = {
            "feature_columns": self.feature_columns,
            "imputation_values": self.imputation_values,
            "num_features": len(self.feature_columns),
            "num_train_samples": len(train_df),
            "num_test_samples": len(test_df),
        }
        with open(self.processed_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Preprocessing finished! Processed datasets saved to {self.processed_dir}")
        return train_df, test_df, self.feature_columns


def main() -> None:
    preprocessor = EHRPreprocessor()
    preprocessor.run_pipeline()


if __name__ == "__main__":
    main()
