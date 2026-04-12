"""Data Cleaning and Feature Pipeline for Chronic Disease Prediction.

Ingests longitudinal EHR data from mock_ehr_data.csv, performs chronological sorting,
categorical encoding, and leakage-free patient-level train/test splitting.
Produces dual representations:
1. Aggregated tabular features (mean, min, max, latest) for XGBoost/LightGBM.
2. 3D Padded sequential tensors and visit masks for PyTorch GRU / RETAIN.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

# Project path configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mock_ehr_data.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Continuous vital and laboratory variables
MEASUREMENT_COLUMNS: List[str] = [
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "bmi",
    "fasting_glucose",
    "hba1c",
    "total_cholesterol",
    "ldl",
    "hdl",
    "triglycerides",
    "serum_creatinine",
    "egfr",
]

# Static demographic and baseline risk features
DEMOGRAPHIC_COLUMNS: List[str] = [
    "age",
    "is_male",
    "baseline_bmi",
    "smoking_numeric",
    "family_history_diabetes",
    "ethnicity_Caucasian",
    "ethnicity_African_American",
    "ethnicity_Hispanic",
    "ethnicity_Asian",
    "ethnicity_Other",
]


class EHRDataLoader:
    """End-to-end data cleaning, splitting, and multi-modal feature builder."""

    def __init__(
        self,
        raw_file: Union[str, Path] = RAW_DATA_PATH,
        processed_dir: Union[str, Path] = PROCESSED_DATA_DIR,
        max_seq_len: int = 15,
        test_size: float = 0.20,
        seed: int = 42,
    ):
        self.raw_file = Path(raw_file)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.max_seq_len = max_seq_len
        self.test_size = test_size
        self.seed = seed

        self.imputation_medians: Dict[str, float] = {}
        self.sequential_feature_names: List[str] = []
        self.tabular_feature_names: List[str] = []

    def ingest_data(self) -> pl.DataFrame:
        """Ingests raw mock EHR data using Polars for high-speed columnar parsing."""
        if not self.raw_file.exists():
            raise FileNotFoundError(
                f"Raw data file not found at {self.raw_file}. "
                "Ensure data/raw/mock_ehr_data.csv has been generated."
            )

        print(f"[EHRDataLoader] Ingesting {self.raw_file} via Polars...")
        df = pl.read_csv(self.raw_file)
        print(f"[EHRDataLoader] Successfully ingested {len(df)} visit rows across {len(df.columns)} columns.")
        return df

    def clean_and_encode_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Sorts visits chronologically per patient and encodes all categorical features."""
        print("[EHRDataLoader] Cleaning data, sorting chronologically, and encoding categoricals...")

        # 1. Chronological sort: by patient_id and days_to_index ascending (e.g. -720 -> -10)
        df_sorted = df.sort(["patient_id", "days_to_index", "encounter_date"])

        # 2. Encode demographics
        df_encoded = df_sorted.with_columns([
            (pl.col("gender") == "Male").cast(pl.Int32).alias("is_male"),
            pl.col("smoking_status")
            .replace_strict({"Never": 0, "Former": 1, "Current": 2}, default=0)
            .cast(pl.Int32)
            .alias("smoking_numeric"),
            (pl.col("ethnicity") == "Caucasian").cast(pl.Int32).alias("ethnicity_Caucasian"),
            (pl.col("ethnicity") == "African American").cast(pl.Int32).alias("ethnicity_African_American"),
            (pl.col("ethnicity") == "Hispanic").cast(pl.Int32).alias("ethnicity_Hispanic"),
            (pl.col("ethnicity") == "Asian").cast(pl.Int32).alias("ethnicity_Asian"),
            (pl.col("ethnicity") == "Other").cast(pl.Int32).alias("ethnicity_Other"),
        ])

        # 3. Encode encounter types
        df_encoded = df_encoded.with_columns([
            (pl.col("encounter_type") == "outpatient").cast(pl.Int32).alias("enc_outpatient"),
            (pl.col("encounter_type") == "telehealth").cast(pl.Int32).alias("enc_telehealth"),
            (pl.col("encounter_type") == "emergency").cast(pl.Int32).alias("enc_emergency"),
            (pl.col("encounter_type") == "inpatient").cast(pl.Int32).alias("enc_inpatient"),
        ])

        return df_encoded

    def split_by_patient_id(
        self, df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str], List[str]]:
        """Performs stratified train/test split on unique patient IDs to eliminate data leakage.

        Returns:
            train_df, test_df, train_patient_ids, test_patient_ids
        """
        print(f"[EHRDataLoader] Performing patient-level stratified split (test_size={self.test_size:.0%})...")

        # Extract distinct patient IDs with their ground-truth chronic onset labels in Polars
        patient_cohort = (
            df.select(["patient_id", "target_label"])
            .unique(subset=["patient_id"])
        )
        patient_ids = np.array(patient_cohort["patient_id"].to_list())
        target_labels = np.array(patient_cohort["target_label"].to_list(), dtype=int)

        train_pts, test_pts = train_test_split(
            patient_ids,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=target_labels,
        )


        train_pt_set = set(train_pts)
        test_pt_set = set(test_pts)

        # Zero-leakage verification assertion
        overlap = train_pt_set.intersection(test_pt_set)
        assert len(overlap) == 0, f"Patient leakage detected! {len(overlap)} patients appear in both splits."

        train_df = df.filter(pl.col("patient_id").is_in(train_pt_set))
        test_df = df.filter(pl.col("patient_id").is_in(test_pt_set))

        print(
            f"[EHRDataLoader] Split completed: {len(train_pts)} train patients ({len(train_df)} visits), "
            f"{len(test_pts)} test patients ({len(test_df)} visits)."
        )
        return train_df, test_df, list(train_pts), list(test_pts)

    def compute_imputation_values(self, train_df: pl.DataFrame) -> None:
        """Calculates cohort medians on training data only to prevent data leakage."""
        self.imputation_medians = {}
        for col in MEASUREMENT_COLUMNS:
            med_val = train_df[col].median()
            self.imputation_medians[col] = float(med_val) if med_val is not None else 0.0

    def prepare_tabular_format(
        self, train_df: pl.DataFrame, test_df: pl.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregates patient visit metrics (mean, min, max, latest values) for XGBoost."""
        print("[EHRDataLoader] Aggregating longitudinal visit metrics for tabular format...")

        def _aggregate_cohort(df: pl.DataFrame) -> pd.DataFrame:
            # 1. Static features per patient
            static_cols = ["patient_id"] + DEMOGRAPHIC_COLUMNS + ["target_label"]
            static_df = df.select(static_cols).unique(subset=["patient_id"])

            # 2. Longitudinal vital & lab aggregations
            aggs = [pl.count("encounter_id").cast(pl.Int32).alias("visit_count")]
            for col in MEASUREMENT_COLUMNS:
                col_valid = pl.col(col).drop_nulls()
                aggs.extend([
                    col_valid.mean().alias(f"{col}_mean"),
                    col_valid.min().alias(f"{col}_min"),
                    col_valid.max().alias(f"{col}_max"),
                    col_valid.last().alias(f"{col}_latest"),
                    (col_valid.last() - col_valid.first()).alias(f"{col}_trajectory_delta"),
                ])

            longitudinal_df = df.group_by("patient_id").agg(aggs)

            # 3. Join static demographics and aggregated time-series metrics
            joined = static_df.join(longitudinal_df, on="patient_id", how="left")
            pdf = joined.to_pandas(use_pyarrow_extension_array=False)


            # 4. Impute missing metrics using training cohort medians
            for col in MEASUREMENT_COLUMNS:
                med = self.imputation_medians.get(col, 0.0)
                pdf[f"{col}_mean"] = pdf[f"{col}_mean"].fillna(med)
                pdf[f"{col}_min"] = pdf[f"{col}_min"].fillna(med)
                pdf[f"{col}_max"] = pdf[f"{col}_max"].fillna(med)
                pdf[f"{col}_latest"] = pdf[f"{col}_latest"].fillna(med)
                pdf[f"{col}_trajectory_delta"] = pdf[f"{col}_trajectory_delta"].fillna(0.0)

            return pdf

        train_tab = _aggregate_cohort(train_df)
        test_tab = _aggregate_cohort(test_df)

        self.tabular_feature_names = [
            c for c in train_tab.columns if c not in ("patient_id", "target_label")
        ]

        # Save tabular outputs to data/processed
        train_tab_path = self.processed_dir / "train_tabular.csv"
        test_tab_path = self.processed_dir / "test_tabular.csv"
        train_tab.to_csv(train_tab_path, index=False)
        test_tab.to_csv(test_tab_path, index=False)

        print(
            f"[EHRDataLoader] Tabular format saved: {train_tab_path.name} "
            f"({train_tab.shape}), {test_tab_path.name} ({test_tab.shape})"
        )
        return train_tab, test_tab

    def prepare_sequential_format(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        train_pts: List[str],
        test_pts: List[str],
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Constructs 3D padded sequence arrays (batch_size, max_seq_len, num_features) and masks for GRU."""
        print(f"[EHRDataLoader] Building 3D sequential arrays with max_seq_len={self.max_seq_len}...")

        # Feature vector for each longitudinal visit step
        visit_features = MEASUREMENT_COLUMNS + [
            "enc_outpatient",
            "enc_telehealth",
            "enc_emergency",
            "enc_inpatient",
            "age",
            "is_male",
            "baseline_bmi",
            "smoking_numeric",
            "family_history_diabetes",
        ]
        self.sequential_feature_names = visit_features
        num_features = len(visit_features)

        def _build_tensors(
            df: pl.DataFrame, pt_ids: List[str]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            pdf = df.to_pandas(use_pyarrow_extension_array=False)
            # Impute missing measurement values using train medians
            for col in MEASUREMENT_COLUMNS:
                pdf[col] = pdf[col].fillna(self.imputation_medians.get(col, 0.0))


            num_patients = len(pt_ids)
            X = np.zeros((num_patients, self.max_seq_len, num_features), dtype=np.float32)
            mask = np.zeros((num_patients, self.max_seq_len), dtype=np.float32)
            y = np.zeros((num_patients,), dtype=np.float32)

            pt_groups = {pid: grp for pid, grp in pdf.groupby("patient_id")}

            for i, pid in enumerate(pt_ids):
                if pid not in pt_groups:
                    continue
                grp = pt_groups[pid]
                y[i] = float(grp["target_label"].iloc[0])

                # Extract visit sequence sorted by time
                visits = grp[visit_features].values
                seq_len = min(len(visits), self.max_seq_len)

                # Store latest sequence up to max_seq_len
                X[i, :seq_len, :] = visits[-seq_len:]
                mask[i, :seq_len] = 1.0

            return X, mask, y

        X_train, mask_train, y_train = _build_tensors(train_df, train_pts)
        X_test, mask_test, y_test = _build_tensors(test_df, test_pts)

        # Save sequential arrays as compressed numpy files
        train_seq_path = self.processed_dir / "train_sequences.npz"
        test_seq_path = self.processed_dir / "test_sequences.npz"

        np.savez_compressed(
            train_seq_path,
            X=X_train,
            mask=mask_train,
            y=y_train,
            patient_ids=np.array(train_pts),
            features=np.array(visit_features),
        )
        np.savez_compressed(
            test_seq_path,
            X=X_test,
            mask=mask_test,
            y=y_test,
            patient_ids=np.array(test_pts),
            features=np.array(visit_features),
        )

        print(
            f"[EHRDataLoader] Sequential format saved: {train_seq_path.name} "
            f"(X: {X_train.shape}, mask: {mask_train.shape}, y: {y_train.shape})"
        )
        print(
            f"[EHRDataLoader] Sequential format saved: {test_seq_path.name} "
            f"(X: {X_test.shape}, mask: {mask_test.shape}, y: {y_test.shape})"
        )
        return X_train, mask_train, y_train, X_test, mask_test, y_test

    def run_pipeline(self) -> Dict[str, Any]:
        """Executes the complete data cleaning, feature engineering, and saving workflow."""
        df_raw = self.ingest_data()
        df_clean = self.clean_and_encode_data(df_raw)
        train_df, test_df, train_pts, test_pts = self.split_by_patient_id(df_clean)

        self.compute_imputation_values(train_df)

        train_tab, test_tab = self.prepare_tabular_format(train_df, test_df)
        X_train_seq, mask_train, y_train_seq, X_test_seq, mask_test, y_test_seq = (
            self.prepare_sequential_format(train_df, test_df, train_pts, test_pts)
        )

        return {
            "tabular": {
                "train_shape": train_tab.shape,
                "test_shape": test_tab.shape,
                "num_features": len(self.tabular_feature_names),
                "features": self.tabular_feature_names,
            },
            "sequential": {
                "X_train_shape": X_train_seq.shape,
                "mask_train_shape": mask_train.shape,
                "y_train_shape": y_train_seq.shape,
                "X_test_shape": X_test_seq.shape,
                "mask_test_shape": mask_test.shape,
                "y_test_shape": y_test_seq.shape,
                "num_features": len(self.sequential_feature_names),
                "features": self.sequential_feature_names,
            },
        }


def main() -> None:
    """Verification execution block demonstrating complete pipeline run and shape inspection."""
    print("=" * 70)
    print("STEP 2: RUNNING DATA CLEANING AND FEATURE PIPELINE")
    print("=" * 70)

    loader = EHRDataLoader()
    results = loader.run_pipeline()

    print("\n" + "=" * 70)
    print("VERIFICATION AND DATA SHAPES REPORT")
    print("=" * 70)
    print(f"Tabular Train Shape (XGBoost) : {results['tabular']['train_shape']}")
    print(f"Tabular Test Shape (XGBoost)  : {results['tabular']['test_shape']}")
    print(f"Tabular Feature Count         : {results['tabular']['num_features']}")
    print("-" * 70)
    print(f"Sequential X_train Shape (GRU): {results['sequential']['X_train_shape']}")
    print(f"Sequential Mask Train Shape   : {results['sequential']['mask_train_shape']}")
    print(f"Sequential y_train Shape      : {results['sequential']['y_train_shape']}")
    print(f"Sequential X_test Shape (GRU) : {results['sequential']['X_test_shape']}")
    print(f"Sequential Mask Test Shape    : {results['sequential']['mask_test_shape']}")
    print(f"Sequential y_test Shape       : {results['sequential']['y_test_shape']}")
    print(f"Sequential Feature Dimension  : {results['sequential']['num_features']}")
    print("=" * 70)
    print("Data cleaning & feature pipeline executed successfully!")


if __name__ == "__main__":
    main()
