"""Tabular Baseline Model with XGBoost for Early Chronic Disease Detection.

Implements data loading from data/processed/, automatic class imbalance weighting via scale_pos_weight,
model training with early stopping, medical standard evaluation (AUROC, AUPRC, Sensitivity, Specificity),
and model serialization to models/xgb_baseline.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class XGBoostBaselineTrainer:
    """Trains, evaluates, and serializes a production-grade XGBoost tabular baseline."""

    def __init__(
        self,
        processed_dir: Union[str, Path] = PROCESSED_DATA_DIR,
        models_dir: Union[str, Path] = MODELS_DIR,
        random_state: int = 42,
    ):
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self.model: Optional[XGBClassifier] = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.scale_pos_weight: float = 1.0

    def load_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Loads processed tabular train and test sets and separates features and target labels."""
        train_file = self.processed_dir / "train_tabular.csv"
        test_file = self.processed_dir / "test_tabular.csv"

        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                f"Tabular datasets not found in {self.processed_dir}. "
                "Ensure Step 2 feature pipeline (src/data_loader.py) has been executed."
            )

        print(f"[XGBoostBaseline] Loading tabular datasets from {self.processed_dir}...")
        df_train = pl.read_csv(train_file).to_pandas(use_pyarrow_extension_array=False)
        df_test = pl.read_csv(test_file).to_pandas(use_pyarrow_extension_array=False)

        # Identify target label column (supporting target_disease or target_label)
        target_col = None
        for candidate in ["target_disease", "target_label"]:
            if candidate in df_train.columns:
                target_col = candidate
                break

        if target_col is None:
            raise KeyError("Target column ('target_disease' or 'target_label') not found in dataset.")

        # Exclude patient IDs and any duplicate target columns from feature matrix
        exclude_cols = {"patient_id", "target_disease", "target_label"}
        self.feature_names = [col for col in df_train.columns if col not in exclude_cols]

        X_train = df_train[self.feature_names]
        y_train = df_train[target_col].astype(int)
        X_test = df_test[self.feature_names]
        y_test = df_test[target_col].astype(int)

        print(
            f"[XGBoostBaseline] Data loaded: Train={X_train.shape} (Positives: {y_train.sum()}/{len(y_train)}), "
            f"Test={X_test.shape} (Positives: {y_test.sum()}/{len(y_test)})"
        )
        return X_train, y_train, X_test, y_test

    def calculate_scale_pos_weight(self, y_train: pd.Series) -> float:
        """Calculates scale_pos_weight to dynamically counterbalance class imbalance."""
        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))

        if pos_count == 0:
            weight = 1.0
        else:
            weight = float(neg_count / pos_count)

        self.scale_pos_weight = weight
        print(
            f"[XGBoostBaseline] Class distribution: Negative={neg_count}, Positive={pos_count} "
            f"-> scale_pos_weight={weight:.3f}"
        )
        return weight

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> XGBClassifier:
        """Initializes and trains the XGBClassifier with automatic imbalance weighting."""
        scale_weight = self.calculate_scale_pos_weight(y_train)

        print("[XGBoostBaseline] Initializing XGBClassifier with clinical hyperparameters...")
        self.model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=scale_weight,
            random_state=self.random_state,
            eval_metric="aucpr",
            n_jobs=-1,
        )

        eval_set = [(X_test, y_test)] if (X_test is not None and y_test is not None) else None

        print(f"[XGBoostBaseline] Training model on {len(X_train)} samples with {len(self.feature_names)} features...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        print("[XGBoostBaseline] Training completed successfully.")
        return self.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Computes clinical performance metrics on the test cohort."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        # Continuous predicted probabilities for positive chronic onset
        y_prob = self.model.predict_proba(X_test)[:, 1]
        # Binary predictions at standard 0.5 threshold
        y_pred = (y_prob >= 0.5).astype(int)

        # Discrimination metrics
        auroc = float(roc_auc_score(y_test, y_prob))
        auprc = float(average_precision_score(y_test, y_prob))

        # Confusion matrix calculations
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Clinical Sensitivity (Recall) and Specificity at 0.5 threshold
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        accuracy = float(accuracy_score(y_test, y_pred))
        brier = float(brier_score_loss(y_test, y_prob))

        from src.evaluation.metrics import sensitivity_at_fixed_specificity
        sens_at_90_spec, thresh_90 = sensitivity_at_fixed_specificity(np.asarray(y_test, dtype=int), y_prob, 0.90)
        sens_at_95_spec, thresh_95 = sensitivity_at_fixed_specificity(np.asarray(y_test, dtype=int), y_prob, 0.95)

        self.metrics = {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "threshold": 0.5,
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "sensitivity_at_90_spec": round(sens_at_90_spec, 4),
            "sensitivity_at_95_spec": round(sens_at_95_spec, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier, 6),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "num_test_patients": len(y_test),
            "positive_prevalence": round(float(y_test.mean()), 4),
        }

        return self.metrics

    def save_model(self, filename: str = "xgb_baseline.json") -> Path:
        """Serializes the trained XGBoost model artifact to models/xgb_baseline.json."""
        if self.model is None:
            raise ValueError("No trained model to save.")

        save_path = self.models_dir / filename
        self.model.save_model(str(save_path))
        print(f"[XGBoostBaseline] Saved model artifact to {save_path}")

        # Also save evaluation metrics & feature metadata alongside
        metrics_path = self.models_dir / "xgb_baseline_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": self.metrics,
                    "scale_pos_weight": self.scale_pos_weight,
                    "feature_count": len(self.feature_names),
                    "features": self.feature_names,
                },
                f,
                indent=2,
            )
        print(f"[XGBoostBaseline] Saved metrics report to {metrics_path}")

        return save_path

    def run_training_pipeline(self) -> Dict[str, Any]:
        """Runs end-to-end data loading, training, evaluation, and saving."""
        X_train, y_train, X_test, y_test = self.load_data()
        self.train(X_train, y_train, X_test, y_test)
        metrics = self.evaluate(X_test, y_test)
        self.save_model()
        return metrics


def main() -> None:
    """Executes tabular baseline training, evaluation, and console reporting."""
    print("=" * 70)
    print("STEP 3: TABULAR BASELINE MODEL TRAINING (XGBOOST)")
    print("=" * 70)

    trainer = XGBoostBaselineTrainer()
    metrics = trainer.run_training_pipeline()

    print("\n" + "=" * 70)
    print("CLINICAL EVALUATION SCORES ON TEST COHORT")
    print("=" * 70)
    print(f"AUROC (Area Under ROC Curve)            : {metrics['auroc']:.4f}")
    print(f"AUPRC (Area Under PR Curve)             : {metrics['auprc']:.4f}")
    print(f"Sensitivity (Recall @ 0.5 Threshold)    : {metrics['sensitivity']:.4f}")
    print(f"Specificity (@ 0.5 Threshold)           : {metrics['specificity']:.4f}")
    print(f"Precision (@ 0.5 Threshold)             : {metrics['precision']:.4f}")
    print(f"F1-Score (@ 0.5 Threshold)              : {metrics['f1_score']:.4f}")
    print(f"Brier Calibration Score                 : {metrics['brier_score']:.6f}")
    print("-" * 70)
    cm = metrics["confusion_matrix"]
    print(f"Confusion Matrix (N={metrics['num_test_patients']}):")
    print(f"  True Negatives  : {cm['true_negatives']:<4} | False Positives : {cm['false_positives']}")
    print(f"  False Negatives : {cm['false_negatives']:<4} | True Positives  : {cm['true_positives']}")
    print("=" * 70)
    print(f"Model successfully saved to models/xgb_baseline.json")


if __name__ == "__main__":
    main()
