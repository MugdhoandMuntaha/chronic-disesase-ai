"""Tabular Gradient Boosting Baseline for Chronic Disease Early Detection.

Implements training, clinical evaluation, model artifact serialization, and inference.
Supports XGBoost with fallback to scikit-learn HistGradientBoostingClassifier.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_clinical_model
from src.utils.config import DEFAULT_RANDOM_SEED, MODELS_DIR, PROCESSED_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("tabular_model")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGBOOST = False


class ChronicDiseaseModel:
    """Production-grade tabular classifier for early chronic disease risk prediction."""

    def __init__(
        self,
        model_dir: Path = MODELS_DIR,
        random_state: int = DEFAULT_RANDOM_SEED,
        scale_pos_weight: float = 4.0,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight

        self.model = None
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}

    def init_model(self) -> None:
        """Initializes the gradient boosting classifier."""
        if HAS_XGBOOST:
            logger.info("Initializing XGBoost classifier...")
            self.model = XGBClassifier(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.80,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                eval_metric="aucpr",
                n_jobs=-1,
            )
        else:
            logger.info("Initializing HistGradientBoostingClassifier (scikit-learn fallback)...")
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.04,
                max_depth=4,
                class_weight="balanced",
                random_state=self.random_state,
            )

    def load_processed_data(
        self, processed_dir: Path = PROCESSED_DATA_DIR
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Loads train and test feature matrices from processed directory."""
        import polars as pl

        train_path = processed_dir / "train_features.parquet"
        test_path = processed_dir / "test_features.parquet"

        if train_path.exists():
            df_train = pl.read_parquet(train_path).to_pandas()
            df_test = pl.read_parquet(test_path).to_pandas()
        else:
            df_train = pd.read_csv(processed_dir / "train_features.csv")
            df_test = pd.read_csv(processed_dir / "test_features.csv")


        # Load metadata
        with open(processed_dir / "feature_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.feature_columns = metadata["feature_columns"]

        X_train = df_train[self.feature_columns]
        y_train = df_train["target_label"]
        X_test = df_test[self.feature_columns]
        y_test = df_test["target_label"]

        return X_train, y_train, X_test, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """Fits the model on training data and computes feature importances."""
        if self.model is None:
            self.init_model()

        self.feature_columns = list(X_train.columns)
        logger.info(f"Training model on {len(X_train)} samples across {len(self.feature_columns)} features...")

        if HAS_XGBOOST and X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        # Extract feature importances
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            self.feature_importance = {
                feat: float(imp) for feat, imp in zip(self.feature_columns, importances)
            }
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda item: item[1], reverse=True)
            )

        logger.info("Model training completed successfully.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates model performance against medical standards."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        y_prob = self.model.predict_proba(X_test)[:, 1]
        self.metrics = evaluate_clinical_model(y_test.values, y_prob)

        logger.info(f"=== Clinical Evaluation Results ===")
        logger.info(f"AUROC:                   {self.metrics['auroc']:.4f}")
        logger.info(f"AUPRC:                   {self.metrics['auprc']:.4f}")
        logger.info(f"Sensitivity @ 90% Spec:  {self.metrics['sensitivity_at_90_spec']:.4f} (thresh={self.metrics['threshold_at_90_spec']:.3f})")
        logger.info(f"Sensitivity @ 95% Spec:  {self.metrics['sensitivity_at_95_spec']:.4f} (thresh={self.metrics['threshold_at_95_spec']:.3f})")
        logger.info(f"Brier Calibration Score: {self.metrics['brier_score']:.4f}")
        logger.info(f"Best F1 Score:           {self.metrics['best_f1']:.4f}")

        return self.metrics

    def save(self, filename: str = "chronic_disease_model.joblib") -> Path:
        """Saves trained model, features, and evaluation metrics to disk."""
        save_path = self.model_dir / filename
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }
        joblib.dump(payload, save_path)
        logger.info(f"Saved model package to {save_path}")

        # Save metrics as JSON for easy tracking
        metrics_path = self.model_dir / "evaluation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        return save_path

    def load(self, filename: str = "chronic_disease_model.joblib") -> None:
        """Loads model package from disk."""
        load_path = self.model_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at {load_path}")

        payload = joblib.load(load_path)
        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]
        self.metrics = payload.get("metrics", {})
        self.feature_importance = payload.get("feature_importance", {})
        logger.info(f"Loaded model from {load_path} with {len(self.feature_columns)} features.")

    def predict_risk(self, patient_features: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """Inference endpoint: takes patient features and returns calibrated risk score and tier."""
        if self.model is None:
            raise ValueError("Model must be loaded or trained before making predictions.")

        if isinstance(patient_features, dict):
            # Align features with training column order
            df = pd.DataFrame([{col: patient_features.get(col, 0.0) for col in self.feature_columns}])
        else:
            df = patient_features[self.feature_columns]

        prob = float(self.model.predict_proba(df)[0, 1])

        # Clinical triage tiers
        if prob < 0.20:
            tier = "Low Risk"
            recommendation = "Routine annual wellness visit and standard lifestyle maintenance."
            color = "#28a745"
        elif prob < 0.45:
            tier = "Moderate Risk"
            recommendation = "Lifestyle counseling (nutrition, exercise), repeat metabolic panel in 6 months."
            color = "#ffc107"
        elif prob < 0.70:
            tier = "High Risk"
            recommendation = "Initiate formal diabetes prevention protocol; consider Metformin if BMI >= 35; 3-month HbA1c check."
            color = "#fd7e14"
        else:
            tier = "Critical Risk (Imminent Onset)"
            recommendation = "Urgent clinical diagnostic confirmation, comprehensive renal and cardiovascular assessment."
            color = "#dc3545"

        return {
            "predicted_risk_probability": round(prob, 4),
            "risk_percentage": round(prob * 100, 1),
            "risk_tier": tier,
            "color_code": color,
            "clinical_recommendation": recommendation,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate chronic disease tabular model")
    parser.add_argument("--train", action="store_true", help="Train model on processed data")
    parser.add_argument("--eval", action="store_true", help="Evaluate model on test split")
    args = parser.parse_args()

    model_handler = ChronicDiseaseModel()
    X_train, y_train, X_test, y_test = model_handler.load_processed_data()

    model_handler.train(X_train, y_train, X_val=X_test, y_val=y_test)
    model_handler.evaluate(X_test, y_test)
    model_handler.save()


if __name__ == "__main__":
    main()
