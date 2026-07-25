"""MLflow Experiment Tracking and Model Registry for Chronic Disease Prediction.

Provides unified logging of:
1. Dataset cohort parameters and train/test demographics.
2. Model hyperparameters, clinical metrics (AUROC, AUPRC, Sens@90%Spec, Brier score).
3. Model binaries, SHAP beeswarm/waterfall plots, and comparative diagnostic artifacts.
4. Champion model selection and tagging for production deployment.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd

import os

# Project directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class ChronicDiseaseExperimentTracker:
    """Orchestrates MLflow tracking experiments across Tabular and Sequence models."""

    def __init__(
        self,
        experiment_name: str = "chronic_disease_early_detection",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        # Use sqlite backend for production-grade tracking in MLflow 3+
        db_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
        self.tracking_uri = tracking_uri or db_uri

        MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        print(f"[MLflowTracker] Tracking URI set to: {self.tracking_uri}")
        print(f"[MLflowTracker] Active Experiment: {self.experiment_name}")

    def log_dataset_overview(self) -> str:
        """Logs dataset cohort statistics and clinical feature representations."""
        with mlflow.start_run(run_name="cohort_metadata_and_features") as run:
            train_tab_path = PROCESSED_DATA_DIR / "train_tabular.csv"
            test_tab_path = PROCESSED_DATA_DIR / "test_tabular.csv"

            if train_tab_path.exists() and test_tab_path.exists():
                df_train = pd.read_csv(train_tab_path)
                df_test = pd.read_csv(test_tab_path)

                target_col = "target_label" if "target_label" in df_train.columns else "target_disease"
                total_n = len(df_train) + len(df_test)
                train_pos = int(df_train[target_col].sum())
                test_pos = int(df_test[target_col].sum())
                prevalence = (train_pos + test_pos) / total_n

                params = {
                    "total_patients": total_n,
                    "train_patients": len(df_train),
                    "test_patients": len(df_test),
                    "split_strategy": "stratified_patient_id_zero_leakage",
                    "num_tabular_features": len(df_train.columns) - 2,
                    "overall_prevalence": round(prevalence, 4),
                    "train_prevalence": round(train_pos / len(df_train), 4),
                    "test_prevalence": round(test_pos / len(df_test), 4),
                }
                mlflow.log_params(params)
                mlflow.log_metric("prevalence_pct", prevalence * 100.0)
                mlflow.set_tag("pipeline_stage", "data_engineering")

                print(f"[MLflowTracker] Logged Cohort Metadata (Run ID: {run.info.run_id})")
                return run.info.run_id
        return ""

    def log_xgboost_run(
        self,
        metrics_path: Union[str, Path] = MODELS_DIR / "xgb_baseline_metrics.json",
        model_path: Union[str, Path] = MODELS_DIR / "xgb_baseline.json",
    ) -> Optional[str]:
        """Logs XGBoost tabular baseline training run, metrics, and SHAP artifacts."""
        metrics_path = Path(metrics_path)
        if not metrics_path.exists():
            print(f"[MLflowTracker] Warning: {metrics_path} not found. Skipping XGBoost log.")
            return None

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        config = data.get("model_config", {})

        with mlflow.start_run(run_name="xgboost_tabular_baseline") as run:
            # Hyperparameters
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("modality", "Tabular Aggregations")
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)

            # Clinical metrics
            metric_keys = [
                "auroc",
                "auprc",
                "sensitivity",
                "specificity",
                "sensitivity_at_90_specificity",
                "brier_score",
                "accuracy",
                "f1",
                "precision",
            ]
            for mk in metric_keys:
                if mk in metrics and metrics[mk] is not None:
                    mlflow.log_metric(mk, float(metrics[mk]))

            mlflow.set_tag("model_family", "gradient_boosting")
            mlflow.set_tag("pipeline_stage", "modeling_tabular")

            # Log artifacts
            model_path = Path(model_path)
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="model_binary")

            if metrics_path.exists():
                mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

            # SHAP plots
            shap_plots = [
                REPORTS_DIR / "shap_summary_beeswarm.png",
                REPORTS_DIR / "shap_importance_bar.png",
                REPORTS_DIR / "shap_patient_waterfall.png",
                REPORTS_DIR / "global_shap_importance.json",
            ]
            for sp in shap_plots:
                if sp.exists():
                    mlflow.log_artifact(str(sp), artifact_path="explainability")

            print(f"[MLflowTracker] Logged XGBoost Baseline (Run ID: {run.info.run_id}, AUROC: {metrics.get('auroc', 'N/A')})")
            return run.info.run_id

    def log_gru_sequence_run(
        self,
        metrics_path: Union[str, Path] = MODELS_DIR / "gru_sequence_metrics.json",
        model_path: Union[str, Path] = MODELS_DIR / "gru_sequence_model.pt",
    ) -> Optional[str]:
        """Logs PyTorch Bidirectional GRU + Attention sequence model run."""
        metrics_path = Path(metrics_path)
        if not metrics_path.exists():
            print(f"[MLflowTracker] Warning: {metrics_path} not found. Skipping GRU log.")
            return None

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        with mlflow.start_run(run_name="pytorch_gru_temporal_attention") as run:
            mlflow.log_param("model_type", "PyTorch GRU")
            mlflow.log_param("modality", "Longitudinal 3D Sequence")
            mlflow.log_param("hidden_dim", 64)
            mlflow.log_param("num_layers", 2)
            mlflow.log_param("bidirectional", True)
            mlflow.log_param("attention_type", "Masked Additive Temporal Attention")

            metric_keys = [
                "auroc",
                "auprc",
                "sensitivity",
                "specificity",
                "sensitivity_at_90_spec",
                "sensitivity_at_90_specificity",
                "brier_score",
                "accuracy",
                "f1_score",
                "precision",
            ]
            for mk in metric_keys:
                if mk in metrics and metrics[mk] is not None:
                    canonical_key = "sensitivity_at_90_specificity" if "90" in mk else mk
                    mlflow.log_metric(canonical_key, float(metrics[mk]))

            mlflow.set_tag("model_family", "deep_recurrent")
            mlflow.set_tag("pipeline_stage", "modeling_sequence")

            model_path = Path(model_path)
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="model_checkpoint")

            if metrics_path.exists():
                mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

            print(f"[MLflowTracker] Logged PyTorch GRU (Run ID: {run.info.run_id}, AUROC: {metrics.get('auroc', 'N/A')})")
            return run.info.run_id

    def log_retain_run(
        self,
        metrics_path: Union[str, Path] = MODELS_DIR / "retain_sequence_metrics.json",
        model_path: Union[str, Path] = MODELS_DIR / "retain_sequence_model.pt",
    ) -> Optional[str]:
        """Logs PyTorch RETAIN (Reverse Time Attention) sequence model run."""
        metrics_path = Path(metrics_path)
        if not metrics_path.exists():
            print(f"[MLflowTracker] Warning: {metrics_path} not found. Skipping RETAIN log.")
            return None

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        with mlflow.start_run(run_name="pytorch_retain_reverse_time_attention") as run:
            mlflow.log_param("model_type", "PyTorch RETAIN")
            mlflow.log_param("modality", "Longitudinal 3D Sequence")
            mlflow.log_param("embed_dim", 64)
            mlflow.log_param("hidden_dim", 64)
            mlflow.log_param("attention_mechanism", "Dual Reverse-Time (alpha visit, beta variable)")

            metric_keys = [
                "auroc",
                "auprc",
                "sensitivity",
                "specificity",
                "sensitivity_at_90_specificity",
                "brier_score",
                "accuracy",
                "f1_score",
                "precision",
            ]
            for mk in metric_keys:
                if mk in metrics and metrics[mk] is not None:
                    mlflow.log_metric(mk, float(metrics[mk]))

            mlflow.set_tag("model_family", "dual_attention_interpretable")
            mlflow.set_tag("pipeline_stage", "modeling_sequence")

            model_path = Path(model_path)
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="model_checkpoint")

            if metrics_path.exists():
                mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

            print(f"[MLflowTracker] Logged PyTorch RETAIN (Run ID: {run.info.run_id}, AUROC: {metrics.get('auroc', 'N/A')})")
            return run.info.run_id

    def generate_leaderboard(self) -> pd.DataFrame:
        """Generates a side-by-side clinical benchmark comparison dataframe."""
        models_data = []

        # 1. XGBoost
        xgb_path = MODELS_DIR / "xgb_baseline_metrics.json"
        if xgb_path.exists():
            with open(xgb_path, "r", encoding="utf-8") as f:
                d = json.load(f).get("metrics", {})
                models_data.append({
                    "Model": "XGBoost Tabular Baseline",
                    "Input Modality": "Engineered Aggregations (73)",
                    "AUROC": d.get("auroc", 0.0),
                    "AUPRC": d.get("auprc", 0.0),
                    "Sensitivity @ 0.5": d.get("sensitivity", 0.0),
                    "Specificity @ 0.5": d.get("specificity", 0.0),
                    "Sens @ 90% Spec": d.get("sensitivity_at_90_spec", d.get("sensitivity_at_90_specificity", 0.0)),
                    "Brier Score": d.get("brier_score", 0.0),
                    "Clinical Interpretability": "Global SHAP + Patient Waterfall",
                })

        # 2. GRU
        gru_path = MODELS_DIR / "gru_sequence_metrics.json"
        if gru_path.exists():
            with open(gru_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                models_data.append({
                    "Model": "PyTorch GRU + Attention",
                    "Input Modality": "Longitudinal Tensors (15, 21)",
                    "AUROC": d.get("auroc", 0.0),
                    "AUPRC": d.get("auprc", 0.0),
                    "Sensitivity @ 0.5": d.get("sensitivity", 0.0),
                    "Specificity @ 0.5": d.get("specificity", 0.0),
                    "Sens @ 90% Spec": d.get("sensitivity_at_90_spec", d.get("sensitivity_at_90_specificity", 0.0)),
                    "Brier Score": d.get("brier_score", 0.0),
                    "Clinical Interpretability": "Temporal Visit Attention (alpha)",
                })

        # 3. RETAIN
        retain_path = MODELS_DIR / "retain_sequence_metrics.json"
        if retain_path.exists():
            with open(retain_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                models_data.append({
                    "Model": "PyTorch RETAIN",
                    "Input Modality": "Longitudinal Tensors (15, 21)",
                    "AUROC": d.get("auroc", 0.0),
                    "AUPRC": d.get("auprc", 0.0),
                    "Sensitivity @ 0.5": d.get("sensitivity", 0.0),
                    "Specificity @ 0.5": d.get("specificity", 0.0),
                    "Sens @ 90% Spec": d.get("sensitivity_at_90_specificity", 0.0),
                    "Brier Score": d.get("brier_score", 0.0),
                    "Clinical Interpretability": "Dual Attention (alpha visit + beta variable)",
                })

        df = pd.DataFrame(models_data)
        return df


def run_experiment_tracking_pipeline() -> pd.DataFrame:
    """Executes the complete MLflow logging pipeline across all models."""
    print("=" * 75)
    print("EXECUTING CLINICAL MLFLOW EXPERIMENT TRACKING PIPELINE")
    print("=" * 75)

    tracker = ChronicDiseaseExperimentTracker()

    # Log dataset overview
    tracker.log_dataset_overview()

    # Log individual models
    tracker.log_xgboost_run()
    tracker.log_gru_sequence_run()
    tracker.log_retain_run()

    # Generate benchmark leaderboard
    leaderboard = tracker.generate_leaderboard()
    print("\n" + "=" * 75)
    print("CLINICAL LEADERBOARD & BENCHMARK COMPARISON")
    print("=" * 75)
    print(leaderboard.to_string(index=False))

    return leaderboard


if __name__ == "__main__":
    run_experiment_tracking_pipeline()
