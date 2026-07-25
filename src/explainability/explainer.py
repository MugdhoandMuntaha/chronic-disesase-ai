"""Clinical Model Interpretability and Feature Attribution with SHAP.

Computes global cohort-level feature importance and local patient-level risk attributions
using SHAP TreeExplainer for the trained XGBoost model.
Produces beeswarm plots, bar importance charts, and individual patient waterfall breakdowns.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Headless backend for server and CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap
from xgboost import XGBClassifier

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


class ClinicalSHAPExplainer:
    """Computes and visualizes SHAP values for clinical decision support."""

    def __init__(
        self,
        model_path: Union[str, Path] = MODELS_DIR / "xgb_baseline.json",
        processed_dir: Union[str, Path] = PROCESSED_DATA_DIR,
        reports_dir: Union[str, Path] = REPORTS_DIR,
    ):
        self.model_path = Path(model_path)
        self.processed_dir = Path(processed_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self.X_test: Optional[pd.DataFrame] = None
        self.shap_values: Optional[np.ndarray] = None
        self.base_value: float = 0.0

    def load_model_and_data(self) -> Tuple[XGBClassifier, pd.DataFrame]:
        """Loads trained XGBoost model and processed test cohort features."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Ensure Step 3 model training has been completed."
            )

        print(f"[ClinicalSHAP] Loading model from {self.model_path}...")
        self.model = XGBClassifier()
        self.model.load_model(str(self.model_path))

        test_file = self.processed_dir / "test_tabular.csv"
        if not test_file.exists():
            raise FileNotFoundError(f"Test tabular file not found at {test_file}")

        print(f"[ClinicalSHAP] Loading test features from {test_file}...")
        df_test = pl.read_csv(test_file).to_pandas(use_pyarrow_extension_array=False)

        exclude_cols = {"patient_id", "target_disease", "target_label"}
        self.feature_names = [c for c in df_test.columns if c not in exclude_cols]
        self.X_test = df_test[self.feature_names]

        print(
            f"[ClinicalSHAP] Model and data loaded: {len(self.X_test)} test patients across "
            f"{len(self.feature_names)} features."
        )
        return self.model, self.X_test

    def init_explainer(self) -> shap.TreeExplainer:
        """Initializes SHAP TreeExplainer and computes background baseline."""
        if self.model is None or self.X_test is None:
            self.load_model_and_data()

        print("[ClinicalSHAP] Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

        print(f"[ClinicalSHAP] Computing SHAP values for {len(self.X_test)} test instances...")
        self.shap_values = self.explainer.shap_values(self.X_test)

        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            self.base_value = float(self.explainer.expected_value[1] if len(self.explainer.expected_value) > 1 else self.explainer.expected_value[0])
        else:
            self.base_value = float(self.explainer.expected_value)

        print(f"[ClinicalSHAP] SHAP computation complete. Base expected value: {self.base_value:.4f}")
        return self.explainer

    def get_global_feature_importance(self, top_k: int = 15) -> pd.DataFrame:
        """Computes cohort-level mean absolute SHAP values for global risk drivers."""
        if self.shap_values is None:
            self.init_explainer()

        # Mean absolute SHAP value across all test patients
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)

        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": mean_abs_shap,
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        return importance_df.head(top_k)

    def explain_patient(
        self, patient_idx: int = 0, top_k: int = 8
    ) -> Dict[str, Any]:
        """Provides local feature attribution breakdown for a specific patient.

        Identifies top risk-increasing features (+ SHAP) and protective factors (- SHAP).
        """
        if self.shap_values is None:
            self.init_explainer()

        patient_shap = self.shap_values[patient_idx]
        patient_features = self.X_test.iloc[patient_idx]

        # Top risk drivers (+ SHAP values pushing towards onset)
        risk_increasing = []
        # Top protective factors (- SHAP values lowering risk)
        protective_factors = []

        for feat, val, s_val in zip(self.feature_names, patient_features.values, patient_shap):
            entry = {
                "feature": feat,
                "observed_value": round(float(val), 2),
                "shap_value": round(float(s_val), 4),
            }
            if s_val > 0.001:
                risk_increasing.append(entry)
            elif s_val < -0.001:
                protective_factors.append(entry)

        risk_increasing = sorted(risk_increasing, key=lambda x: x["shap_value"], reverse=True)[:top_k]
        protective_factors = sorted(protective_factors, key=lambda x: x["shap_value"])[:top_k]

        pred_prob = float(self.model.predict_proba(self.X_test.iloc[[patient_idx]])[0, 1])

        return {
            "patient_index": patient_idx,
            "predicted_probability": round(pred_prob, 4),
            "base_expected_value": round(self.base_value, 4),
            "top_risk_drivers": risk_increasing,
            "top_protective_factors": protective_factors,
        }

    def generate_summary_plots(self) -> Dict[str, Path]:
        """Generates and saves publication-quality SHAP beeswarm and bar charts."""
        if self.shap_values is None:
            self.init_explainer()

        saved_plots = {}

        # 1. Global Bar Plot of Top 15 Features
        print("[ClinicalSHAP] Rendering global feature importance bar plot...")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=15,
            show=False,
        )
        plt.title("EndoPredict AI: Top 15 Global Predictive Features (Mean |SHAP|)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        bar_path = self.reports_dir / "shap_importance_bar.png"
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()
        saved_plots["bar_plot"] = bar_path
        print(f"[ClinicalSHAP] Saved {bar_path}")

        # 2. Global Beeswarm Summary Plot
        print("[ClinicalSHAP] Rendering global feature impact beeswarm plot...")
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            feature_names=self.feature_names,
            max_display=15,
            show=False,
        )
        plt.title("EndoPredict AI: Feature Impact on Chronic Disease Onset (SHAP Beeswarm)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        beeswarm_path = self.reports_dir / "shap_summary_beeswarm.png"
        plt.savefig(beeswarm_path, bbox_inches="tight")
        plt.close()
        saved_plots["beeswarm_plot"] = beeswarm_path
        print(f"[ClinicalSHAP] Saved {beeswarm_path}")

        # 3. Individual Patient Waterfall / Decision Plot
        print("[ClinicalSHAP] Rendering sample patient decision plot...")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        explanation = shap.Explanation(
            values=self.shap_values[0],
            base_values=self.base_value,
            data=self.X_test.iloc[0].values,
            feature_names=self.feature_names,
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.title("EndoPredict AI: Patient 0 Local Risk Attribution Breakdown", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        waterfall_path = self.reports_dir / "shap_patient_waterfall.png"
        plt.savefig(waterfall_path, bbox_inches="tight")
        plt.close()
        saved_plots["waterfall_plot"] = waterfall_path
        print(f"[ClinicalSHAP] Saved {waterfall_path}")

        return saved_plots

    def save_importance_report(self) -> Path:
        """Saves ranked feature importance report to JSON."""
        importance_df = self.get_global_feature_importance(top_k=25)
        report_path = self.reports_dir / "global_shap_importance.json"

        report_data = {
            "base_expected_value": self.base_value,
            "num_test_patients": len(self.X_test),
            "top_features": importance_df.to_dict(orient="records"),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        print(f"[ClinicalSHAP] Saved importance report to {report_path}")
        return report_path


def main() -> None:
    """Executes SHAP explainability analysis and prints top clinical risk drivers."""
    print("=" * 75)
    print("STEP 5: CLINICAL MODEL EXPLAINABILITY WITH SHAP")
    print("=" * 75)

    explainer = ClinicalSHAPExplainer()
    explainer.init_explainer()

    print("\n" + "=" * 75)
    print("TOP 15 GLOBAL CHRONIC DISEASE PREDICTIVE DRIVERS (MEAN |SHAP|)")
    print("=" * 75)
    importance = explainer.get_global_feature_importance(top_k=15)

    for rank, row in importance.iterrows():
        print(f"  {rank + 1:02d}. {row['Feature']:<32} | Mean |SHAP|: {row['Mean_Abs_SHAP']:.4f}")

    print("\n" + "=" * 75)
    print("INDIVIDUAL PATIENT ATTRIBUTION BREAKDOWN (PATIENT #0)")
    print("=" * 75)
    pt_explanation = explainer.explain_patient(patient_idx=0, top_k=5)
    print(f"Base Expected Value      : {pt_explanation['base_expected_value']:.4f}")
    print(f"Predicted Onset Risk     : {pt_explanation['predicted_probability']:.2%}")

    print("\n  [Top Risk-Increasing Drivers]:")
    for d in pt_explanation["top_risk_drivers"]:
        print(f"    [+] {d['feature']:<28} = {d['observed_value']:<6} (SHAP: +{d['shap_value']:.4f})")

    print("\n  [Top Protective / Risk-Decreasing Factors]:")
    for p in pt_explanation["top_protective_factors"]:
        print(f"    [-] {p['feature']:<28} = {p['observed_value']:<6} (SHAP: {p['shap_value']:.4f})")


    print("\n" + "=" * 75)
    print("GENERATING EXPLANATION ARTIFACTS AND PLOTS")
    print("=" * 75)
    explainer.generate_summary_plots()
    explainer.save_importance_report()
    print("=" * 75)
    print("SHAP Explainability analysis completed successfully!")


if __name__ == "__main__":
    main()
