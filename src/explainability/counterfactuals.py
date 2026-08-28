"""Actionable Clinical Recourse and Counterfactual Optimization for Chronic Disease Risk.

Implements constrained clinical recourse search for high-risk patients:
1. Enforces strict biological plausibility and immutability constraints:
   - Immutable: age, sex, ethnicity, family history, visit history.
   - Mutable downwards (lifestyle & pharmacotherapy): Fasting Glucose, HbA1c, Systolic BP, Diastolic BP, BMI, Smoking, LDL/Cholesterol.
2. Solves constrained recourse optimization to downgrade patient disease risk below target threshold (e.g. <= 0.20).
3. Produces concrete, physician-interpretable lifestyle and pharmacotherapy directives.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass
class ClinicalLever:
    """Definition of an actionable physiological biomarker lever."""

    name: str
    display_name: str
    unit: str
    related_columns: List[str]
    primary_column: str
    min_feasible_value: float
    effort_weight: float
    clinical_guideline: str


# Clinical biomarker action levers with physiological safety floors
CLINICAL_LEVERS: List[ClinicalLever] = [
    ClinicalLever(
        name="fasting_glucose",
        display_name="Fasting Blood Glucose",
        unit="mg/dL",
        related_columns=["fasting_glucose_latest", "fasting_glucose_mean", "fasting_glucose_max"],
        primary_column="fasting_glucose_latest",
        min_feasible_value=85.0,  # Avoid hypoglycemia
        effort_weight=1.0,
        clinical_guideline="Initiate medical nutrition therapy, reduce carbohydrate intake, and consider metformin.",
    ),
    ClinicalLever(
        name="hba1c",
        display_name="Glycated Hemoglobin (HbA1c)",
        unit="%",
        related_columns=["hba1c_latest", "hba1c_mean", "hba1c_max"],
        primary_column="hba1c_latest",
        min_feasible_value=5.2,  # Normal physiological floor
        effort_weight=1.2,
        clinical_guideline="Target HbA1c < 5.7% through sustained lifestyle intervention and glycemic control.",
    ),
    ClinicalLever(
        name="systolic_bp",
        display_name="Systolic Blood Pressure",
        unit="mmHg",
        related_columns=["systolic_bp_latest", "systolic_bp_mean", "systolic_bp_max"],
        primary_column="systolic_bp_latest",
        min_feasible_value=115.0,  # Avoid hypotension
        effort_weight=1.0,
        clinical_guideline="Adopt DASH diet, restrict dietary sodium (<2g/day), and optimize antihypertensive therapy.",
    ),
    ClinicalLever(
        name="diastolic_bp",
        display_name="Diastolic Blood Pressure",
        unit="mmHg",
        related_columns=["diastolic_bp_latest", "diastolic_bp_mean", "diastolic_bp_max"],
        primary_column="diastolic_bp_latest",
        min_feasible_value=75.0,  # Avoid hypotension
        effort_weight=1.0,
        clinical_guideline="Complement systolic BP reduction with regular aerobic exercise and stress management.",
    ),
    ClinicalLever(
        name="baseline_bmi",
        display_name="Body Mass Index (BMI)",
        unit="kg/m²",
        related_columns=["baseline_bmi", "bmi_latest", "bmi_mean", "bmi_max"],
        primary_column="baseline_bmi",
        min_feasible_value=22.0,  # Healthy BMI floor
        effort_weight=1.8,
        clinical_guideline="Target 5-10% body weight reduction via structured caloric deficit and strength training.",
    ),
    ClinicalLever(
        name="smoking_numeric",
        display_name="Smoking Status",
        unit="status (0=None, 1=Former, 2=Current)",
        related_columns=["smoking_numeric"],
        primary_column="smoking_numeric",
        min_feasible_value=0.0,
        effort_weight=1.5,
        clinical_guideline="Complete smoking cessation via nicotine replacement therapy and behavioral counseling.",
    ),
    ClinicalLever(
        name="ldl",
        display_name="LDL Cholesterol",
        unit="mg/dL",
        related_columns=["ldl_latest", "ldl_mean", "ldl_max", "total_cholesterol_latest", "total_cholesterol_mean"],
        primary_column="ldl_latest",
        min_feasible_value=70.0,
        effort_weight=1.1,
        clinical_guideline="Initiate moderate-to-high intensity statin therapy and reduce dietary saturated fats.",
    ),
]


class ClinicalCounterfactualExplainer:
    """Generates optimal, actionable counterfactual recourse for clinical risk reduction."""

    def __init__(
        self,
        model_path: Union[str, Path] = MODELS_DIR / "xgb_baseline.json",
        processed_dir: Union[str, Path] = PROCESSED_DATA_DIR,
    ):
        self.model_path = Path(model_path)
        self.processed_dir = Path(processed_dir)
        self.model: Optional[XGBClassifier] = None
        self.feature_names: List[str] = []
        self._initialize()

    def _initialize(self) -> None:
        """Loads model and feature schema."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        self.model = XGBClassifier()
        self.model.load_model(str(self.model_path))

        train_path = self.processed_dir / "train_tabular.csv"
        if train_path.exists():
            df_train = pd.read_csv(train_path, nrows=5)
            exclude = {"patient_id", "target_disease", "target_label"}
            self.feature_names = [c for c in df_train.columns if c not in exclude]

    def get_patient_probability(self, patient_features: Union[pd.Series, Dict[str, float], np.ndarray]) -> float:
        """Computes chronic disease onset probability for a patient feature vector."""
        if isinstance(patient_features, np.ndarray):
            x = patient_features.reshape(1, -1)
        elif isinstance(patient_features, dict):
            x = np.array([[patient_features.get(f, 0.0) for f in self.feature_names]], dtype=np.float32)
        else:
            x = np.array([[float(patient_features[f]) for f in self.feature_names]], dtype=np.float32)

        prob = float(self.model.predict_proba(x)[0, 1])
        return prob

    def generate_recourse(
        self,
        patient_data: Union[pd.Series, Dict[str, Any]],
        target_risk: float = 0.20,
    ) -> Dict[str, Any]:
        """Finds minimal actionable physiological modifications to achieve target_risk.

        Evaluates single-lever, clinical-package, and joint multi-lever interventions
        using ultra-fast batched inference with biological boundary guarantees.
        """
        if isinstance(patient_data, pd.Series):
            current_dict = patient_data.to_dict()
        else:
            current_dict = dict(patient_data)

        # Build base numpy vector
        x0 = np.array([float(current_dict.get(f, 0.0)) for f in self.feature_names], dtype=np.float32)
        p_initial = float(self.model.predict_proba(x0.reshape(1, -1))[0, 1])

        # If patient is already at or below target risk
        if p_initial <= target_risk:
            return {
                "initial_risk": round(p_initial, 4),
                "target_risk": round(target_risk, 4),
                "counterfactual_risk": round(p_initial, 4),
                "absolute_risk_reduction": 0.0,
                "target_achieved": True,
                "recourse_needed": False,
                "message": f"Patient is already at low risk ({p_initial:.1%}), below target threshold ({target_risk:.1%}).",
                "actions_count": 0,
                "recommended_actions": [],
                "counterfactual_features": current_dict,
            }

        # Identify active mutable levers
        lever_specs = []
        for lever in CLINICAL_LEVERS:
            idxs = [self.feature_names.index(c) for c in lever.related_columns if c in self.feature_names]
            if len(idxs) > 0 and lever.primary_column in self.feature_names:
                prim_idx = self.feature_names.index(lever.primary_column)
                cur_val = float(x0[prim_idx])
                if cur_val > lever.min_feasible_value:
                    lever_specs.append({
                        "lever": lever,
                        "indices": idxs,
                        "prim_idx": prim_idx,
                        "cur_val": cur_val,
                        "floor": lever.min_feasible_value,
                        "diff": cur_val - lever.min_feasible_value,
                        "weight": lever.effort_weight,
                    })

        if not lever_specs:
            return {
                "initial_risk": round(p_initial, 4),
                "target_risk": round(target_risk, 4),
                "counterfactual_risk": round(p_initial, 4),
                "absolute_risk_reduction": 0.0,
                "target_achieved": False,
                "recourse_needed": True,
                "message": "No mutable clinical biomarkers available for intervention.",
                "actions_count": 0,
                "recommended_actions": [],
                "counterfactual_features": current_dict,
            }

        K = len(lever_specs)

        def make_candidate_vector(alphas: np.ndarray) -> np.ndarray:
            cand = x0.copy()
            for spec, a in zip(lever_specs, alphas):
                for idx in spec["indices"]:
                    orig_c = x0[idx]
                    target_c = spec["floor"]
                    if orig_c > target_c:
                        cand[idx] = orig_c - a * (orig_c - target_c)
            return cand

        # Generate comprehensive candidate plans
        candidates = []
        alphas_list = []

        # 1. Single-lever sweeps (can one factor alone resolve risk?)
        for i in range(K):
            for a in np.linspace(0.1, 1.0, 10):
                alpha = np.zeros(K, dtype=np.float32)
                alpha[i] = a
                alphas_list.append(alpha)
                candidates.append(make_candidate_vector(alpha))

        # 2. Joint uniform scaling across all active levers
        for a in np.linspace(0.05, 1.0, 20):
            alpha = np.full(K, a, dtype=np.float32)
            alphas_list.append(alpha)
            candidates.append(make_candidate_vector(alpha))

        # 3. Glycemic control package (Glucose + HbA1c)
        for g in np.linspace(0.2, 1.0, 5):
            for h in np.linspace(0.2, 1.0, 5):
                alpha = np.zeros(K, dtype=np.float32)
                alpha[0] = g
                if K > 1:
                    alpha[1] = h
                alphas_list.append(alpha)
                candidates.append(make_candidate_vector(alpha))

        # 4. Cardiometabolic package (Glycemic + BP)
        for a in np.linspace(0.1, 1.0, 10):
            alpha = np.zeros(K, dtype=np.float32)
            num_cardio = min(4, K)
            alpha[:num_cardio] = a
            alphas_list.append(alpha)
            candidates.append(make_candidate_vector(alpha))

        # 5. Glycemic + BMI weight loss package
        bmi_idx = next((i for i, s in enumerate(lever_specs) if s["lever"].name == "baseline_bmi"), None)
        if bmi_idx is not None:
            for a in np.linspace(0.2, 1.0, 6):
                alpha = np.zeros(K, dtype=np.float32)
                alpha[0] = a
                if K > 1:
                    alpha[1] = a
                alpha[bmi_idx] = a * 0.7  # Moderate weight loss
                alphas_list.append(alpha)
                candidates.append(make_candidate_vector(alpha))

        # Batched inference
        cand_arr = np.array(candidates, dtype=np.float32)
        probs = self.model.predict_proba(cand_arr)[:, 1]

        # Filter candidates meeting the target risk
        target_indices = np.where(probs <= target_risk)[0]

        if len(target_indices) > 0:
            # Objective: minimize weighted clinical effort among target-reaching candidates
            efforts = [
                sum(lever_specs[i]["weight"] * (alphas_list[idx][i] ** 1.5) for i in range(K))
                for idx in target_indices
            ]
            best_idx = target_indices[np.argmin(efforts)]
            best_prob = float(probs[best_idx])
            best_alpha = alphas_list[best_idx].copy()
            target_achieved = True
        else:
            # Best candidate with maximum absolute risk reduction
            best_idx = int(np.argmin(probs))
            best_prob = float(probs[best_idx])
            best_alpha = alphas_list[best_idx].copy()
            target_achieved = False

        # Fine-grained coordinate pruning: try reducing individual alpha coordinates if target remains met
        if target_achieved:
            for i in range(K):
                if best_alpha[i] > 0.05:
                    test_alpha = best_alpha.copy()
                    test_alpha[i] = max(0.0, test_alpha[i] - 0.2)
                    test_cand = make_candidate_vector(test_alpha)
                    test_p = float(self.model.predict_proba(test_cand.reshape(1, -1))[0, 1])
                    if test_p <= target_risk:
                        best_alpha = test_alpha
                        best_prob = test_p

        # Build final counterfactual feature dictionary and recommended actions
        final_cand = make_candidate_vector(best_alpha)
        cf_features = dict(current_dict)
        for idx, feat_name in enumerate(self.feature_names):
            cf_features[feat_name] = float(final_cand[idx])

        recommended_actions = []
        for i, spec in enumerate(lever_specs):
            a = best_alpha[i]
            if a > 0.02:
                orig_val = spec["cur_val"]
                prim_idx = spec["prim_idx"]
                new_val = float(final_cand[prim_idx])
                reduction = orig_val - new_val
                if reduction > 1e-2:
                    recommended_actions.append({
                        "lever_name": spec["lever"].name,
                        "display_name": spec["lever"].display_name,
                        "unit": spec["lever"].unit,
                        "baseline_value": round(orig_val, 2),
                        "target_value": round(new_val, 2),
                        "required_reduction": round(reduction, 2),
                        "reduction_pct": round((reduction / orig_val) * 100.0, 1) if orig_val > 0 else 0.0,
                        "clinical_directive": spec["lever"].clinical_guideline,
                    })

        # Sort actions by highest impact/reduction
        recommended_actions.sort(key=lambda x: x["required_reduction"], reverse=True)

        summary = {
            "initial_risk": round(p_initial, 4),
            "target_risk": round(target_risk, 4),
            "counterfactual_risk": round(best_prob, 4),
            "absolute_risk_reduction": round(p_initial - best_prob, 4),
            "target_achieved": bool(target_achieved),
            "recourse_needed": True,
            "actions_count": len(recommended_actions),
            "recommended_actions": recommended_actions,
            "message": (
                f"Risk successfully reduced from {p_initial:.1%} to {best_prob:.1%} "
                f"({p_initial - best_prob:.1%} absolute reduction)."
                if target_achieved
                else f"Partial recourse achieved: risk reduced from {p_initial:.1%} to {best_prob:.1%}."
            ),
            "counterfactual_features": cf_features,
        }
        return summary


def run_counterfactual_demonstration() -> Dict[str, Any]:
    """Runs counterfactual recourse generation on high-risk test cohort patients."""
    print("=" * 75)
    print("EXECUTING ACTIONABLE CLINICAL RECOURSE (COUNTERFACTUAL OPTIMIZATION)")
    print("=" * 75)

    test_tab_path = PROCESSED_DATA_DIR / "test_tabular.csv"
    if not test_tab_path.exists():
        raise FileNotFoundError("test_tabular.csv not found.")

    df_test = pd.read_csv(test_tab_path)
    explainer = ClinicalCounterfactualExplainer()

    # Find candidate patient with highest predicted risk
    high_risk_pt = None
    highest_prob = 0.0

    for idx in range(min(50, len(df_test))):
        pt_row = df_test.iloc[idx]
        p = explainer.get_patient_probability(pt_row)
        if p > highest_prob:
            highest_prob = p
            high_risk_pt = pt_row

    if high_risk_pt is None:
        high_risk_pt = df_test.iloc[0]
        highest_prob = explainer.get_patient_probability(high_risk_pt)

    pt_id = high_risk_pt.get("patient_id", "TEST_PT_001")
    print(f"\nSelected Candidate Patient: {pt_id}")
    print(f"Initial Predicted Disease Risk: {highest_prob:.1%}")

    # Generate clinical recourse
    recourse = explainer.generate_recourse(high_risk_pt, target_risk=0.20)

    print(f"\nCounterfactual Optimization Result:")
    print(f"  Target Achieved:           {recourse['target_achieved']}")
    print(f"  Final Counterfactual Risk: {recourse['counterfactual_risk']:.1%}")
    print(f"  Absolute Risk Reduction:   {recourse['absolute_risk_reduction']:.1%}")
    print(f"\nPrescribed Actionable Directives ({recourse['actions_count']}):")
    for a in recourse["recommended_actions"]:
        print(f"  - {a['display_name']}: {a['baseline_value']} -> {a['target_value']} {a['unit']} (-{a['required_reduction']} {a['unit']})")
        print(f"    Guideline: {a['clinical_directive']}")

    # Export report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "counterfactual_recourse_demo.json"
    demo_export = {k: v for k, v in recourse.items() if k != "counterfactual_features"}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(demo_export, f, indent=2)
    print(f"\n[Counterfactual] Exported demo recourse report to {report_path}")

    return recourse


if __name__ == "__main__":
    run_counterfactual_demonstration()
