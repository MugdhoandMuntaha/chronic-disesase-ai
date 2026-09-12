"""Production FastAPI Service for Early Chronic Disease Detection & Explainability.

Endpoints:
- GET  /health           : Model loading and health status
- POST /predict/tabular  : Fast inference with XGBoost tabular baseline
- POST /predict/sequence : Longitudinal sequence inference with PyTorch GRU + Attention
- POST /explain          : Local patient SHAP risk attribution breakdown
- GET  /metrics          : Side-by-side medical validation benchmarks
"""

from contextlib import asynccontextmanager
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CohortDriftResponse,
    ConformalPredictionResponse,
    CounterfactualRecourseRequest,
    CounterfactualRecourseResponse,
    ExplainResponse,
    FairnessAuditResponse,
    FeatureDriftReport,
    FeatureImpact,
    HealthResponse,
    PatientSequenceRequest,
    PatientTabularRequest,
    RecommendedRecourseAction,
    RetainPredictionResponse,
    RiskPredictionResponse,
    VisitAttentionScore,
)
from src.evaluation.conformal import ClinicalConformalPredictor
from src.evaluation.fairness import ClinicalFairnessAuditor
from src.explainability.counterfactuals import ClinicalCounterfactualExplainer
from src.explainability.explainer import ClinicalSHAPExplainer
from src.models.baseline_xgb import XGBoostBaselineTrainer
from src.models.retain_model import RETAINSequenceTrainer
from src.models.sequence_model import EHRSequenceGRU, SequenceModelTrainer

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_clinical_triage(prob: float) -> Tuple[str, str, str, List[str]]:
    """Calculates risk tier, triage color code, and guideline recommendations."""
    if prob < 0.20:
        return (
            "Low Risk",
            "#10b981",
            "Maintain standard annual preventive wellness visits and lifestyle maintenance.",
            ["Annual wellness checkup", "Standard lipid and metabolic screening", "Dietary lifestyle maintenance"],
        )
    elif prob < 0.45:
        return (
            "Moderate Risk",
            "#f59e0b",
            "Prescribe structured nutritional counseling and repeat metabolic panel in 6 months.",
            ["6-month follow-up metabolic panel", "Nutritional and physical activity counseling", "Home blood pressure monitoring"],
        )
    elif prob < 0.70:
        return (
            "High Risk",
            "#f97316",
            "Initiate intensive Diabetes Prevention Program (DPP); consider Metformin therapy if BMI >= 35; recheck HbA1c in 3 months.",
            ["3-month HbA1c re-evaluation", "Enrollment in Diabetes Prevention Program", "Comprehensive cardiovascular risk assessment", "Consider Metformin evaluation"],
        )
    else:
        return (
            "Critical Risk (Imminent Onset)",
            "#ef4444",
            "Urgent diagnostic confirmation; initiate clinical pharmacotherapy; schedule immediate renal and cardiovascular assessment.",
            ["Urgent diagnostic confirmation visit", "Immediate pharmacological intervention", "Renal (eGFR/uACR) and ophthalmology referrals", "Continuous glucose monitoring initiation"],
        )


# Lifespan context manager to preload models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and explainers into memory on service startup."""
    print("[API Lifespan] Initializing clinical models into memory...")

    # 1. Tabular XGBoost model
    try:
        xgb_trainer = XGBoostBaselineTrainer()
        xgb_trainer.load_data()
        xgb_model_path = MODELS_DIR / "xgb_baseline.json"
        if xgb_model_path.exists():
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(str(xgb_model_path))
            xgb_trainer.model = model
            app.state.xgb_trainer = xgb_trainer
            print("[API Lifespan] XGBoost tabular baseline loaded successfully.")
    except Exception as e:
        print(f"[API Lifespan] Warning: XGBoost baseline could not be loaded: {e}")
        app.state.xgb_trainer = None

    # 2. PyTorch Sequence GRU model
    try:
        seq_trainer = SequenceModelTrainer()
        seq_model_path = MODELS_DIR / "gru_sequence_model.pt"
        if seq_model_path.exists():
            seq_trainer.load_model(seq_model_path.name)
            app.state.seq_trainer = seq_trainer
            print("[API Lifespan] PyTorch Sequence GRU loaded successfully.")
    except Exception as e:
        print(f"[API Lifespan] Warning: PyTorch Sequence GRU could not be loaded: {e}")
        app.state.seq_trainer = None

    # 3. PyTorch RETAIN model
    try:
        retain_model_path = MODELS_DIR / "retain_sequence_model.pt"
        if retain_model_path.exists():
            retain_trainer = RETAINSequenceTrainer.load_checkpoint(retain_model_path)
            app.state.retain_trainer = retain_trainer
            print("[API Lifespan] PyTorch RETAIN model loaded successfully.")
        else:
            app.state.retain_trainer = None
    except Exception as e:
        print(f"[API Lifespan] Warning: PyTorch RETAIN model could not be loaded: {e}")
        app.state.retain_trainer = None

    # 4. SHAP Explainer
    try:
        shap_explainer = ClinicalSHAPExplainer()
        shap_explainer.init_explainer()
        app.state.shap_explainer = shap_explainer
        print("[API Lifespan] SHAP TreeExplainer initialized successfully.")
    except Exception as e:
        print(f"[API Lifespan] Warning: SHAP explainer could not be initialized: {e}")
        app.state.shap_explainer = None

    # 5. Conformal Predictor
    try:
        conformal_path = REPORTS_DIR / "conformal_calibration_report.json"
        conformal_predictor = ClinicalConformalPredictor(alpha=0.10)
        if conformal_path.exists():
            with open(conformal_path, "r", encoding="utf-8") as f:
                conf_data = json.load(f)
                conformal_predictor.q_hat = conf_data.get("conformal_threshold_q_hat", 0.05)
        else:
            conformal_predictor.q_hat = 0.05
        app.state.conformal_predictor = conformal_predictor
        print(f"[API Lifespan] Conformal Predictor loaded (q_hat={conformal_predictor.q_hat}).")
    except Exception as e:
        print(f"[API Lifespan] Warning: Conformal predictor failed to init: {e}")
        app.state.conformal_predictor = None

    # 6. Counterfactual Explainer
    try:
        cf_explainer = ClinicalCounterfactualExplainer()
        app.state.cf_explainer = cf_explainer
        print("[API Lifespan] Counterfactual Recourse Explainer loaded.")
    except Exception as e:
        print(f"[API Lifespan] Warning: Counterfactual explainer failed to init: {e}")
        app.state.cf_explainer = None

    yield
    print("[API Lifespan] Shutting down clinical microservice.")


app = FastAPI(
    title="EndoPredict AI | Clinical Decision Support API",
    description="Production REST microservice for early chronic disease prediction, trajectory modeling, and SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root_endpoint() -> Dict[str, str]:
    """Welcome endpoint with interactive documentation links."""
    return {
        "service": "EndoPredict AI - Chronic Disease Early Detection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Returns microservice health and model loading readiness."""
    xgb_ok = hasattr(app.state, "xgb_trainer") and app.state.xgb_trainer is not None
    seq_ok = hasattr(app.state, "seq_trainer") and app.state.seq_trainer is not None
    retain_ok = hasattr(app.state, "retain_trainer") and app.state.retain_trainer is not None
    shap_ok = hasattr(app.state, "shap_explainer") and app.state.shap_explainer is not None
    mlflow_db = PROJECT_ROOT / "mlflow.db"
    mlflow_ok = mlflow_db.exists()

    return HealthResponse(
        status="healthy" if (xgb_ok or seq_ok or retain_ok) else "degraded",
        xgboost_tabular_model_loaded=xgb_ok,
        pytorch_sequence_gru_loaded=seq_ok,
        pytorch_retain_loaded=retain_ok,
        shap_explainer_ready=shap_ok,
        mlflow_tracking_enabled=mlflow_ok,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts input to float, falling back to default if None or unparseable."""
    if val is None:
        return float(default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return float(default)


@app.post("/predict/tabular", response_model=RiskPredictionResponse, tags=["Prediction"])
def predict_tabular(patient: PatientTabularRequest) -> RiskPredictionResponse:
    """Predicts 1-year chronic disease onset risk using the XGBoost tabular model."""
    xgb_trainer = getattr(app.state, "xgb_trainer", None)
    if xgb_trainer is None or xgb_trainer.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="XGBoost model is not loaded. Train the model first via Step 3.",
        )

    # Convert request to feature dictionary aligned with training columns
    req_dict = patient.model_dump(by_alias=True)

    # Populate defaults for missing mean values if user only provided latest
    if req_dict.get("fasting_glucose_mean") is None:
        req_dict["fasting_glucose_mean"] = req_dict["fasting_glucose_latest"] - (safe_float(req_dict.get("fasting_glucose_delta"), 0.0) / 2)
    if req_dict.get("hba1c_mean") is None:
        req_dict["hba1c_mean"] = req_dict["hba1c_latest"] - (safe_float(req_dict.get("hba1c_delta"), 0.0) / 2)
    if req_dict.get("systolic_bp_mean") is None:
        req_dict["systolic_bp_mean"] = req_dict["systolic_bp_latest"] - 4.0
    if req_dict.get("diastolic_bp_mean") is None:
        req_dict["diastolic_bp_mean"] = req_dict["diastolic_bp_latest"] - 2.0

    # Build input row matching model feature columns
    feature_row = {}
    for col in xgb_trainer.feature_names:
        feature_row[col] = safe_float(req_dict.get(col), 0.0)

    df_input = pd.DataFrame([feature_row])
    prob = float(xgb_trainer.model.predict_proba(df_input)[0, 1])


    tier, color, rec, steps = get_clinical_triage(prob)

    return RiskPredictionResponse(
        patient_id=patient.patient_id or "PT_UNKNOWN",
        predicted_risk_probability=round(prob, 4),
        risk_percentage=round(prob * 100, 1),
        risk_tier=tier,
        triage_color=color,
        model_version="XGBoost-Tabular-v1.0",
        clinical_recommendation=rec,
        actionable_next_steps=steps,
    )


@app.post("/predict/sequence", response_model=RiskPredictionResponse, tags=["Prediction"])
def predict_sequence(payload: PatientSequenceRequest) -> RiskPredictionResponse:
    """Predicts 1-year chronic disease onset risk using the PyTorch Sequence GRU with Attention."""
    seq_trainer = getattr(app.state, "seq_trainer", None)
    if seq_trainer is None or seq_trainer.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PyTorch Sequence GRU model is not loaded. Train the model first via Step 4.",
        )

    # Sort visits chronologically
    sorted_visits = sorted(payload.visits, key=lambda v: v.days_to_index)

    max_len = 15
    feature_dim = len(seq_trainer.feature_names)
    X_raw = np.zeros((1, max_len, feature_dim), dtype=np.float32)
    mask = np.zeros((1, max_len), dtype=np.float32)

    seq_len = min(len(sorted_visits), max_len)
    active_visits = sorted_visits[-seq_len:]

    for t, visit in enumerate(active_visits):
        v_dict = visit.model_dump()
        # Add static demographics into visit feature vector
        v_dict["age"] = payload.age
        v_dict["is_male"] = payload.is_male
        v_dict["baseline_bmi"] = payload.baseline_bmi
        v_dict["smoking_numeric"] = payload.smoking_numeric
        v_dict["family_history_diabetes"] = payload.family_history_diabetes
        v_dict["enc_outpatient"] = 1 if visit.encounter_type == "outpatient" else 0
        v_dict["enc_telehealth"] = 1 if visit.encounter_type == "telehealth" else 0
        v_dict["enc_emergency"] = 1 if visit.encounter_type == "emergency" else 0
        v_dict["enc_inpatient"] = 1 if visit.encounter_type == "inpatient" else 0

        for d_idx, feat_name in enumerate(seq_trainer.feature_names):
            X_raw[0, t, d_idx] = float(v_dict.get(feat_name, 0.0))

        mask[0, t] = 1.0

    # Apply z-score normalization
    X_norm = np.zeros_like(X_raw)
    valid_mask = mask.astype(bool)
    X_norm[valid_mask] = (X_raw[valid_mask] - seq_trainer.mean) / seq_trainer.std

    # Model inference
    t_X = torch.tensor(X_norm, dtype=torch.float32).to(seq_trainer.device)
    t_mask = torch.tensor(mask, dtype=torch.float32).to(seq_trainer.device)

    probs, _ = seq_trainer.model.predict_proba(t_X, t_mask)
    prob = float(probs.cpu().numpy()[0, 0])

    tier, color, rec, steps = get_clinical_triage(prob)

    return RiskPredictionResponse(
        patient_id=payload.patient_id or "PT_SEQ_UNKNOWN",
        predicted_risk_probability=round(prob, 4),
        risk_percentage=round(prob * 100, 1),
        risk_tier=tier,
        triage_color=color,
        model_version="PyTorch-Sequence-GRU-v1.0",
        clinical_recommendation=rec,
        actionable_next_steps=steps,
    )


@app.post("/predict/retain", response_model=RetainPredictionResponse, tags=["Prediction"])
def predict_retain(payload: PatientSequenceRequest) -> RetainPredictionResponse:
    """Predicts 1-year chronic disease risk using PyTorch RETAIN with dual visit and variable attention."""
    retain_trainer = getattr(app.state, "retain_trainer", None)
    if retain_trainer is None or retain_trainer.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PyTorch RETAIN model is not loaded. Train the model first via src.models.retain_model.",
        )

    sorted_visits = sorted(payload.visits, key=lambda v: v.days_to_index)
    max_len = 15
    feature_dim = len(retain_trainer.feature_names) if retain_trainer.feature_names else 21
    X_raw = np.zeros((1, max_len, feature_dim), dtype=np.float32)
    mask = np.zeros((1, max_len), dtype=np.float32)

    seq_len = min(len(sorted_visits), max_len)
    active_visits = sorted_visits[-seq_len:]

    feature_names = retain_trainer.feature_names or [
        "age", "is_male", "baseline_bmi", "smoking_numeric", "family_history_diabetes",
        "enc_outpatient", "enc_telehealth", "enc_emergency", "enc_inpatient",
        "systolic_bp", "diastolic_bp", "heart_rate", "bmi", "fasting_glucose",
        "hba1c", "total_cholesterol", "ldl", "hdl", "triglycerides", "serum_creatinine", "egfr"
    ]

    for t, visit in enumerate(active_visits):
        v_dict = visit.model_dump()
        v_dict["age"] = payload.age
        v_dict["is_male"] = payload.is_male
        v_dict["baseline_bmi"] = payload.baseline_bmi
        v_dict["smoking_numeric"] = payload.smoking_numeric
        v_dict["family_history_diabetes"] = payload.family_history_diabetes
        v_dict["enc_outpatient"] = 1 if visit.encounter_type == "outpatient" else 0
        v_dict["enc_telehealth"] = 1 if visit.encounter_type == "telehealth" else 0
        v_dict["enc_emergency"] = 1 if visit.encounter_type == "emergency" else 0
        v_dict["enc_inpatient"] = 1 if visit.encounter_type == "inpatient" else 0

        for d_idx, feat_name in enumerate(feature_names):
            if d_idx < feature_dim:
                X_raw[0, t, d_idx] = float(v_dict.get(feat_name, 0.0))

        mask[0, t] = 1.0

    X_norm = np.zeros_like(X_raw)
    if retain_trainer.norm_mean is not None and retain_trainer.norm_std is not None:
        valid_mask = mask.astype(bool)
        X_norm[valid_mask] = (X_raw[valid_mask] - retain_trainer.norm_mean) / retain_trainer.norm_std
    else:
        X_norm = X_raw

    t_X = torch.tensor(X_norm, dtype=torch.float32).to(retain_trainer.device)
    t_mask = torch.tensor(mask, dtype=torch.float32).to(retain_trainer.device)

    with torch.no_grad():
        contributions, alpha_t, _ = retain_trainer.model.decompose_contributions(t_X, t_mask)
        logits, _, _ = retain_trainer.model(t_X, t_mask)
        prob = float(torch.sigmoid(logits).detach().cpu().numpy()[0, 0])

        alpha_np = alpha_t.detach().cpu().numpy()[0]
        contrib_np = contributions.detach().cpu().numpy()[0]

    visit_scores = []
    for t_idx, visit in enumerate(active_visits):
        v_alpha = float(alpha_np[t_idx])
        v_contribs = contrib_np[t_idx]

        feat_rankings = []
        for f_idx, f_name in enumerate(feature_names):
            if f_idx < feature_dim:
                score = float(v_contribs[f_idx])
                feat_rankings.append({
                    "feature": f_name,
                    "contribution_score": round(score, 5),
                    "impact": "elevates_risk" if score > 0 else "protective"
                })
        feat_rankings.sort(key=lambda x: abs(x["contribution_score"]), reverse=True)

        visit_scores.append(
            VisitAttentionScore(
                visit_index=t_idx + 1,
                days_to_index=visit.days_to_index,
                visit_attention_alpha=round(v_alpha, 4),
                top_driving_features=feat_rankings[:3],
            )
        )

    tier, color, rec, steps = get_clinical_triage(prob)

    return RetainPredictionResponse(
        patient_id=payload.patient_id or "PT_RETAIN_UNKNOWN",
        predicted_risk_probability=round(prob, 4),
        risk_percentage=round(prob * 100, 1),
        risk_tier=tier,
        triage_color=color,
        model_version="PyTorch-RETAIN-v1.0",
        clinical_recommendation=rec,
        actionable_next_steps=steps,
        visit_attributions=visit_scores,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_patient(patient: PatientTabularRequest) -> ExplainResponse:
    """Computes patient-level SHAP values identifying top positive risk drivers and protective factors."""
    shap_explainer = getattr(app.state, "shap_explainer", None)
    xgb_trainer = getattr(app.state, "xgb_trainer", None)

    if shap_explainer is None or shap_explainer.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer is not loaded. Run Step 5 explainability pipeline first.",
        )

    req_dict = patient.model_dump(by_alias=True)

    if req_dict.get("fasting_glucose_mean") is None:
        req_dict["fasting_glucose_mean"] = req_dict["fasting_glucose_latest"] - (safe_float(req_dict.get("fasting_glucose_delta"), 0.0) / 2)
    if req_dict.get("hba1c_mean") is None:
        req_dict["hba1c_mean"] = req_dict["hba1c_latest"] - (safe_float(req_dict.get("hba1c_delta"), 0.0) / 2)
    if req_dict.get("systolic_bp_mean") is None:
        req_dict["systolic_bp_mean"] = req_dict["systolic_bp_latest"] - 4.0
    if req_dict.get("diastolic_bp_mean") is None:
        req_dict["diastolic_bp_mean"] = req_dict["diastolic_bp_latest"] - 2.0

    feature_row = {col: safe_float(req_dict.get(col), 0.0) for col in shap_explainer.feature_names}
    df_patient = pd.DataFrame([feature_row])


    # Compute single-instance SHAP values
    shap_vals = shap_explainer.explainer.shap_values(df_patient)[0]
    base_val = shap_explainer.base_value

    risk_drivers: List[FeatureImpact] = []
    protective_factors: List[FeatureImpact] = []

    for feat, val, s_val in zip(shap_explainer.feature_names, df_patient.iloc[0].values, shap_vals):
        impact = FeatureImpact(
            feature=feat,
            observed_value=round(float(val), 2),
            shap_value=round(float(s_val), 4),
            impact_direction="Increases Risk (+)" if s_val > 0 else "Protective (-)",
        )
        if s_val > 0.001:
            risk_drivers.append(impact)
        elif s_val < -0.001:
            protective_factors.append(impact)

    risk_drivers = sorted(risk_drivers, key=lambda x: x.shap_value, reverse=True)[:6]
    protective_factors = sorted(protective_factors, key=lambda x: x.shap_value)[:6]

    prob = float(xgb_trainer.model.predict_proba(df_patient)[0, 1]) if xgb_trainer else 0.5

    return ExplainResponse(
        patient_id=patient.patient_id or "PT_CUSTOM",
        predicted_risk_probability=round(prob, 4),
        cohort_base_value=round(base_val, 4),
        top_risk_drivers=risk_drivers,
        top_protective_factors=protective_factors,
    )


@app.get("/metrics", tags=["Evaluation"])
def get_benchmarks() -> Dict[str, Any]:
    """Returns side-by-side clinical benchmark metrics for XGBoost, GRU, and RETAIN models."""
    xgb_path = MODELS_DIR / "xgb_baseline_metrics.json"
    gru_path = MODELS_DIR / "gru_sequence_metrics.json"
    retain_path = MODELS_DIR / "retain_sequence_metrics.json"

    result: Dict[str, Any] = {"status": "success", "models": {}}

    if xgb_path.exists():
        with open(xgb_path, "r", encoding="utf-8") as f:
            result["models"]["xgboost_tabular"] = json.load(f)

    if gru_path.exists():
        with open(gru_path, "r", encoding="utf-8") as f:
            result["models"]["pytorch_sequence_gru"] = json.load(f)

    if retain_path.exists():
        with open(retain_path, "r", encoding="utf-8") as f:
            result["models"]["pytorch_retain"] = json.load(f)

    return result


@app.get("/dca", tags=["Evaluation"])
def get_decision_curve_analysis() -> Dict[str, Any]:
    """Returns Decision Curve Analysis (DCA) Net Benefit metrics across threshold probabilities."""
    reports_dir = PROJECT_ROOT / "reports"
    dca_file = reports_dir / "decision_curve_metrics.json"

    if dca_file.exists():
        with open(dca_file, "r", encoding="utf-8") as f:
            return json.load(f)

    from src.evaluation.dca import ClinicalDecisionCurveAnalyzer
    analyzer = ClinicalDecisionCurveAnalyzer()
    return analyzer.run_analysis()


@app.get("/monitoring/drift", response_model=CohortDriftResponse, tags=["Monitoring"])
def get_clinical_drift_report() -> CohortDriftResponse:
    """Returns latest clinical distribution shift and PSI report evaluated against training baseline."""
    reports_dir = PROJECT_ROOT / "reports"
    drift_file = reports_dir / "clinical_drift_report.json"

    if drift_file.exists():
        with open(drift_file, "r", encoding="utf-8") as f:
            return CohortDriftResponse(**json.load(f))

    from src.monitoring.drift_detector import ClinicalDriftDetector
    detector = ClinicalDriftDetector()
    test_path = PROCESSED_DATA_DIR / "test_tabular.csv"
    if not test_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reference data not found. Run preprocessing first.",
        )
    df_test = pd.read_csv(test_path)
    report = detector.evaluate_cohort(df_test)
    detector.export_report(report)
    return CohortDriftResponse(**report)


@app.post("/predict/conformal", response_model=ConformalPredictionResponse, tags=["Uncertainty"])
def predict_with_conformal_guarantee(payload: PatientTabularRequest) -> ConformalPredictionResponse:
    """Returns continuous risk prediction alongside distribution-free Conformal Prediction Set C(x).

    Coverage guarantee >= 90%. Flags ambiguous borderline patients where C(x) = {0, 1} for mandatory physician review.
    """
    if getattr(app.state, "xgb_trainer", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tabular model not loaded.",
        )

    trainer: XGBoostBaselineTrainer = app.state.xgb_trainer
    patient_dict = payload.model_dump(by_alias=True)

    feature_row = {}
    for feat in trainer.feature_names:
        if feat in patient_dict and patient_dict[feat] is not None:
            feature_row[feat] = patient_dict[feat]
        else:
            feature_row[feat] = 0.0

    df_patient = pd.DataFrame([feature_row])[trainer.feature_names]
    prob = float(trainer.model.predict_proba(df_patient)[0, 1])

    predictor: Optional[ClinicalConformalPredictor] = getattr(app.state, "conformal_predictor", None)
    if predictor is None or predictor.q_hat is None:
        predictor = ClinicalConformalPredictor(alpha=0.10)
        predictor.q_hat = 0.05

    pred_set = predictor.predict_set(prob)
    is_ambiguous = len(pred_set) == 2

    label_map = {0: "Low Risk (Negative)", 1: "High Risk (Positive)"}
    labels = [label_map[s] for s in pred_set]

    if is_ambiguous:
        directive = (
            "MANDATORY CLINICAL REVIEW: Borderline risk profile. Conformal prediction set spans both "
            "negative and positive outcomes at 90% confidence. Order comprehensive diagnostic panel."
        )
    elif pred_set == [1]:
        directive = (
            "CONFIRMED HIGH-RISK ONSET: High statistical confidence at 90% coverage. Initiate guideline-directed medical therapy immediately."
        )
    else:
        directive = (
            "CONFIRMED LOW-RISK CONTROL: High statistical confidence at 90% coverage. Patient safely managed with routine annual wellness care."
        )

    return ConformalPredictionResponse(
        patient_id=payload.patient_id or "PT_CUSTOM",
        predicted_risk_probability=round(prob, 4),
        target_coverage_guarantee=0.90,
        prediction_set=pred_set,
        prediction_labels=labels,
        is_ambiguous=is_ambiguous,
        requires_physician_review=is_ambiguous,
        clinical_directive=directive,
    )


@app.post("/explain/counterfactual", response_model=CounterfactualRecourseResponse, tags=["Explainability"])
def generate_counterfactual_recourse(payload: CounterfactualRecourseRequest) -> CounterfactualRecourseResponse:
    """Computes actionable clinical recourse: minimal biomarker shifts to achieve target risk level."""
    cf_explainer: Optional[ClinicalCounterfactualExplainer] = getattr(app.state, "cf_explainer", None)
    if cf_explainer is None:
        cf_explainer = ClinicalCounterfactualExplainer()
        app.state.cf_explainer = cf_explainer

    # Resolve patient features
    if payload.patient_features:
        patient_data = payload.patient_features
        patient_id = payload.patient_id or "PT_CUSTOM"
    elif payload.patient_id:
        test_path = PROCESSED_DATA_DIR / "test_tabular.csv"
        if not test_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test cohort not found.",
            )
        df_test = pd.read_csv(test_path)
        match = df_test[df_test["patient_id"] == payload.patient_id]
        if match.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient {payload.patient_id} not found in test cohort.",
            )
        patient_data = match.iloc[0].to_dict()
        patient_id = payload.patient_id
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either patient_id or patient_features must be provided.",
        )

    recourse = cf_explainer.generate_recourse(patient_data, target_risk=payload.target_risk)
    recourse_actions = [
        RecommendedRecourseAction(**a) for a in recourse["recommended_actions"]
    ]

    return CounterfactualRecourseResponse(
        patient_id=patient_id,
        initial_risk=recourse["initial_risk"],
        target_risk=recourse["target_risk"],
        counterfactual_risk=recourse["counterfactual_risk"],
        absolute_risk_reduction=recourse["absolute_risk_reduction"],
        target_achieved=recourse["target_achieved"],
        recourse_needed=recourse["recourse_needed"],
        actions_count=recourse["actions_count"],
        recommended_actions=recourse_actions,
        message=recourse["message"],
    )


@app.get("/fairness", response_model=FairnessAuditResponse, tags=["Evaluation"])
def get_demographic_fairness_audit() -> FairnessAuditResponse:
    """Returns disaggregated demographic fairness audit across Biological Sex, Age, and Ethnicity."""
    report_path = REPORTS_DIR / "fairness_audit_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return FairnessAuditResponse(**json.load(f))

    auditor = ClinicalFairnessAuditor()
    report = auditor.run_full_audit()
    return FairnessAuditResponse(**report)


