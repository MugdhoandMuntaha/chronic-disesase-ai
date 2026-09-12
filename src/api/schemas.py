"""Pydantic Request and Response Schemas for Clinical Prediction and Explainability APIs."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PatientTabularRequest(BaseModel):
    """Input payload for tabular chronic disease risk scoring (XGBoost)."""

    patient_id: Optional[str] = Field("PT_CUSTOM", description="Unique patient identifier")
    age: float = Field(..., ge=18.0, le=110.0, description="Age in years", examples=[58.0])
    is_male: int = Field(..., ge=0, le=1, description="Gender (1 for Male, 0 for Female)", examples=[1])
    baseline_bmi: float = Field(..., ge=12.0, le=70.0, description="Baseline Body Mass Index (kg/m²)", examples=[31.4])
    smoking_numeric: int = Field(0, ge=0, le=2, description="Smoking status: 0=Never, 1=Former, 2=Current", examples=[1])
    family_history_diabetes: int = Field(0, ge=0, le=1, description="Family history of diabetes (1=Yes, 0=No)", examples=[1])

    # Glycemic profile
    fasting_glucose_latest: float = Field(..., ge=40.0, le=450.0, description="Latest Fasting Glucose (mg/dL)", examples=[122.0])
    fasting_glucose_mean: Optional[float] = Field(None, description="Mean Fasting Glucose over observation window")
    fasting_glucose_delta: Optional[float] = Field(0.0, description="Trajectory drift (latest - earliest glucose)", examples=[18.0])
    hba1c_latest: float = Field(..., ge=3.5, le=18.0, description="Latest HbA1c (%)", examples=[6.3])
    hba1c_mean: Optional[float] = Field(None, description="Mean HbA1c over observation window")
    hba1c_delta: Optional[float] = Field(0.0, description="Trajectory drift (latest - earliest HbA1c)", examples=[0.6])

    # Cardiovascular & Renal Profile
    systolic_bp_latest: float = Field(..., ge=60.0, le=250.0, description="Latest Systolic Blood Pressure (mmHg)", examples=[138.0])
    systolic_bp_mean: Optional[float] = Field(None, description="Mean Systolic Blood Pressure over observation window")
    diastolic_bp_latest: float = Field(..., ge=40.0, le=150.0, description="Latest Diastolic Blood Pressure (mmHg)", examples=[86.0])
    diastolic_bp_mean: Optional[float] = Field(None, description="Mean Diastolic Blood Pressure over observation window")
    egfr_latest: Optional[float] = Field(85.0, ge=5.0, le=150.0, description="Estimated GFR (mL/min/1.73m²)", examples=[78.0])
    triglycerides_latest: Optional[float] = Field(150.0, ge=30.0, le=800.0, description="Triglycerides (mg/dL)", examples=[210.0])

    # Comorbidity indicators (ICD-10)
    has_I10: int = Field(0, ge=0, le=1, description="Essential Hypertension diagnosed (1=Yes, 0=No)", examples=[1])
    has_R73_03: int = Field(0, ge=0, le=1, alias="has_R73.03", description="Prediabetes diagnosed (1=Yes, 0=No)", examples=[1])
    has_E78_5: int = Field(0, ge=0, le=1, alias="has_E78.5", description="Hyperlipidemia diagnosed (1=Yes, 0=No)", examples=[1])

    model_config = {
        "populate_by_name": True,
    }


class LongitudinalVisitRecord(BaseModel):
    """Single clinical encounter record in a longitudinal history."""

    days_to_index: int = Field(..., le=0, description="Days before index date (e.g. -365)", examples=[-180])
    encounter_type: str = Field("outpatient", description="Encounter type: outpatient, telehealth, emergency, inpatient")
    systolic_bp: float = Field(..., ge=60.0, le=250.0, description="Systolic BP (mmHg)", examples=[135.0])
    diastolic_bp: float = Field(..., ge=40.0, le=150.0, description="Diastolic BP (mmHg)", examples=[84.0])
    heart_rate: Optional[float] = Field(72.0, description="Heart rate (bpm)", examples=[76.0])
    bmi: Optional[float] = Field(30.0, description="Current BMI (kg/m²)", examples=[31.0])
    fasting_glucose: Optional[float] = Field(110.0, description="Fasting plasma glucose (mg/dL)", examples=[118.0])
    hba1c: Optional[float] = Field(5.9, description="Hemoglobin A1c (%)", examples=[6.1])
    triglycerides: Optional[float] = Field(175.0, description="Triglycerides (mg/dL)", examples=[195.0])
    egfr: Optional[float] = Field(88.0, description="Estimated GFR (mL/min/1.73m²)", examples=[82.0])


class PatientSequenceRequest(BaseModel):
    """Input payload for sequential deep learning inference (PyTorch GRU with Attention)."""

    patient_id: Optional[str] = Field("PT_SEQ_CUSTOM", description="Patient identifier")
    age: float = Field(..., ge=18.0, le=110.0, description="Age in years", examples=[54.0])
    is_male: int = Field(..., ge=0, le=1, description="Gender (1=Male, 0=Female)", examples=[1])
    baseline_bmi: float = Field(..., ge=12.0, le=70.0, description="Baseline BMI", examples=[30.5])
    smoking_numeric: int = Field(0, ge=0, le=2, description="Smoking: 0=Never, 1=Former, 2=Current", examples=[1])
    family_history_diabetes: int = Field(0, ge=0, le=1, description="Family history of diabetes", examples=[1])
    visits: List[LongitudinalVisitRecord] = Field(..., min_length=1, max_length=50, description="Longitudinal clinical encounters")


class RiskPredictionResponse(BaseModel):
    """Standardized clinical prediction output schema."""

    patient_id: str
    predicted_risk_probability: float
    risk_percentage: float
    risk_tier: str
    triage_color: str
    model_version: str
    clinical_recommendation: str
    actionable_next_steps: List[str]


class VisitAttentionScore(BaseModel):
    """Clinical encounter attention weight and driving feature decomposition."""

    visit_index: int
    days_to_index: Optional[int] = None
    visit_attention_alpha: float
    top_driving_features: List[Dict[str, Any]] = []


class RetainPredictionResponse(BaseModel):
    """Prediction output schema for PyTorch RETAIN model with dual attention interpretability."""

    patient_id: str
    predicted_risk_probability: float
    risk_percentage: float
    risk_tier: str
    triage_color: str
    model_version: str
    clinical_recommendation: str
    actionable_next_steps: List[str]
    visit_attributions: List[VisitAttentionScore]


class FeatureImpact(BaseModel):
    """Individual feature contribution entry from SHAP."""

    feature: str
    observed_value: Any
    shap_value: float
    impact_direction: str


class ExplainResponse(BaseModel):
    """SHAP explanation response with top risk drivers and protective factors."""

    patient_id: str
    predicted_risk_probability: float
    cohort_base_value: float
    top_risk_drivers: List[FeatureImpact]
    top_protective_factors: List[FeatureImpact]


class HealthResponse(BaseModel):
    """API health status and model readiness report."""

    status: str
    xgboost_tabular_model_loaded: bool
    pytorch_sequence_gru_loaded: bool
    pytorch_retain_loaded: bool
    shap_explainer_ready: bool
    mlflow_tracking_enabled: bool
    timestamp: str


class FeatureDriftReport(BaseModel):
    """Drift metrics for an individual clinical feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    is_statistically_significant: bool
    drift_tier: str
    status_color: str
    ref_mean: float
    curr_mean: float


class CohortDriftResponse(BaseModel):
    """Cohort-level distribution drift summary report."""

    cohort_safety_code: str
    cohort_status: str
    max_psi: float
    drifted_features_count: int
    drifted_features: List[str]
    monitored_samples: int
    features: List[FeatureDriftReport]


class ConformalPredictionResponse(BaseModel):
    """Conformal prediction set and uncertainty quantification output."""

    patient_id: str
    predicted_risk_probability: float
    target_coverage_guarantee: float
    prediction_set: List[int]
    prediction_labels: List[str]
    is_ambiguous: bool
    requires_physician_review: bool
    clinical_directive: str


class CounterfactualRecourseRequest(BaseModel):
    """Request payload for actionable counterfactual recourse optimization."""

    patient_id: Optional[str] = Field("PT_000019", description="Patient identifier in test cohort")
    target_risk: float = Field(0.20, ge=0.05, le=0.50, description="Target probability for risk downgrade")
    patient_features: Optional[Dict[str, float]] = Field(None, description="Optional raw feature dictionary")


class RecommendedRecourseAction(BaseModel):
    """Prescribed biomarker modification directive."""

    lever_name: str
    display_name: str
    unit: str
    baseline_value: float
    target_value: float
    required_reduction: float
    reduction_pct: float
    clinical_directive: str


class CounterfactualRecourseResponse(BaseModel):
    """Optimal actionable clinical recourse response."""

    patient_id: str
    initial_risk: float
    target_risk: float
    counterfactual_risk: float
    absolute_risk_reduction: float
    target_achieved: bool
    recourse_needed: bool
    actions_count: int
    recommended_actions: List[RecommendedRecourseAction]
    message: str


class FairnessAuditResponse(BaseModel):
    """Demographic equity and algorithmic fairness audit report."""

    audit_title: str
    decision_threshold: float
    overall_cohort: Dict[str, Any]
    biological_sex_audit: Dict[str, Any]
    age_bracket_audit: Dict[str, Any]
    racial_ethnic_audit: Dict[str, Any]
    regulatory_summary: Dict[str, Any]

