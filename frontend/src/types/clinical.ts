export interface PatientProfile {
  patient_id: string;
  name: string;
  age: number;
  is_male: number;
  baseline_bmi: number;
  smoking_numeric: number;
  family_history_diabetes: number;
  ethnicity: string;
  fasting_glucose_latest: number;
  fasting_glucose_mean: number;
  hba1c_latest: number;
  hba1c_mean: number;
  systolic_bp_latest: number;
  systolic_bp_mean: number;
  diastolic_bp_latest: number;
  diastolic_bp_mean: number;
  heart_rate_latest: number;
  total_cholesterol_latest: number;
  ldl_latest: number;
  hdl_latest: number;
  triglycerides_latest: number;
  serum_creatinine_latest: number;
  egfr_latest: number;
  clinical_summary: string;
  risk_label: string;
}

export interface RiskPrediction {
  patient_id: string;
  predicted_risk_probability: number;
  risk_tier: "Low Risk" | "Moderate Risk" | "High Risk" | "Critical Risk (Imminent Onset)";
  triage_color: string;
  clinical_action: string;
  guideline_recommendations: string[];
  model_used: string;
}

export interface ConformalPrediction {
  patient_id: string;
  predicted_risk_probability: number;
  target_coverage_guarantee: number;
  prediction_set: number[];
  prediction_labels: string[];
  is_ambiguous: boolean;
  requires_physician_review: boolean;
  clinical_directive: string;
}

export interface RecommendedAction {
  feature_name: string;
  display_name: string;
  baseline_value: number;
  target_value: number;
  required_reduction: number;
  unit: string;
  clinical_directive: string;
}

export interface CounterfactualRecourse {
  patient_id: string;
  initial_risk: number;
  target_risk: number;
  counterfactual_risk: number;
  absolute_risk_reduction: number;
  target_achieved: boolean;
  recourse_needed: boolean;
  actions_count: number;
  recommended_actions: RecommendedAction[];
  message: string;
}

export interface RetainVisitRecord {
  encounter_id: string;
  days_to_index: number;
  encounter_date: string;
  visit_weight: number; // alpha_t
  top_biomarker: string;
  biomarker_weight: number; // beta_t magnitude
  fasting_glucose?: number;
  hba1c?: number;
  systolic_bp?: number;
  diastolic_bp?: number;
  egfr?: number;
}

export interface ClinicalLeaderboardEntry {
  model: string;
  modality: string;
  auroc: number;
  auprc: number;
  sensitivity_50: number;
  specificity_50: number;
  sens_at_90_spec: number;
  brier_score: number;
  interpretability: string;
}
