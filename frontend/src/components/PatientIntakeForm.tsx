import React, { useState } from "react";
import { PatientProfile } from "../types/clinical";
import { supabase } from "../lib/supabase";
import { ActivityIcon, CheckCircleIcon, UserIcon, ShieldCheckIcon } from "./Icons";

interface PatientIntakeFormProps {
  onPatientEvaluated: (patient: PatientProfile) => void;
}

export function PatientIntakeForm({ onPatientEvaluated }: PatientIntakeFormProps) {
  const [formData, setFormData] = useState<PatientProfile>({
    patient_id: `PT_CUSTOM_${Math.floor(1000 + Math.random() * 9000)}`,
    name: "Jane Doe",
    age: 55,
    is_male: 0,
    baseline_bmi: 31.5,
    smoking_numeric: 0,
    family_history_diabetes: 1,
    ethnicity: "Caucasian",
    fasting_glucose_latest: 128.0,
    fasting_glucose_mean: 124.0,
    hba1c_latest: 6.4,
    hba1c_mean: 6.2,
    systolic_bp_latest: 138.0,
    systolic_bp_mean: 135.0,
    diastolic_bp_latest: 86.0,
    diastolic_bp_mean: 84.0,
    heart_rate_latest: 74.0,
    total_cholesterol_latest: 215.0,
    ldl_latest: 138.0,
    hdl_latest: 45.0,
    triglycerides_latest: 180.0,
    serum_creatinine_latest: 1.10,
    egfr_latest: 72.0,
    clinical_summary: "Custom clinical patient assessment entered by clinician.",
    risk_label: "Custom Assessment",
  });

  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  const handleChange = (field: keyof PatientProfile, value: any) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPatientEvaluated(formData);
  };

  const handleSaveToSupabase = async () => {
    setSaveStatus("Saving to Supabase...");
    try {
      const { error } = await supabase.from("patients").insert([
        {
          patient_id: formData.patient_id,
          age: formData.age,
          gender: formData.is_male ? "Male" : "Female",
          ethnicity: formData.ethnicity,
          smoking_status: formData.smoking_numeric ? "Current" : "Never",
          baseline_bmi: formData.baseline_bmi,
          family_history_diabetes: formData.family_history_diabetes,
          index_date: new Date().toISOString().split("T")[0],
          target_label: 0,
        },
      ]);

      if (error) {
        setSaveStatus(`Notice: ${error.message}`);
      } else {
        setSaveStatus("Successfully saved to Supabase patients table!");
      }
    } catch (err: any) {
      setSaveStatus(`Notice: Could not connect to Supabase: ${err?.message || err}`);
    }
  };

  const loadPreset = (type: "high" | "borderline" | "low") => {
    if (type === "high") {
      setFormData({
        patient_id: `PT_CUSTOM_${Math.floor(1000 + Math.random() * 9000)}`,
        name: "Robert Taylor",
        age: 64,
        is_male: 1,
        baseline_bmi: 35.8,
        smoking_numeric: 1,
        family_history_diabetes: 1,
        ethnicity: "Caucasian",
        fasting_glucose_latest: 144.0,
        fasting_glucose_mean: 138.0,
        hba1c_latest: 7.1,
        hba1c_mean: 6.8,
        systolic_bp_latest: 148.0,
        systolic_bp_mean: 142.0,
        diastolic_bp_latest: 92.0,
        diastolic_bp_mean: 88.0,
        heart_rate_latest: 82.0,
        total_cholesterol_latest: 245.0,
        ldl_latest: 162.0,
        hdl_latest: 36.0,
        triglycerides_latest: 230.0,
        serum_creatinine_latest: 1.32,
        egfr_latest: 58.0,
        clinical_summary: "64yo male with uncontrolled glycated hemoglobin, severe hypertension, and early nephropathy.",
        risk_label: "Custom High Risk",
      });
    } else if (type === "borderline") {
      setFormData({
        patient_id: `PT_CUSTOM_${Math.floor(1000 + Math.random() * 9000)}`,
        name: "David Kim",
        age: 51,
        is_male: 1,
        baseline_bmi: 27.6,
        smoking_numeric: 0,
        family_history_diabetes: 1,
        ethnicity: "Asian",
        fasting_glucose_latest: 108.0,
        fasting_glucose_mean: 104.0,
        hba1c_latest: 5.85,
        hba1c_mean: 5.80,
        systolic_bp_latest: 132.0,
        systolic_bp_mean: 128.0,
        diastolic_bp_latest: 82.0,
        diastolic_bp_mean: 80.0,
        heart_rate_latest: 72.0,
        total_cholesterol_latest: 198.0,
        ldl_latest: 118.0,
        hdl_latest: 48.0,
        triglycerides_latest: 155.0,
        serum_creatinine_latest: 0.95,
        egfr_latest: 88.0,
        clinical_summary: "51yo male with prediabetic impaired fasting glucose and stage 1 borderline hypertension.",
        risk_label: "Custom Borderline",
      });
    } else {
      setFormData({
        patient_id: `PT_CUSTOM_${Math.floor(1000 + Math.random() * 9000)}`,
        name: "Sarah Jenkins",
        age: 38,
        is_male: 0,
        baseline_bmi: 22.4,
        smoking_numeric: 0,
        family_history_diabetes: 0,
        ethnicity: "Caucasian",
        fasting_glucose_latest: 86.0,
        fasting_glucose_mean: 88.0,
        hba1c_latest: 5.05,
        hba1c_mean: 5.0,
        systolic_bp_latest: 114.0,
        systolic_bp_mean: 116.0,
        diastolic_bp_latest: 72.0,
        diastolic_bp_mean: 74.0,
        heart_rate_latest: 66.0,
        total_cholesterol_latest: 165.0,
        ldl_latest: 88.0,
        hdl_latest: 62.0,
        triglycerides_latest: 85.0,
        serum_creatinine_latest: 0.72,
        egfr_latest: 110.0,
        clinical_summary: "38yo female with optimal biomarker levels and no cardiometabolic risk factors.",
        risk_label: "Custom Low Risk",
      });
    }
  };

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem", marginBottom: "1.5rem" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span className="badge badge-mint" style={{ background: "rgba(45, 212, 191, 0.12)", color: "var(--accent-mint)", border: "1px solid rgba(45, 212, 191, 0.25)" }}>
              Interactive Clinician Intake
            </span>
            <span className="badge badge-neutral">Real-Time Scoring</span>
          </div>
          <h3 style={{ fontSize: "1.35rem", fontWeight: 700, color: "#fff", marginTop: "0.4rem" }}>
            Direct Patient Data Entry & Risk Assessment
          </h3>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>
            Enter new or custom patient demographics and laboratory values to calculate instant multi-model predictions, conformal sets, and actionable counterfactual recourse.
          </p>
        </div>

        {/* Quick Presets */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Presets:</span>
          <button
            type="button"
            onClick={() => loadPreset("high")}
            style={{ fontSize: "0.75rem", padding: "0.3rem 0.6rem", borderRadius: 6, background: "rgba(239, 68, 68, 0.15)", color: "#f87171", border: "1px solid rgba(239, 68, 68, 0.3)", cursor: "pointer" }}
          >
            High Risk
          </button>
          <button
            type="button"
            onClick={() => loadPreset("borderline")}
            style={{ fontSize: "0.75rem", padding: "0.3rem 0.6rem", borderRadius: 6, background: "rgba(245, 158, 11, 0.15)", color: "#fbbf24", border: "1px solid rgba(245, 158, 11, 0.3)", cursor: "pointer" }}
          >
            Borderline
          </button>
          <button
            type="button"
            onClick={() => loadPreset("low")}
            style={{ fontSize: "0.75rem", padding: "0.3rem 0.6rem", borderRadius: 6, background: "rgba(16, 185, 129, 0.15)", color: "#34d399", border: "1px solid rgba(16, 185, 129, 0.3)", cursor: "pointer" }}
          >
            Healthy
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Section 1: Demographics */}
        <div style={{ marginBottom: "1.25rem" }}>
          <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--accent-mint)", textTransform: "uppercase", letterSpacing: "0.04em", display: "block", marginBottom: "0.75rem" }}>
            1. Patient Identification & Demographics
          </span>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Patient Full Name
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => handleChange("name", e.target.value)}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Patient ID
              </label>
              <input
                type="text"
                value={formData.patient_id}
                onChange={(e) => handleChange("patient_id", e.target.value)}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Age (Years)
              </label>
              <input
                type="number"
                min="18"
                max="99"
                value={formData.age}
                onChange={(e) => handleChange("age", parseInt(e.target.value) || 0)}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Biological Sex
              </label>
              <select
                value={formData.is_male}
                onChange={(e) => handleChange("is_male", parseInt(e.target.value))}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              >
                <option value={0}>Female</option>
                <option value={1}>Male</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Baseline BMI (kg/m&sup2;)
              </label>
              <input
                type="number"
                step="0.1"
                value={formData.baseline_bmi}
                onChange={(e) => handleChange("baseline_bmi", parseFloat(e.target.value) || 0)}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Smoking Status
              </label>
              <select
                value={formData.smoking_numeric}
                onChange={(e) => handleChange("smoking_numeric", parseInt(e.target.value))}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              >
                <option value={0}>Non-Smoker / Never</option>
                <option value={1}>Smoker / Former</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Family History of Diabetes
              </label>
              <select
                value={formData.family_history_diabetes}
                onChange={(e) => handleChange("family_history_diabetes", parseInt(e.target.value))}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              >
                <option value={0}>No Known Family History (0)</option>
                <option value={1}>First-Degree Relative (+1)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section 2: Glycemic & Metabolic Biomarkers */}
        <div style={{ marginBottom: "1.25rem", borderTop: "1px solid var(--border-subtle)", paddingTop: "1rem" }}>
          <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--accent-mint)", textTransform: "uppercase", letterSpacing: "0.04em", display: "block", marginBottom: "0.75rem" }}>
            2. Glycemic & Metabolic Laboratory Panel
          </span>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Fasting Glucose (mg/dL)
              </label>
              <input
                type="number"
                step="0.1"
                value={formData.fasting_glucose_latest}
                onChange={(e) => {
                  const val = parseFloat(e.target.value) || 0;
                  handleChange("fasting_glucose_latest", val);
                  handleChange("fasting_glucose_mean", Math.round(val * 0.96 * 10) / 10);
                }}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Glycated Hemoglobin (HbA1c %)
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.hba1c_latest}
                onChange={(e) => {
                  const val = parseFloat(e.target.value) || 0;
                  handleChange("hba1c_latest", val);
                  handleChange("hba1c_mean", Math.round((val - 0.15) * 100) / 100);
                }}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Total Cholesterol (mg/dL)
              </label>
              <input
                type="number"
                step="1"
                value={formData.total_cholesterol_latest}
                onChange={(e) => handleChange("total_cholesterol_latest", parseFloat(e.target.value) || 0)}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Triglycerides (mg/dL)
              </label>
              <input
                type="number"
                step="1"
                value={formData.triglycerides_latest}
                onChange={(e) => handleChange("triglycerides_latest", parseFloat(e.target.value) || 0)}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>
          </div>
        </div>

        {/* Section 3: Cardiovascular & Renal Profile */}
        <div style={{ marginBottom: "1.5rem", borderTop: "1px solid var(--border-subtle)", paddingTop: "1rem" }}>
          <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--accent-mint)", textTransform: "uppercase", letterSpacing: "0.04em", display: "block", marginBottom: "0.75rem" }}>
            3. Cardiovascular Vitals & Renal Function
          </span>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Systolic BP (mmHg)
              </label>
              <input
                type="number"
                step="1"
                value={formData.systolic_bp_latest}
                onChange={(e) => {
                  const val = parseFloat(e.target.value) || 0;
                  handleChange("systolic_bp_latest", val);
                  handleChange("systolic_bp_mean", Math.round(val - 3));
                }}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Diastolic BP (mmHg)
              </label>
              <input
                type="number"
                step="1"
                value={formData.diastolic_bp_latest}
                onChange={(e) => {
                  const val = parseFloat(e.target.value) || 0;
                  handleChange("diastolic_bp_latest", val);
                  handleChange("diastolic_bp_mean", Math.round(val - 2));
                }}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                eGFR (mL/min/1.73m&sup2;)
              </label>
              <input
                type="number"
                step="1"
                value={formData.egfr_latest}
                onChange={(e) => handleChange("egfr_latest", parseFloat(e.target.value) || 0)}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.3rem" }}>
                Serum Creatinine (mg/dL)
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.serum_creatinine_latest}
                onChange={(e) => handleChange("serum_creatinine_latest", parseFloat(e.target.value) || 0)}
                style={{ width: "100%", padding: "0.55rem 0.75rem", background: "var(--bg-surface-elevated)", border: "1px solid var(--border-subtle)", borderRadius: 6, color: "#fff", outline: "none" }}
              />
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "1rem", borderTop: "1px solid var(--border-subtle)", paddingTop: "1rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <button
              type="submit"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.5rem",
                padding: "0.65rem 1.25rem",
                background: "linear-gradient(135deg, var(--accent-emerald), var(--accent-mint))",
                color: "#090d16",
                fontWeight: 700,
                fontSize: "0.85rem",
                border: "none",
                borderRadius: 8,
                cursor: "pointer",
                boxShadow: "0 0 12px rgba(45, 212, 191, 0.35)",
              }}
            >
              <ActivityIcon size={16} /> Evaluate Patient Risk & Recourse
            </button>

            <button
              type="button"
              onClick={handleSaveToSupabase}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.5rem",
                padding: "0.65rem 1.25rem",
                background: "var(--bg-surface-elevated)",
                color: "#ffffff",
                fontWeight: 600,
                fontSize: "0.85rem",
                border: "1px solid var(--border-highlight)",
                borderRadius: 8,
                cursor: "pointer",
              }}
            >
              <ShieldCheckIcon size={16} color="var(--accent-mint)" /> Save to Cloud EHR (Supabase)
            </button>
          </div>

          {saveStatus && (
            <span style={{ fontSize: "0.8rem", color: saveStatus.includes("Success") ? "var(--accent-emerald)" : "var(--accent-amber)" }}>
              {saveStatus}
            </span>
          )}
        </div>
      </form>
    </div>
  );
}
