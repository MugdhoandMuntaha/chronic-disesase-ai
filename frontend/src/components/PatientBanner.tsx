import React from "react";
import { PatientProfile } from "../types/clinical";
import { UserIcon, ActivityIcon, CheckCircleIcon } from "./Icons";

interface PatientBannerProps {
  patient: PatientProfile;
}

export function PatientBanner({ patient }: PatientBannerProps) {
  return (
    <div className="card-elevated" style={{ marginBottom: "1.5rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
        {/* Left: Patient Avatar & Demographics */}
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <div
            style={{
              width: 46,
              height: 46,
              borderRadius: "50%",
              background: "rgba(45, 212, 191, 0.12)",
              border: "1px solid rgba(45, 212, 191, 0.3)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--accent-mint)",
            }}
          >
            <UserIcon size={24} />
          </div>

          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <h2 style={{ fontSize: "1.2rem", fontWeight: 700, color: "#ffffff" }}>
                {patient.name}
              </h2>
              <span className="badge badge-neutral" style={{ fontSize: "0.7rem" }}>
                ID: {patient.patient_id}
              </span>
              <span className="badge badge-iris" style={{ fontSize: "0.7rem" }}>
                {patient.ethnicity}
              </span>
            </div>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.15rem" }}>
              {patient.age} years old &bull; {patient.is_male ? "Male" : "Female"} &bull; BMI {patient.baseline_bmi} kg/m&sup2; &bull; {patient.smoking_numeric ? "Smoker" : "Non-Smoker"} &bull; {patient.family_history_diabetes ? "Family History (+)" : "No Family History"}
            </p>
          </div>
        </div>

        {/* Right: Key Laboratory Biomarkers Pills */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
          <div style={{ background: "var(--bg-surface)", padding: "0.35rem 0.65rem", borderRadius: 6, border: "1px solid var(--border-subtle)", textAlign: "center" }}>
            <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", display: "block" }}>FPG</span>
            <strong style={{ fontSize: "0.85rem", color: patient.fasting_glucose_latest >= 126 ? "var(--accent-coral)" : patient.fasting_glucose_latest >= 100 ? "var(--accent-amber)" : "var(--accent-emerald)" }}>
              {patient.fasting_glucose_latest} mg/dL
            </strong>
          </div>

          <div style={{ background: "var(--bg-surface)", padding: "0.35rem 0.65rem", borderRadius: 6, border: "1px solid var(--border-subtle)", textAlign: "center" }}>
            <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", display: "block" }}>HbA1c</span>
            <strong style={{ fontSize: "0.85rem", color: patient.hba1c_latest >= 6.5 ? "var(--accent-coral)" : patient.hba1c_latest >= 5.7 ? "var(--accent-amber)" : "var(--accent-emerald)" }}>
              {patient.hba1c_latest}%
            </strong>
          </div>

          <div style={{ background: "var(--bg-surface)", padding: "0.35rem 0.65rem", borderRadius: 6, border: "1px solid var(--border-subtle)", textAlign: "center" }}>
            <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", display: "block" }}>BP</span>
            <strong style={{ fontSize: "0.85rem", color: patient.systolic_bp_latest >= 140 ? "var(--accent-coral)" : patient.systolic_bp_latest >= 130 ? "var(--accent-amber)" : "var(--accent-emerald)" }}>
              {patient.systolic_bp_latest}/{patient.diastolic_bp_latest}
            </strong>
          </div>

          <div style={{ background: "var(--bg-surface)", padding: "0.35rem 0.65rem", borderRadius: 6, border: "1px solid var(--border-subtle)", textAlign: "center" }}>
            <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", display: "block" }}>eGFR</span>
            <strong style={{ fontSize: "0.85rem", color: patient.egfr_latest < 60 ? "var(--accent-coral)" : patient.egfr_latest < 90 ? "var(--accent-amber)" : "var(--accent-emerald)" }}>
              {patient.egfr_latest} mL/min
            </strong>
          </div>
        </div>
      </div>

      {/* Clinical Summary */}
      <div style={{ marginTop: "0.75rem", paddingTop: "0.75rem", borderTop: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
        <ActivityIcon size={14} color="var(--accent-mint)" />
        <span>{patient.clinical_summary}</span>
      </div>
    </div>
  );
}
