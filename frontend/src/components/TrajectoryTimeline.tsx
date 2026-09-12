import React from "react";
import { PatientProfile } from "../types/clinical";
import { MOCK_RETAIN_TIMELINE } from "../data/mockPatients";
import { CalendarIcon, TrendingUpIcon, ActivityIcon } from "./Icons";

interface TrajectoryTimelineProps {
  patient: PatientProfile;
}

export function TrajectoryTimeline({ patient }: TrajectoryTimelineProps) {
  const visits = MOCK_RETAIN_TIMELINE[patient.patient_id] || [
    {
      encounter_id: "ENC_01",
      days_to_index: -365,
      encounter_date: "12 Months Ago",
      visit_weight: 0.20,
      top_biomarker: `Fasting Glucose (${patient.fasting_glucose_mean} mg/dL)`,
      biomarker_weight: 0.35,
      fasting_glucose: patient.fasting_glucose_mean,
      hba1c: patient.hba1c_mean,
      systolic_bp: patient.systolic_bp_mean,
      diastolic_bp: patient.diastolic_bp_mean,
      egfr: patient.egfr_latest + 5,
    },
    {
      encounter_id: "ENC_02",
      days_to_index: -30,
      encounter_date: "Recent Encounter",
      visit_weight: 0.80,
      top_biomarker: `HbA1c (${patient.hba1c_latest}%)`,
      biomarker_weight: 0.85,
      fasting_glucose: patient.fasting_glucose_latest,
      hba1c: patient.hba1c_latest,
      systolic_bp: patient.systolic_bp_latest,
      diastolic_bp: patient.diastolic_bp_latest,
      egfr: patient.egfr_latest,
    },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      {/* Header Info */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span className="badge badge-iris">PyTorch RETAIN Attention</span>
              <span className="badge badge-neutral">Reverse-Time Trajectory</span>
            </div>
            <h3 style={{ fontSize: "1.35rem", fontWeight: 700, color: "#fff", marginTop: "0.4rem" }}>
              Longitudinal EHR Encounters & Temporal Attention (&alpha;<sub>t</sub>, &beta;<sub>t</sub>)
            </h3>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>
              Decomposes patient risk into visit-level attention weights (&alpha;<sub>t</sub>) and biomarker-level contribution vectors (&beta;<sub>t</sub>), revealing which past encounters drove the predicted onset.
            </p>
          </div>
        </div>
      </div>

      {/* Encounter Timeline Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
        {visits.map((v, idx) => {
          const isLatest = idx === visits.length - 1;
          const alphaPct = Math.round(v.visit_weight * 100);

          return (
            <div
              key={v.encounter_id}
              className="card"
              style={{
                borderColor: isLatest ? "rgba(45, 212, 191, 0.4)" : "var(--border-subtle)",
                background: isLatest ? "rgba(17, 24, 39, 0.95)" : "var(--bg-surface)",
                position: "relative",
              }}
            >
              {isLatest && (
                <span
                  className="badge badge-mint"
                  style={{
                    position: "absolute",
                    top: 12,
                    right: 12,
                    background: "rgba(45, 212, 191, 0.15)",
                    color: "var(--accent-mint)",
                    fontSize: "0.65rem",
                  }}
                >
                  Latest Encounter
                </span>
              )}

              {/* Encounter Header */}
              <div style={{ display: "flex", alignItems: "center", gap: "0.45rem", color: "var(--text-muted)", fontSize: "0.75rem", marginBottom: "0.4rem" }}>
                <CalendarIcon size={14} color="var(--accent-mint)" />
                <span>{v.encounter_date} (t = {v.days_to_index}d)</span>
              </div>

              <h4 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#fff", marginBottom: "0.75rem" }}>
                Encounter {v.encounter_id}
              </h4>

              {/* Visit Attention Score (alpha_t) */}
              <div style={{ background: "var(--bg-surface-elevated)", padding: "0.65rem 0.85rem", borderRadius: 8, marginBottom: "0.85rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", marginBottom: "0.35rem" }}>
                  <span style={{ color: "var(--text-muted)" }}>Visit Attention Weight (&alpha;<sub>t</sub>)</span>
                  <strong style={{ color: isLatest ? "var(--accent-mint)" : "var(--accent-iris)" }}>{alphaPct}%</strong>
                </div>
                <div style={{ width: "100%", height: 6, background: "var(--bg-surface)", borderRadius: 3, overflow: "hidden" }}>
                  <div
                    style={{
                      width: `${alphaPct}%`,
                      height: "100%",
                      background: isLatest
                        ? "linear-gradient(90deg, var(--accent-emerald), var(--accent-mint))"
                        : "var(--accent-iris)",
                      borderRadius: 3,
                    }}
                  />
                </div>
              </div>

              {/* Dominant Biomarker Driver (beta_t) */}
              <div style={{ marginBottom: "0.85rem" }}>
                <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block", textTransform: "uppercase" }}>
                  Primary Trajectory Driver (&beta;<sub>t</sub>)
                </span>
                <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-primary)" }}>
                  {v.top_biomarker}
                </span>
              </div>

              {/* Laboratory Snapshot */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", fontSize: "0.75rem", borderTop: "1px solid var(--border-subtle)", paddingTop: "0.75rem" }}>
                <div>
                  <span style={{ color: "var(--text-muted)", display: "block" }}>FPG</span>
                  <strong style={{ color: "#fff" }}>{v.fasting_glucose} mg/dL</strong>
                </div>
                <div>
                  <span style={{ color: "var(--text-muted)", display: "block" }}>HbA1c</span>
                  <strong style={{ color: "#fff" }}>{v.hba1c}%</strong>
                </div>
                <div>
                  <span style={{ color: "var(--text-muted)", display: "block" }}>BP</span>
                  <strong style={{ color: "#fff" }}>{v.systolic_bp}/{v.diastolic_bp}</strong>
                </div>
                <div>
                  <span style={{ color: "var(--text-muted)", display: "block" }}>eGFR</span>
                  <strong style={{ color: "#fff" }}>{v.egfr} mL/min</strong>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
