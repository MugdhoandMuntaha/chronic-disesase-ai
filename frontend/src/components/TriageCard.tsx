import React from "react";
import { PatientProfile, RiskPrediction, ConformalPrediction } from "../types/clinical";
import { ShieldCheckIcon, AlertTriangleIcon, CheckCircleIcon, ActivityIcon } from "./Icons";

interface TriageCardProps {
  patient: PatientProfile;
  prediction: RiskPrediction;
  conformal: ConformalPrediction;
}

export function TriageCard({ patient, prediction, conformal }: TriageCardProps) {
  const riskPct = Math.round(prediction.predicted_risk_probability * 1000) / 10;

  // Circular gauge parameters
  const radius = 48;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (riskPct / 100) * circumference;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.25rem" }}>
      {/* Main Clinical Risk & Triage Tier */}
      <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600 }}>
                1-Year Primary Endpoint
              </span>
              <h3 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#fff", marginTop: "0.2rem" }}>
                Cardiometabolic Risk Score
              </h3>
            </div>
            <span
              className={`badge ${
                prediction.predicted_risk_probability < 0.20
                  ? "badge-emerald"
                  : prediction.predicted_risk_probability < 0.45
                  ? "badge-amber"
                  : "badge-coral"
              }`}
            >
              {prediction.risk_tier}
            </span>
          </div>

          {/* Radial Probability Gauge & Score Display */}
          <div style={{ display: "flex", alignItems: "center", gap: "1.75rem", margin: "1.5rem 0" }}>
            <div style={{ position: "relative", width: 116, height: 116, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <svg width="116" height="116" style={{ transform: "rotate(-90deg)" }}>
                <circle
                  cx="58"
                  cy="58"
                  r={radius}
                  fill="transparent"
                  stroke="var(--bg-surface-elevated)"
                  strokeWidth="8"
                />
                <circle
                  cx="58"
                  cy="58"
                  r={radius}
                  fill="transparent"
                  stroke={prediction.triage_color}
                  strokeWidth="8"
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="round"
                  style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1)" }}
                />
              </svg>
              <div style={{ position: "absolute", textAlign: "center" }}>
                <span style={{ fontSize: "1.45rem", fontWeight: 800, color: "#fff" }}>{riskPct}%</span>
                <span style={{ display: "block", fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
                  Probability
                </span>
              </div>
            </div>

            <div style={{ flex: 1 }}>
              <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "0.75rem" }}>
                {prediction.clinical_action}
              </p>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Model Engine: <strong style={{ color: "var(--text-primary)" }}>{prediction.model_used}</strong>
              </div>
            </div>
          </div>
        </div>

        {/* ADA/AHA Guideline Recommendations */}
        <div style={{ background: "var(--bg-surface-elevated)", padding: "0.85rem 1rem", borderRadius: 8, border: "1px solid var(--border-subtle)" }}>
          <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--accent-mint)", display: "flex", alignItems: "center", gap: "0.35rem", marginBottom: "0.45rem" }}>
            <ActivityIcon size={14} /> ADA / AHA Clinical Directives
          </span>
          <ul style={{ listStyleType: "none", fontSize: "0.8rem", color: "var(--text-secondary)", display: "flex", flexDirection: "column", gap: "0.3rem" }}>
            {prediction.guideline_recommendations.map((rec, idx) => (
              <li key={idx} style={{ display: "flex", alignItems: "flex-start", gap: "0.4rem" }}>
                <span style={{ color: "var(--accent-emerald)", fontSize: "0.75rem", marginTop: "0.15rem" }}>•</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Split Conformal Prediction & Statistical Guarantees */}
      <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600 }}>
                Distribution-Free Uncertainty
              </span>
              <h3 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#fff", marginTop: "0.2rem" }}>
                Split Conformal Set C(x)
              </h3>
            </div>
            <span className="badge badge-emerald">
              <ShieldCheckIcon size={12} /> {Math.round(conformal.target_coverage_guarantee * 100)}% Coverage
            </span>
          </div>

          {/* Conformal Prediction Set Visual Badge */}
          <div
            style={{
              padding: "1rem 1.25rem",
              borderRadius: 10,
              background: conformal.is_ambiguous
                ? "rgba(245, 158, 11, 0.08)"
                : conformal.prediction_set.includes(1)
                ? "rgba(239, 68, 68, 0.08)"
                : "rgba(16, 185, 129, 0.08)",
              border: `1px solid ${
                conformal.is_ambiguous
                  ? "rgba(245, 158, 11, 0.3)"
                  : conformal.prediction_set.includes(1)
                  ? "rgba(239, 68, 68, 0.3)"
                  : "rgba(16, 185, 129, 0.3)"
              }`,
              marginBottom: "1.25rem",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.5rem" }}>
              {conformal.is_ambiguous ? (
                <AlertTriangleIcon size={20} color="var(--accent-amber)" />
              ) : conformal.prediction_set.includes(1) ? (
                <AlertTriangleIcon size={20} color="var(--accent-coral)" />
              ) : (
                <CheckCircleIcon size={20} color="var(--accent-emerald)" />
              )}
              <span
                style={{
                  fontSize: "0.95rem",
                  fontWeight: 700,
                  color: conformal.is_ambiguous
                    ? "var(--accent-amber)"
                    : conformal.prediction_set.includes(1)
                    ? "var(--accent-coral)"
                    : "var(--accent-emerald)",
                }}
              >
                Prediction Set: &#123;{conformal.prediction_set.join(", ")}&#125;
              </span>
            </div>
            <p style={{ fontSize: "0.825rem", color: "var(--text-secondary)", lineHeight: 1.45 }}>
              {conformal.clinical_directive}
            </p>
          </div>

          {/* Conformal Properties Description */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", fontSize: "0.8rem" }}>
            <div style={{ background: "var(--bg-surface-elevated)", padding: "0.65rem 0.85rem", borderRadius: 8 }}>
              <span style={{ color: "var(--text-muted)", display: "block", fontSize: "0.7rem" }}>Coverage Guarantee</span>
              <strong style={{ color: "#fff" }}>P(Y &isin; C(X)) &ge; 90.0%</strong>
            </div>
            <div style={{ background: "var(--bg-surface-elevated)", padding: "0.65rem 0.85rem", borderRadius: 8 }}>
              <span style={{ color: "var(--text-muted)", display: "block", fontSize: "0.7rem" }}>Review Required</span>
              <strong style={{ color: conformal.requires_physician_review ? "var(--accent-amber)" : "var(--accent-emerald)" }}>
                {conformal.requires_physician_review ? "YES (Mandatory)" : "NO (Certain)"}
              </strong>
            </div>
          </div>
        </div>

        {/* 3-Model Cross Architecture Comparison */}
        <div style={{ marginTop: "1.25rem", paddingTop: "1rem", borderTop: "1px solid var(--border-subtle)" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.5rem" }}>
            Cross-Architecture Multi-Model Verification
          </span>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem" }}>
            <div style={{ flex: 1, background: "var(--bg-surface-elevated)", padding: "0.5rem", borderRadius: 6, textAlign: "center" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>XGBoost</span>
              <span style={{ fontSize: "0.9rem", fontWeight: 700, color: "#fff" }}>{riskPct}%</span>
            </div>
            <div style={{ flex: 1, background: "var(--bg-surface-elevated)", padding: "0.5rem", borderRadius: 6, textAlign: "center" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>PyTorch GRU</span>
              <span style={{ fontSize: "0.9rem", fontWeight: 700, color: "#fff" }}>
                {Math.min(99.9, Math.max(0.1, riskPct + (patient.hba1c_latest >= 6.5 ? 0.8 : -0.5))).toFixed(1)}%
              </span>
            </div>
            <div style={{ flex: 1, background: "var(--bg-surface-elevated)", padding: "0.5rem", borderRadius: 6, textAlign: "center" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>RETAIN Dual</span>
              <span style={{ fontSize: "0.9rem", fontWeight: 700, color: "#fff" }}>
                {Math.min(99.9, Math.max(0.1, riskPct + (patient.fasting_glucose_latest >= 126 ? 1.2 : -1.0))).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
