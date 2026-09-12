import React, { useState } from "react";
import { PatientProfile, CounterfactualRecourse } from "../types/clinical";
import { SliderIcon, CheckCircleIcon, ActivityIcon, ShieldCheckIcon } from "./Icons";

interface CounterfactualRecourseProps {
  patient: PatientProfile;
  recourse: CounterfactualRecourse;
  onUpdateTargetRisk: (risk: number) => void;
}

export function CounterfactualRecourseView({ patient, recourse, onUpdateTargetRisk }: CounterfactualRecourseProps) {
  const [sliderValue, setSliderValue] = useState<number>(recourse.target_risk * 100);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setSliderValue(val);
    onUpdateTargetRisk(val / 100);
  };

  const initialPct = Math.round(recourse.initial_risk * 1000) / 10;
  const counterfactualPct = Math.round(recourse.counterfactual_risk * 1000) / 10;
  const arrPct = Math.round(recourse.absolute_risk_reduction * 1000) / 10;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      {/* Overview Card */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem", marginBottom: "1.25rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span className="badge badge-mint" style={{ background: "rgba(45, 212, 191, 0.12)", color: "var(--accent-mint)", border: "1px solid rgba(45, 212, 191, 0.25)" }}>
                Actionable Clinical Optimization
              </span>
              <span className="badge badge-neutral">
                <ShieldCheckIcon size={12} /> Biologically Bounded
              </span>
            </div>
            <h3 style={{ fontSize: "1.35rem", fontWeight: 700, color: "#fff", marginTop: "0.4rem" }}>
              Counterfactual Recourse & Therapeutic Roadmap
            </h3>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>
              Solves for the minimal physiological modifications required to achieve target risk while respecting physiological boundaries and non-modifiable demographic invariants.
            </p>
          </div>

          {/* Absolute Risk Reduction Highlight */}
          <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
            <div style={{ background: "var(--bg-surface-elevated)", padding: "0.65rem 1rem", borderRadius: 10, border: "1px solid var(--border-subtle)", textAlign: "center" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>Initial Baseline Risk</span>
              <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--accent-coral)" }}>{initialPct}%</span>
            </div>
            <span style={{ color: "var(--text-muted)", fontSize: "1.2rem" }}>&rarr;</span>
            <div style={{ background: "var(--bg-surface-elevated)", padding: "0.65rem 1rem", borderRadius: 10, border: "1px solid rgba(16, 185, 129, 0.3)", textAlign: "center" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>Counterfactual Target</span>
              <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--accent-emerald)" }}>{counterfactualPct}%</span>
            </div>
            <div style={{ background: "rgba(16, 185, 129, 0.1)", padding: "0.65rem 1rem", borderRadius: 10, border: "1px solid rgba(16, 185, 129, 0.3)", textAlign: "center" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--accent-mint)", display: "block" }}>Absolute Reduction (ARR)</span>
              <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--accent-mint)" }}>-{arrPct}%</span>
            </div>
          </div>
        </div>

        {/* Interactive Target Risk Slider */}
        <div style={{ background: "var(--bg-surface-elevated)", padding: "1.25rem", borderRadius: 10, border: "1px solid var(--border-subtle)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
            <label htmlFor="target-risk-slider" style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <SliderIcon size={16} color="var(--accent-mint)" /> Clinician Target Operating Risk Level:
            </label>
            <span style={{ fontSize: "1rem", fontWeight: 700, color: "var(--accent-mint)" }}>
              {sliderValue.toFixed(0)}%
            </span>
          </div>
          <input
            id="target-risk-slider"
            type="range"
            min="10"
            max="40"
            step="1"
            value={sliderValue}
            onChange={handleSliderChange}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "0.4rem" }}>
            <span>10% (Strict Glycemic Normalization)</span>
            <span>20% (ADA Low-Risk Clinical Triage Threshold)</span>
            <span>40% (Moderate Risk Tolerant)</span>
          </div>
        </div>
      </div>

      {/* Recommended Action Directives */}
      <div>
        <h4 style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.85rem", display: "flex", alignItems: "center", gap: "0.45rem" }}>
          <ActivityIcon size={18} color="var(--accent-mint)" />
          Prescribed Biomarker Modifications ({recourse.actions_count} Actionable Directives)
        </h4>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
          {recourse.recommended_actions.map((action, idx) => (
            <div key={idx} className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
                  <h5 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#fff" }}>
                    {action.display_name}
                  </h5>
                  <span className="badge badge-emerald">
                    -{action.required_reduction} {action.unit}
                  </span>
                </div>

                {/* Before / After Delta Display */}
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", background: "var(--bg-surface-elevated)", padding: "0.6rem 0.8rem", borderRadius: 8, marginBottom: "0.75rem" }}>
                  <div>
                    <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", display: "block" }}>Baseline</span>
                    <span style={{ fontSize: "1rem", fontWeight: 700, color: "var(--accent-coral)" }}>
                      {action.baseline_value} <span style={{ fontSize: "0.7rem", fontWeight: 400 }}>{action.unit}</span>
                    </span>
                  </div>
                  <span style={{ color: "var(--text-muted)", fontSize: "1rem" }}>&rarr;</span>
                  <div>
                    <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", display: "block" }}>Target Goal</span>
                    <span style={{ fontSize: "1rem", fontWeight: 700, color: "var(--accent-emerald)" }}>
                      {action.target_value} <span style={{ fontSize: "0.7rem", fontWeight: 400 }}>{action.unit}</span>
                    </span>
                  </div>
                </div>

                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
                  {action.clinical_directive}
                </p>
              </div>

              <div style={{ marginTop: "0.85rem", paddingTop: "0.6rem", borderTop: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.7rem", color: "var(--text-muted)" }}>
                <CheckCircleIcon size={12} color="var(--accent-mint)" /> ADA / AHA Aligned Recommendation
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Demographic Invariant Protection Card */}
      <div className="card-elevated" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.65rem" }}>
          <ShieldCheckIcon size={20} color="var(--accent-iris)" />
          <div>
            <strong style={{ fontSize: "0.85rem", color: "#fff", display: "block" }}>Demographic Immutability Guardrail Active</strong>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Patient attributes ({patient.age}yo, {patient.is_male ? "Male" : "Female"}, {patient.ethnicity}) were constrained as strictly non-modifiable during optimization.
            </span>
          </div>
        </div>
        <span className="badge badge-iris">Ethics & Safety Enforced</span>
      </div>
    </div>
  );
}
