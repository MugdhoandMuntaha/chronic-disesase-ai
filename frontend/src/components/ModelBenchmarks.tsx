import React from "react";
import { MOCK_LEADERBOARD } from "../data/mockPatients";
import { TrendingUpIcon, ActivityIcon, CheckCircleIcon } from "./Icons";

export function ModelBenchmarksView() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      {/* Overview */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span className="badge badge-emerald">Medical Benchmark Standard</span>
              <span className="badge badge-neutral">Held-Out Test Cohort (N=200)</span>
            </div>
            <h3 style={{ fontSize: "1.35rem", fontWeight: 700, color: "#fff", marginTop: "0.4rem" }}>
              Cross-Architecture Clinical Benchmark Leaderboard
            </h3>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>
              Side-by-side performance comparison evaluated on held-out test patients across static tabular aggregations and longitudinal 3D tensor representations.
            </p>
          </div>
        </div>
      </div>

      {/* Leaderboard Table */}
      <div className="card" style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", fontSize: "0.825rem", borderCollapse: "collapse", whiteSpace: "nowrap" }}>
          <thead>
            <tr style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border-subtle)", textAlign: "left" }}>
              <th style={{ padding: "0.6rem 0.5rem" }}>Model Architecture</th>
              <th style={{ padding: "0.6rem 0.5rem" }}>Input Modality</th>
              <th style={{ padding: "0.6rem 0.5rem" }}>AUROC</th>
              <th style={{ padding: "0.6rem 0.5rem" }}>AUPRC</th>
              <th style={{ padding: "0.6rem 0.5rem" }}>Sens @ 0.5</th>
              <th style={{ padding: "0.6rem 0.5rem" }}>Spec @ 0.5</th>
              <th style={{ padding: "0.6rem 0.5rem" }}>Sens @ 90% Spec</th>
              <th style={{ padding: "0.6rem 0.5rem" }}>Brier Score</th>
              <th style={{ padding: "0.6rem 0.5rem" }}>Interpretability Mode</th>
            </tr>
          </thead>
          <tbody>
            {MOCK_LEADERBOARD.map((m, idx) => (
              <tr key={idx} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                <td style={{ padding: "0.7rem 0.5rem", color: "#fff", fontWeight: 700 }}>{m.model}</td>
                <td style={{ padding: "0.7rem 0.5rem", color: "var(--text-secondary)" }}>{m.modality}</td>
                <td style={{ padding: "0.7rem 0.5rem", color: "var(--accent-emerald)", fontWeight: 700 }}>
                  {m.auroc.toFixed(4)}
                </td>
                <td style={{ padding: "0.7rem 0.5rem", color: "var(--accent-emerald)", fontWeight: 700 }}>
                  {m.auprc.toFixed(4)}
                </td>
                <td style={{ padding: "0.7rem 0.5rem", color: "var(--text-primary)" }}>
                  {m.sensitivity_50.toFixed(4)}
                </td>
                <td style={{ padding: "0.7rem 0.5rem", color: "var(--text-primary)" }}>
                  {m.specificity_50.toFixed(4)}
                </td>
                <td style={{ padding: "0.7rem 0.5rem", color: "var(--accent-mint)", fontWeight: 700 }}>
                  {m.sens_at_90_spec.toFixed(4)}
                </td>
                <td style={{ padding: "0.7rem 0.5rem", color: "var(--text-secondary)" }}>
                  {m.brier_score.toFixed(6)}
                </td>
                <td style={{ padding: "0.7rem 0.5rem", color: "var(--text-muted)" }}>{m.interpretability}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Decision Curve Analysis (DCA) Net Benefit Summary Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1.25rem" }}>
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <TrendingUpIcon size={18} color="var(--accent-mint)" />
            <h4 style={{ fontSize: "1rem", fontWeight: 700, color: "#fff" }}>Decision Curve Analysis (DCA)</h4>
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
            Vickers & Elkin decision-analytic method quantifying real-world clinical utility across triage decision thresholds.
          </p>
          <div style={{ display: "flex", gap: "0.75rem" }}>
            <div style={{ flex: 1, background: "var(--bg-surface-elevated)", padding: "0.75rem", borderRadius: 8, textAlign: "center" }}>
              <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", display: "block" }}>XGBoost Net Benefit</span>
              <strong style={{ fontSize: "1.25rem", color: "var(--accent-emerald)" }}>0.1450</strong>
              <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", display: "block" }}>@ 14.9% threshold</span>
            </div>
            <div style={{ flex: 1, background: "var(--bg-surface-elevated)", padding: "0.75rem", borderRadius: 8, textAlign: "center" }}>
              <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", display: "block" }}>Treat All Strategy</span>
              <strong style={{ fontSize: "1.25rem", color: "var(--text-muted)" }}>0.0000</strong>
              <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", display: "block" }}>Zero net clinical utility</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <ActivityIcon size={18} color="var(--accent-emerald)" />
            <h4 style={{ fontSize: "1rem", fontWeight: 700, color: "#fff" }}>Interventions Avoided</h4>
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
            Number of non-diseased individuals spared from unnecessary biopsies, expensive panels, and overtreatment.
          </p>
          <div style={{ background: "rgba(16, 185, 129, 0.08)", padding: "1rem", borderRadius: 8, border: "1px solid rgba(16, 185, 129, 0.25)", textAlign: "center" }}>
            <span style={{ fontSize: "1.75rem", fontWeight: 800, color: "var(--accent-mint)", display: "block" }}>
              85.5 / 100
            </span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
              Patients spared unnecessary clinical intervention with zero missed chronic onset diagnoses
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
