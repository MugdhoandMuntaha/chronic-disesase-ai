import React from "react";
import { ScaleIcon, ActivityIcon, CheckCircleIcon, ShieldCheckIcon } from "./Icons";

export function FairnessAndDriftView() {
  const sexDisparity = [
    { group: "Male (N=103)", auroc: "1.000", fpr: "0.000", fnr: "0.000", parity: "Optimal" },
    { group: "Female (N=97)", auroc: "1.000", fpr: "0.000", fnr: "0.000", parity: "Optimal" },
  ];

  const ageDisparity = [
    { group: "Young Adults (<50)", sampleSize: 67, prevalence: "9.0%", auroc: "1.000" },
    { group: "Middle Aged (50-65)", sampleSize: 92, prevalence: "9.8%", auroc: "1.000" },
    { group: "Seniors (>65)", sampleSize: 41, prevalence: "34.2%", auroc: "1.000" },
  ];

  const driftFeatures = [
    { feature: "HbA1c (latest)", psi: 0.1162, ks_stat: 0.065, p_val: 0.497, status: "Moderate Shift", tier: "amber" },
    { feature: "Age", psi: 0.0989, ks_stat: 0.058, p_val: 0.653, status: "Stable", tier: "emerald" },
    { feature: "Fasting Glucose (latest)", psi: 0.0745, ks_stat: 0.044, p_val: 0.912, status: "Stable", tier: "emerald" },
    { feature: "Systolic BP (latest)", psi: 0.0289, ks_stat: 0.069, p_val: 0.425, status: "Stable", tier: "emerald" },
    { feature: "eGFR (latest)", psi: 0.0522, ks_stat: 0.103, p_val: 0.066, status: "Stable", tier: "emerald" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      {/* Overview Card */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span className="badge badge-emerald">Regulatory Compliance</span>
              <span className="badge badge-neutral">EEOC 80% Rule</span>
            </div>
            <h3 style={{ fontSize: "1.35rem", fontWeight: 700, color: "#fff", marginTop: "0.4rem" }}>
              Algorithmic Fairness Audit & Population Stability Monitor
            </h3>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>
              Ensures demographic equity across protected patient characteristics and continuously monitors Population Stability Index (PSI) to detect covariate distribution shifts.
            </p>
          </div>
        </div>
      </div>

      {/* Fairness & Equity Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.25rem" }}>
        {/* Sex Disparity Audit */}
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.85rem" }}>
            <ScaleIcon size={18} color="var(--accent-mint)" />
            <h4 style={{ fontSize: "1rem", fontWeight: 700, color: "#fff" }}>Sex Disparity & Equalized Odds</h4>
          </div>

          <div style={{ display: "flex", gap: "0.75rem", marginBottom: "1rem" }}>
            <div style={{ flex: 1, background: "var(--bg-surface-elevated)", padding: "0.65rem", borderRadius: 8, textAlign: "center" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>Equalized Odds Gap</span>
              <strong style={{ fontSize: "1.1rem", color: "var(--accent-emerald)" }}>0.000</strong>
            </div>
            <div style={{ flex: 1, background: "var(--bg-surface-elevated)", padding: "0.65rem", borderRadius: 8, textAlign: "center" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>Parity Compliance</span>
              <strong style={{ fontSize: "1.1rem", color: "var(--accent-emerald)" }}>PASS</strong>
            </div>
          </div>

          <table style={{ width: "100%", fontSize: "0.8rem", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border-subtle)", textAlign: "left" }}>
                <th style={{ padding: "0.4rem 0" }}>Cohort Group</th>
                <th style={{ padding: "0.4rem 0" }}>AUROC</th>
                <th style={{ padding: "0.4rem 0" }}>FPR</th>
                <th style={{ padding: "0.4rem 0" }}>FNR</th>
              </tr>
            </thead>
            <tbody>
              {sexDisparity.map((s, idx) => (
                <tr key={idx} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                  <td style={{ padding: "0.5rem 0", color: "#fff", fontWeight: 600 }}>{s.group}</td>
                  <td style={{ padding: "0.5rem 0", color: "var(--accent-emerald)" }}>{s.auroc}</td>
                  <td style={{ padding: "0.5rem 0", color: "var(--text-secondary)" }}>{s.fpr}</td>
                  <td style={{ padding: "0.5rem 0", color: "var(--text-secondary)" }}>{s.fnr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Age Bracket Equity */}
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.85rem" }}>
            <ActivityIcon size={18} color="var(--accent-iris)" />
            <h4 style={{ fontSize: "1rem", fontWeight: 700, color: "#fff" }}>Age Bracket Stratification</h4>
          </div>

          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
            Validates discrimination across young, middle-aged, and geriatric populations with elevated baseline prevalence.
          </p>

          <table style={{ width: "100%", fontSize: "0.8rem", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border-subtle)", textAlign: "left" }}>
                <th style={{ padding: "0.4rem 0" }}>Age Bracket</th>
                <th style={{ padding: "0.4rem 0" }}>Sample Size</th>
                <th style={{ padding: "0.4rem 0" }}>Prevalence</th>
                <th style={{ padding: "0.4rem 0" }}>AUROC</th>
              </tr>
            </thead>
            <tbody>
              {ageDisparity.map((a, idx) => (
                <tr key={idx} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                  <td style={{ padding: "0.5rem 0", color: "#fff", fontWeight: 600 }}>{a.group}</td>
                  <td style={{ padding: "0.5rem 0", color: "var(--text-secondary)" }}>N={a.sampleSize}</td>
                  <td style={{ padding: "0.5rem 0", color: "var(--text-secondary)" }}>{a.prevalence}</td>
                  <td style={{ padding: "0.5rem 0", color: "var(--accent-emerald)" }}>{a.auroc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Population Covariate Drift (PSI) Table */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <h4 style={{ fontSize: "1rem", fontWeight: 700, color: "#fff" }}>
              Population Stability Index (PSI) Covariate Monitoring
            </h4>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Thresholds: PSI &lt; 0.10 (Stable) | 0.10 - 0.20 (Moderate Drift) | &ge; 0.20 (Significant Drift)
            </span>
          </div>
          <span className="badge badge-amber">
            Cohort Safety Code: AMBER (Max PSI: 0.1162)
          </span>
        </div>

        <table style={{ width: "100%", fontSize: "0.825rem", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border-subtle)", textAlign: "left" }}>
              <th style={{ padding: "0.5rem 0" }}>Monitored Biomarker</th>
              <th style={{ padding: "0.5rem 0" }}>PSI Value</th>
              <th style={{ padding: "0.5rem 0" }}>KS Statistic</th>
              <th style={{ padding: "0.5rem 0" }}>p-value</th>
              <th style={{ padding: "0.5rem 0" }}>Drift Status</th>
            </tr>
          </thead>
          <tbody>
            {driftFeatures.map((d, idx) => (
              <tr key={idx} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                <td style={{ padding: "0.55rem 0", color: "#fff", fontWeight: 600 }}>{d.feature}</td>
                <td style={{ padding: "0.55rem 0", color: d.tier === "amber" ? "var(--accent-amber)" : "var(--accent-emerald)" }}>
                  {d.psi.toFixed(4)}
                </td>
                <td style={{ padding: "0.55rem 0", color: "var(--text-secondary)" }}>{d.ks_stat.toFixed(3)}</td>
                <td style={{ padding: "0.55rem 0", color: "var(--text-secondary)" }}>{d.p_val.toFixed(3)}</td>
                <td style={{ padding: "0.55rem 0" }}>
                  <span className={`badge ${d.tier === "amber" ? "badge-amber" : "badge-emerald"}`} style={{ fontSize: "0.65rem" }}>
                    {d.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
