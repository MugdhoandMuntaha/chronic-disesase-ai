import React from "react";
import { PatientProfile } from "../types/clinical";
import { MOCK_PATIENTS } from "../data/mockPatients";
import { PulseIcon, ServerIcon, UserIcon } from "./Icons";

interface HeaderProps {
  selectedPatient: PatientProfile;
  onSelectPatient: (patient: PatientProfile) => void;
  isBackendOnline: boolean;
  activeTab: string;
  onSelectTab: (tabId: string) => void;
}

export function Header({
  selectedPatient,
  onSelectPatient,
  isBackendOnline,
  activeTab,
  onSelectTab,
}: HeaderProps) {
  const tabs = [
    { id: "triage", label: "Triage & CDSS" },
    { id: "intake", label: "+ Patient Intake" },
    { id: "recourse", label: "Actionable Recourse" },
    { id: "trajectory", label: "RETAIN Trajectory" },
    { id: "fairness", label: "Fairness & Drift" },
    { id: "benchmarks", label: "Model Leaderboard" },
  ];

  return (
    <header className="glass-header" style={{ position: "sticky", top: 0, zIndex: 50 }}>
      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "0.85rem 1.5rem" }}>
        {/* Top row: Brand + Status + Patient Switcher */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "1rem" }}>
          {/* Logo & Title */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <div
              style={{
                width: 38,
                height: 38,
                borderRadius: 10,
                background: "linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(45, 212, 191, 0.1))",
                border: "1px solid rgba(45, 212, 191, 0.3)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--accent-mint)",
              }}
            >
              <PulseIcon size={22} />
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ fontSize: "1.15rem", fontWeight: 700, letterSpacing: "-0.02em", color: "#ffffff" }}>
                  EndoPredict<span style={{ color: "var(--accent-mint)" }}>.ai</span>
                </span>
                <span className="badge badge-iris" style={{ fontSize: "0.65rem", padding: "0.15rem 0.45rem" }}>
                  Clinical CDSS
                </span>
              </div>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Cardiometabolic 1-Year Early Detection & Recourse System
              </p>
            </div>
          </div>

          {/* Right controls: Patient Selector & Backend Ping */}
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            {/* Patient Selector */}
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", background: "var(--bg-surface)", padding: "0.35rem 0.75rem", borderRadius: 8, border: "1px solid var(--border-subtle)" }}>
              <UserIcon size={16} color="var(--accent-mint)" />
              <label htmlFor="patient-select" style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 500 }}>
                Patient:
              </label>
              <select
                id="patient-select"
                value={selectedPatient.patient_id}
                onChange={(e) => {
                  const p = MOCK_PATIENTS.find((item) => item.patient_id === e.target.value);
                  if (p) onSelectPatient(p);
                }}
                style={{
                  background: "transparent",
                  color: "var(--text-primary)",
                  border: "none",
                  outline: "none",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {MOCK_PATIENTS.map((p) => (
                  <option key={p.patient_id} value={p.patient_id} style={{ background: "#111827", color: "#fff" }}>
                    {p.name} ({p.patient_id}) - {p.risk_label}
                  </option>
                ))}
              </select>
            </div>

            {/* Backend Connectivity Status */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.45rem",
                fontSize: "0.75rem",
                padding: "0.35rem 0.75rem",
                borderRadius: 8,
                background: isBackendOnline ? "rgba(16, 185, 129, 0.08)" : "rgba(245, 158, 11, 0.08)",
                border: isBackendOnline ? "1px solid rgba(16, 185, 129, 0.2)" : "1px solid rgba(245, 158, 11, 0.2)",
              }}
            >
              <span className="live-dot" style={{ backgroundColor: isBackendOnline ? "var(--accent-emerald)" : "var(--accent-amber)" }} />
              <span style={{ color: isBackendOnline ? "var(--accent-emerald)" : "var(--accent-amber)", fontWeight: 500 }}>
                {isBackendOnline ? "FastAPI Online (:8000)" : "Client Mode (Local)"}
              </span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav style={{ display: "flex", gap: "0.5rem", marginTop: "1rem", borderTop: "1px solid var(--border-subtle)", paddingTop: "0.75rem", overflowX: "auto" }}>
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onSelectTab(tab.id)}
                style={{
                  padding: "0.45rem 0.95rem",
                  fontSize: "0.85rem",
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? "#ffffff" : "var(--text-secondary)",
                  background: isActive ? "var(--bg-surface-elevated)" : "transparent",
                  border: isActive ? "1px solid var(--border-highlight)" : "1px solid transparent",
                  borderRadius: 8,
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                  whiteSpace: "nowrap",
                }}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
