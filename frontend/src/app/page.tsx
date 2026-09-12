"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Header } from "../components/Header";
import { PatientBanner } from "../components/PatientBanner";
import { TriageCard } from "../components/TriageCard";
import { CounterfactualRecourseView } from "../components/CounterfactualRecourse";
import { TrajectoryTimeline } from "../components/TrajectoryTimeline";
import { FairnessAndDriftView } from "../components/FairnessAndDrift";
import { ModelBenchmarksView } from "../components/ModelBenchmarks";
import { PatientIntakeForm } from "../components/PatientIntakeForm";

import { PatientProfile, RiskPrediction, ConformalPrediction, CounterfactualRecourse } from "../types/clinical";
import { MOCK_PATIENTS } from "../data/mockPatients";
import {
  checkApiHealth,
  fetchTabularPrediction,
  fetchConformalPrediction,
  fetchCounterfactualRecourse,
} from "../lib/api";

export default function ClinicalPortalPage() {
  const [selectedPatient, setSelectedPatient] = useState<PatientProfile>(MOCK_PATIENTS[0]);
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<string>("triage");
  const [targetRisk, setTargetRisk] = useState<number>(0.20);

  const [prediction, setPrediction] = useState<RiskPrediction>({
    patient_id: MOCK_PATIENTS[0].patient_id,
    predicted_risk_probability: 0.9998,
    risk_tier: "Critical Risk (Imminent Onset)",
    triage_color: "#ef4444",
    clinical_action: "Immediate clinician consultation: initiate guideline-directed medical therapy and active cardiometabolic surveillance.",
    guideline_recommendations: [
      "Urgent endocrine or PCP consultation within 14 days",
      "Initiate guideline-directed pharmacotherapy",
      "Order comprehensive renal microalbuminuria screening",
      "Prescribe continuous or frequent glucose self-monitoring",
    ],
    model_used: "XGBoost Tabular Baseline",
  });

  const [conformal, setConformal] = useState<ConformalPrediction>({
    patient_id: MOCK_PATIENTS[0].patient_id,
    predicted_risk_probability: 0.9998,
    target_coverage_guarantee: 0.90,
    prediction_set: [1],
    prediction_labels: ["High Risk (Positive)"],
    is_ambiguous: false,
    requires_physician_review: false,
    clinical_directive: "CONFIRMED HIGH-RISK ONSET: High statistical confidence at 90% coverage. Initiate guideline-directed medical therapy immediately.",
  });

  const [recourse, setRecourse] = useState<CounterfactualRecourse>({
    patient_id: MOCK_PATIENTS[0].patient_id,
    initial_risk: 0.999,
    target_risk: 0.20,
    counterfactual_risk: 0.111,
    absolute_risk_reduction: 0.888,
    target_achieved: true,
    recourse_needed: true,
    actions_count: 4,
    recommended_actions: [
      {
        feature_name: "fasting_glucose_latest",
        display_name: "Fasting Blood Glucose",
        baseline_value: 138.7,
        target_value: 117.2,
        required_reduction: 21.5,
        unit: "mg/dL",
        clinical_directive: "Initiate medical nutrition therapy, reduce refined carbohydrate intake, and consider metformin.",
      },
      {
        feature_name: "hba1c_latest",
        display_name: "Glycated Hemoglobin (HbA1c)",
        baseline_value: 6.79,
        target_value: 5.84,
        required_reduction: 0.95,
        unit: "%",
        clinical_directive: "Target HbA1c < 5.7% through sustained lifestyle intervention and glycemic self-monitoring.",
      },
      {
        feature_name: "systolic_bp_latest",
        display_name: "Systolic Blood Pressure",
        baseline_value: 143.0,
        target_value: 126.2,
        required_reduction: 16.8,
        unit: "mmHg",
        clinical_directive: "Adopt DASH diet, restrict dietary sodium (<2g/day), and optimize antihypertensive therapy.",
      },
      {
        feature_name: "diastolic_bp_latest",
        display_name: "Diastolic Blood Pressure",
        baseline_value: 86.5,
        target_value: 79.6,
        required_reduction: 6.9,
        unit: "mmHg",
        clinical_directive: "Complement systolic BP reduction with regular aerobic exercise (150 min/week).",
      },
    ],
    message: "Risk successfully reduced from 99.9% to 11.1% (88.8% absolute reduction).",
  });

  // Health check on mount
  useEffect(() => {
    async function checkHealth() {
      const health = await checkApiHealth();
      setIsBackendOnline(health.isOnline);
    }
    checkHealth();
  }, []);

  // Fetch predictions on patient or target risk change
  const loadPatientData = useCallback(async (patient: PatientProfile, riskTarget: number) => {
    const [predRes, confRes, recRes] = await Promise.all([
      fetchTabularPrediction(patient),
      fetchConformalPrediction(patient.patient_id),
      fetchCounterfactualRecourse(patient.patient_id, riskTarget),
    ]);

    setPrediction(predRes);
    setConformal(confRes);
    setRecourse(recRes);
  }, []);

  useEffect(() => {
    loadPatientData(selectedPatient, targetRisk);
  }, [selectedPatient, targetRisk, loadPatientData]);

  const handlePatientSelect = (p: PatientProfile) => {
    setSelectedPatient(p);
  };

  const handleTargetRiskChange = (newTarget: number) => {
    setTargetRisk(newTarget);
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <Header
        selectedPatient={selectedPatient}
        onSelectPatient={handlePatientSelect}
        isBackendOnline={isBackendOnline}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
      />

      {/* Main Content Area */}
      <main style={{ flex: 1, maxWidth: 1280, width: "100%", margin: "0 auto", padding: "1.5rem" }}>
        {/* Patient Demographic Banner */}
        <PatientBanner patient={selectedPatient} />

        {/* Tab Specific Content */}
        {activeTab === "triage" && (
          <TriageCard
            patient={selectedPatient}
            prediction={prediction}
            conformal={conformal}
          />
        )}

        {activeTab === "intake" && (
          <PatientIntakeForm
            onPatientEvaluated={(newPatient) => {
              setSelectedPatient(newPatient);
              setActiveTab("triage");
            }}
          />
        )}

        {activeTab === "recourse" && (
          <CounterfactualRecourseView
            patient={selectedPatient}
            recourse={recourse}
            onUpdateTargetRisk={handleTargetRiskChange}
          />
        )}

        {activeTab === "trajectory" && (
          <TrajectoryTimeline patient={selectedPatient} />
        )}

        {activeTab === "fairness" && (
          <FairnessAndDriftView />
        )}

        {activeTab === "benchmarks" && (
          <ModelBenchmarksView />
        )}
      </main>

      {/* Minimal Footer */}
      <footer
        style={{
          borderTop: "1px solid var(--border-subtle)",
          padding: "1.25rem 1.5rem",
          marginTop: "2rem",
          background: "var(--bg-surface)",
        }}
      >
        <div
          style={{
            maxWidth: 1280,
            margin: "0 auto",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "1rem",
            fontSize: "0.75rem",
            color: "var(--text-muted)",
          }}
        >
          <div>
            <strong style={{ color: "var(--text-secondary)" }}>EndoPredict AI v2.4.0</strong> &bull; ADA / AHA Aligned Clinical CDSS
          </div>
          <div>
            For clinical research & decision-support simulation. Not an autonomous substitute for physician diagnostic judgment.
          </div>
        </div>
      </footer>
    </div>
  );
}
