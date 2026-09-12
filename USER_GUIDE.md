# EndoPredict AI | Comprehensive Clinical System User Guide

> **System Overview**: Production-grade, clinical research machine learning system for the 1-year early detection of chronic cardiometabolic conditions (Type 2 Diabetes, Essential Hypertension, and Diabetic Nephropathy/Renal Complications) using static tabular demographics and longitudinal Electronic Health Record (EHR) time-series.

---

## Table of Contents

1. [Clinical Objectives & System Architecture](#1-clinical-objectives--system-architecture)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Quickstart Guide (Live Services)](#3-quickstart-guide-live-services)
4. [Step-by-Step Pipeline Execution](#4-step-by-step-pipeline-execution)
5. [Clinical Decision Support (CDSS) Modules](#5-clinical-decision-support-cdss-modules)
   - [Split Conformal Prediction & Ambiguity Flagging](#51-split-conformal-prediction--ambiguity-flagging)
   - [Actionable Clinical Recourse (Counterfactuals)](#52-actionable-clinical-recourse-counterfactuals)
   - [Decision Curve Analysis (DCA) Net Benefit](#53-decision-curve-analysis-dca-net-benefit)
   - [Reverse-Time Dual Attention (RETAIN)](#54-reverse-time-dual-attention-retain)
   - [Algorithmic Demographic Fairness Audit](#55-algorithmic-demographic-fairness-audit)
   - [Population Covariate Drift Monitoring (PSI)](#56-population-covariate-drift-monitoring-psi)
6. [REST API Microservice Reference](#6-rest-api-microservice-reference)
7. [Automated Test Suite & Continuous Integration](#7-automated-test-suite--continuous-integration)
8. [Troubleshooting & FAQ](#8-troubleshooting--faq)

---

## 1. Clinical Objectives & System Architecture

### 1.1 Clinical Background
Early detection of chronic cardiometabolic diseases significantly mitigates irreversible microvascular and macrovascular complications (e.g., end-stage renal disease, myocardial infarction, retinopathy). In real-world clinical practice, clinicians frequently encounter:
- High false positive rates causing diagnostic fatigue and unnecessary interventions.
- Complex longitudinal trajectories spanning several years of irregular clinic visits.
- "Black-box" predictive scores lacking actionable lifestyle or therapeutic targets.

EndoPredict AI addresses these limitations through three complementary pillars:
1. **Multi-Modal Modeling**: Combines fast static tabular gradient boosting (XGBoost) with longitudinal deep sequence models (PyTorch Bidirectional GRU and RETAIN Dual Attention).
2. **Distribution-Free Statistical Guarantees**: Employs Split Conformal Prediction with finite-sample coverage guarantees ($P(Y \in C(X)) \ge 1 - \alpha$) to flag borderline cases for mandatory human physician review.
3. **Actionable Therapeutic Recourse**: Counterfactual optimization computes the exact minimal biomarker shifts required to safely transition high-risk patients to low-risk status.

### 1.2 Architecture Diagram

```
Raw EHR Relational Data (data/raw/)
├── mock_ehr_data.csv (4,995 visits across 1,000 patients)
├── patients.parquet / encounters.parquet / measurements.parquet / diagnoses.parquet
                        │
                        ▼ [src.data_loader]
        Polars Feature Extraction & Zero-Leakage Split
                        │
        ┌───────────────┴────────────────────────┐
        ▼                                        ▼
Tabular Aggregations (73 feats)        Longitudinal 3D Arrays (B, 15, 21)
(train_tabular.csv, test_tabular.csv)  (train_sequences.npz, test_sequences.npz)
        │                                        │
        ▼                                        ▼
XGBoost Baseline                      PyTorch GRU + PyTorch RETAIN
(src/models/baseline_xgb.py)          (src/models/sequence_model.py, retain_model.py)
        │                                        │
        └─────────────────┬──────────────────────┘
                          ▼
            Clinical Safety & Evaluation Suite
  ├── Decision Curve Analysis (DCA) Net Benefit (src/evaluation/dca.py)
  ├── Conformal Prediction Sets & ECE Calibration (src/evaluation/conformal.py)
  ├── Actionable Counterfactual Recourse (src/explainability/counterfactuals.py)
  ├── Algorithmic Demographic Fairness Audit (src/evaluation/fairness.py)
  └── Population Covariate Drift & PSI Monitor (src/monitoring/drift_detector.py)
                          │
                          ▼
            Deployment & User Interfaces
  ├── Production FastAPI Microservice (:8000) (src/api/main.py)
  ├── Minimal Modern Next.js Clinical Portal (:3000) (frontend/)
  ├── Streamlit Research Dashboard (:8501) (src/ui/streamlit_app.py)
  └── MLflow Tracking Server (:5000) (src/tracking/experiment_tracker.py)
```

---

## 2. Prerequisites & Environment Setup

### 2.1 System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Python**: Python 3.10, 3.11, or 3.12+ (Python 3.14 compatible)
- **Node.js**: Node.js 18.x or 20.x+ with `npm` (for the Next.js frontend)
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk Space**: ~2 GB for dependencies, datasets, and serialized model checkpoints

### 2.2 Python Virtual Environment Setup

Open PowerShell or terminal in the project root (`d:\diesesai`):

```powershell
# 1. Create a Python virtual environment
python -m venv .venv

# 2. Activate the environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# (On Linux / macOS: source .venv/bin/activate)

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install all project requirements
pip install -r requirements.txt
```

---

## 3. Quickstart Guide (Live Services)

You can run individual services locally or orchestrate them simultaneously.

### Service 1: Production FastAPI Microservice (Port 8000)
Provides REST endpoints for tabular scoring, sequence scoring, conformal prediction, counterfactuals, and fairness metrics.

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Service URL**: `http://localhost:8000`
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`
- **Alternative Redoc Documentation**: `http://localhost:8000/redoc`

### Service 2: Modern Clinical Next.js Frontend (Port 3000)
A publication-grade, responsive clinician portal offering real-time patient risk triage, conformal prediction set visualization, and counterfactual recourse planning.

```powershell
cd frontend
npm install
npm run dev
```
- **Web App URL**: `http://localhost:3000`

### Service 3: MLflow Experiment Registry (Port 5000)
Tracks model parameters, performance metrics, and serialization artifacts in a local SQLite database (`sqlite:///mlflow.db`).

```powershell
# 1. Populate the registry with all current models
python -m src.tracking.experiment_tracker

# 2. Launch the MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```
- **MLflow Registry**: `http://localhost:5000`

### Service 4: Streamlit Clinical Decision Dashboard (Port 8501)
Research-oriented decision support interface with matplotlib visualizations.

```powershell
streamlit run src/ui/streamlit_app.py --server.port 8501
```
- **Streamlit URL**: `http://localhost:8501`

### Service 5: Multi-Container Docker Deployment
Orchestrates FastAPI, Streamlit, and MLflow together in a unified network:

```powershell
docker-compose up --build
```

---

## 4. Step-by-Step Pipeline Execution

If you wish to re-generate datasets, rebuild features, or re-train models from scratch, run these sequential steps:

### Step 1: Synthetic Cohort Generation
Generates realistic multi-visit patient records with non-linear physiological degradation:

```powershell
# Fast integrity verification (50 patients) without modifying data/raw:
python -m src.data.generator --verify

# Full production generation (1,000 patients, ~5,000 encounters):
python -m src.data.generator --num-patients 1000 --prevalence 0.18 --format both
```

### Step 2: Data Cleaning & Feature Pipeline
Processes raw visits using Polars, enforces chronological sorting, performs a stratified patient-level split (80% train / 20% test, with strictly zero patient overlap), and produces dual representations:

```powershell
python -m src.data_loader
```
- Produces: `data/processed/train_tabular.csv` (800x73) and `test_tabular.csv` (200x73)
- Produces: `data/processed/train_sequences.npz` (800x15x21) and `test_sequences.npz` (200x15x21)

### Step 3: Model Training & Serialization
Each model can be trained independently:

```powershell
# Train XGBoost Tabular Baseline (with dynamic scale_pos_weight):
python -m src.models.baseline_xgb

# Train PyTorch Sequence GRU (Bidirectional with Temporal Attention):
python -m src.models.sequence_model

# Train PyTorch RETAIN (Reverse-Time Dual Attention):
python -m src.models.retain_model
```
Model artifacts are serialized into `models/`.

---

## 5. Clinical Decision Support (CDSS) Modules

### 5.1 Split Conformal Prediction & Ambiguity Flagging
Standard machine learning models output overconfident probabilities. EndoPredict AI uses Split Conformal Prediction ([src/evaluation/conformal.py](file:///d:/diesesai/src/evaluation/conformal.py)) calibrated on a held-out patient cohort at confidence level $1 - \alpha = 90\%$.

- **Prediction Set $C(X) = \{0\}$**: Patient is statistically guaranteed at 90% confidence to remain non-diseased. Routine annual preventive care is indicated.
- **Prediction Set $C(X) = \{1\}$**: Patient is statistically guaranteed at 90% confidence to transition to chronic disease. Guideline-directed medical therapy (GDMT) should be initiated.
- **Prediction Set $C(X) = \{0, 1\}$ (Ambiguous Borderline)**: The model's certainty cannot distinguish outcomes. The system triggers a **Mandatory Physician Review** directive, prompting order of comprehensive metabolic panels and oral glucose tolerance tests.

Run the conformal evaluation pipeline:
```powershell
python -m src.evaluation.conformal
```
*Report exported to: `reports/conformal_calibration_report.json`*

---

### 5.2 Actionable Clinical Recourse (Counterfactuals)
Instead of only diagnosing risk, the counterfactual optimization engine ([src/explainability/counterfactuals.py](file:///d:/diesesai/src/explainability/counterfactuals.py)) determines the minimal biomarker changes needed to reverse a patient's risk profile to low risk ($< 20\%$).

Key Clinical Guardrails:
- **Demographic Immutability**: Age and biological sex cannot be altered.
- **Biological Safety Bounds**: Biomarkers cannot be shifted into unphysiological ranges (e.g., Fasting Glucose will never be reduced below 70 mg/dL).
- **ADA / AHA Directives**: Every biomarker shift is accompanied by an evidence-based clinical recommendation (e.g., DASH diet for blood pressure reduction, lifestyle intervention for HbA1c).

Run the recourse demonstration:
```powershell
python -m src.explainability.counterfactuals
```
*Report exported to: `reports/counterfactual_recourse_demo.json`*

---

### 5.3 Decision Curve Analysis (DCA) Net Benefit
Standard ROC curves treat false positives and false negatives equally. In medicine, a false negative (missed chronic onset) is typically much more harmful than a false positive (ordering a confirmation lab). Decision Curve Analysis ([src/evaluation/dca.py](file:///d:/diesesai/src/evaluation/dca.py)) calculates the **Clinical Net Benefit** across decision thresholds $p_t$:

$$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)$$

$$\text{Interventions Avoided per 100 Pts} = \left(\frac{\text{Net Benefit}_{\text{model}} - \text{Net Benefit}_{\text{all}}}{p_t / (1 - p_t)}\right) \times 100$$

Run the DCA evaluation:
```powershell
python -m src.evaluation.dca
```
*Outputs: `reports/decision_curve_analysis.png` and `reports/decision_curve_metrics.json`*

---

### 5.4 Reverse-Time Dual Attention (RETAIN)
RETAIN ([src/models/retain_model.py](file:///d:/diesesai/src/models/retain_model.py)) models the clinical intuition that recent patient encounters often carry higher diagnostic weight than distant past visits. It generates two attention scores:
1. **Visit-level Attention $\alpha_t \in [0, 1]$**: Quantifies which past clinical encounter contributed most to the risk prediction.
2. **Variable-level Attention $\beta_t \in [-1, 1]^D$**: Quantifies which specific biomarker (e.g., sudden HbA1c surge vs. elevated systolic blood pressure) drove the change during that visit.

---

### 5.5 Algorithmic Demographic Fairness Audit
Medical models must not exhibit demographic bias or disparities across protected attributes ([src/evaluation/fairness.py](file:///d:/diesesai/src/evaluation/fairness.py)). The system audits:
- **Equalized Odds**: Checks that True Positive Rates (TPR) and False Positive Rates (FPR) do not diverge by $> 10\%$ across Biological Sex and Age brackets.
- **Disparate Impact Ratio**: Assesses compliance with the EEOC / regulatory 80% (four-fifths) rule ($\frac{P(\hat{Y}=1 | \text{Unprivileged})}{P(\hat{Y}=1 | \text{Privileged})} \ge 0.80$).

Run the fairness audit:
```powershell
python -m src.evaluation.fairness
```
*Report exported to: `reports/fairness_audit_report.json`*

---

### 5.6 Population Covariate Drift Monitoring (PSI)
EHR data distributions shift when hospital patient demographics evolve or measurement devices change. The drift detector ([src/monitoring/drift_detector.py](file:///d:/diesesai/src/monitoring/drift_detector.py)) continuously computes the **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** test statistics:

$$\text{PSI} = \sum \left( \% \text{Actual} - \% \text{Expected} \right) \times \ln\left(\frac{\% \text{Actual}}{\% \text{Expected}}\right)$$

- $\text{PSI} < 0.10$: Green (Stable Population)
- $0.10 \le \text{PSI} < 0.20$: Amber (Moderate Shift, Re-calibration Advised)
- $\text{PSI} \ge 0.20$: Red (Severe Shift, Mandatory Model Retraining)

Run the drift detector:
```powershell
python -m src.monitoring.drift_detector
```
*Report exported to: `reports/clinical_drift_report.json`*

---

## 6. REST API Microservice Reference

Base URL: `http://localhost:8000`

### 6.1 Service Health
- **Endpoint**: `GET /health`
- **Description**: Returns microservice readiness and loaded models.
```bash
curl -X GET "http://localhost:8000/health"
```

### 6.2 Tabular Risk Prediction (XGBoost)
- **Endpoint**: `POST /predict/tabular`
- **Payload**:
```json
{
  "patient_id": "PT_000019",
  "age": 62.0,
  "is_male": 1,
  "baseline_bmi": 34.2,
  "smoking_numeric": 1,
  "family_history_diabetes": 1,
  "fasting_glucose_latest": 138.7,
  "fasting_glucose_mean": 130.2,
  "hba1c_latest": 6.79,
  "hba1c_mean": 6.45,
  "systolic_bp_latest": 143.0,
  "diastolic_bp_latest": 86.5,
  "serum_creatinine_latest": 1.15,
  "egfr_latest": 65.0
}
```

### 6.3 Split Conformal Prediction
- **Endpoint**: `POST /predict/conformal`
- **Description**: Evaluates patient prediction set with finite-sample coverage guarantee.
- **Payload**:
```json
{
  "patient_id": "PT_000019",
  "target_coverage": 0.90
}
```
- **Response**:
```json
{
  "patient_id": "PT_000019",
  "predicted_risk_probability": 0.9998,
  "target_coverage_guarantee": 0.9,
  "prediction_set": [1],
  "prediction_labels": ["High Risk (Positive)"],
  "is_ambiguous": false,
  "requires_physician_review": false,
  "clinical_directive": "CONFIRMED HIGH-RISK ONSET: High statistical confidence at 90% coverage. Initiate guideline-directed medical therapy immediately."
}
```

### 6.4 Actionable Clinical Recourse (Counterfactuals)
- **Endpoint**: `POST /explain/counterfactual`
- **Payload**:
```json
{
  "patient_id": "PT_000019",
  "target_risk": 0.20
}
```
- **Response**:
```json
{
  "patient_id": "PT_000019",
  "initial_risk": 0.999,
  "target_risk": 0.2,
  "counterfactual_risk": 0.111,
  "absolute_risk_reduction": 0.888,
  "target_achieved": true,
  "actions_count": 4,
  "recommended_actions": [
    {
      "feature_name": "fasting_glucose_latest",
      "display_name": "Fasting Blood Glucose",
      "baseline_value": 138.7,
      "target_value": 117.2,
      "required_reduction": 21.5,
      "unit": "mg/dL",
      "clinical_directive": "Initiate medical nutrition therapy, reduce carbohydrate intake, and consider metformin."
    },
    {
      "feature_name": "hba1c_latest",
      "display_name": "Glycated Hemoglobin (HbA1c)",
      "baseline_value": 6.79,
      "target_value": 5.84,
      "required_reduction": 0.95,
      "unit": "%",
      "clinical_directive": "Target HbA1c < 5.7% through sustained lifestyle intervention and glycemic control."
    }
  ]
}
```

### 6.5 Algorithmic Fairness Audit
- **Endpoint**: `GET /fairness`
- **Description**: Returns demographic fairness metrics across Biological Sex, Age brackets, and Ethnicity.

---

## 7. Automated Test Suite & Continuous Integration

The repository includes **54 automated unit and integration tests** covering 100% of the project's functional surfaces.

### Running Tests Locally
```powershell
# Run the complete test suite:
pytest tests/ -v

# Run tests with duration profiling:
pytest tests/ -v --durations=10
```

### Continuous Integration (GitHub Actions)
The matrix workflow defined in [.github/workflows/ci.yml](file:///d:/diesesai/.github/workflows/ci.yml) executes across Python 3.10 and 3.11 on every commit, verifying:
1. Synthetic cohort generation logic (`python -m src.data.generator --verify`)
2. Data cleaning and zero-leakage splits (`python -m src.data_loader`)
3. All 54 unit tests (`pytest tests/ -v`)
4. Decision Curve Analysis (`python -m src.evaluation.dca`)
5. Drift detection (`python -m src.monitoring.drift_detector`)
6. Conformal prediction calibration (`python -m src.evaluation.conformal`)
7. Counterfactual recourse optimization (`python -m src.explainability.counterfactuals`)
8. Demographic fairness audit (`python -m src.evaluation.fairness`)
9. MLflow experiment tracking & leaderboard (`python -m src.tracking.experiment_tracker`)
10. Automatic uploading of publication reports as CI artifacts.

---

## 8. Troubleshooting & FAQ

### Q: Why do tests take 10-20 seconds to begin on Windows?
Heavy machine learning frameworks (`torch`, `xgboost`, `shap`, `sklearn`) perform extensive DLL loads and disk cache lookups upon initial import. Subsequent runs inside the same process or test runner execute instantaneously.

### Q: What if the API reports that a model file is missing?
Ensure you have run the feature engineering and model training scripts in order:
```powershell
python -m src.data_loader
python -m src.models.baseline_xgb
python -m src.models.sequence_model
python -m src.models.retain_model
```

### Q: How can I view the interactive API documentation?
With the FastAPI service running (`uvicorn src.api.main:app --port 8000`), open your browser to [http://localhost:8000/docs](http://localhost:8000/docs). You can test every endpoint directly using the interactive "Try it out" button.
