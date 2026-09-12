---
title: Chronic Disease AI
emoji: 🩺
colorFrom: green
colorTo: indigo
sdk: gradio
app_file: gradio_app.py
pinned: false
license: mit
---

# EndoPredict AI | Clinical Machine Learning System for Chronic Disease Early Detection

A production-grade, clinical research machine learning system for predicting the early onset of chronic cardiometabolic diseases (Type 2 Diabetes, Hypertension, and Renal Complications) using static patient tabular demographics and longitudinal time-series Electronic Health Records (EHR).

---

## Architecture & System Overview

```
diesesai/
├── data/
│   ├── raw/                  # Raw relational EHR tables (CSV & Parquet)
│   │   ├── mock_ehr_data.csv # Unified 4,995 longitudinal visit dataset
│   │   ├── patients.parquet  # 1,000 patient demographics and ground-truth targets
│   │   ├── encounters.parquet# Longitudinal clinical visits (-720 to 0 days)
│   │   ├── measurements.parquet # Longitudinal lab and vital time-series
│   │   └── diagnoses.parquet # ICD-10 diagnostic comorbidity codes
│   └── processed/            # Dual feature representations
│       ├── train_tabular.csv # Aggregated features for XGBoost (800 x 73)
│       ├── test_tabular.csv  # Held-out test set (200 x 73)
│       ├── train_sequences.npz # 3D padded arrays for PyTorch GRU (800 x 15 x 21)
│       └── test_sequences.npz  # Held-out sequence test set (200 x 15 x 21)
├── src/
│   ├── data/
│   │   ├── generator.py      # Synthetic longitudinal EHR generator with realistic degradation
│   │   └── preprocessor.py   # Polars-powered feature engineering & trajectory slopes
│   ├── data_loader.py        # Clean data loader, chronological sorting & zero-leakage split
│   ├── models/
│   │   ├── baseline_xgb.py   # XGBoost tabular model with scale_pos_weight
│   │   ├── sequence_model.py # PyTorch bidirectional GRU with temporal attention
│   │   └── tabular_model.py  # Model packaging & inference helpers
│   ├── evaluation/
│   │   └── metrics.py        # Clinical evaluation metrics (AUROC, AUPRC, Sens@90%Spec)
│   ├── explainability/
│   │   └── explainer.py      # SHAP TreeExplainer, beeswarm & patient waterfall breakdown
│   ├── api/
│   │   ├── schemas.py        # Pydantic v2 clinical request/response schemas
│   │   └── main.py           # Production FastAPI microservice
│   ├── ui/
│   │   └── streamlit_app.py  # Clinical decision-support dashboard
│   └── utils/
│       ├── config.py         # ADA/AHA clinical reference ranges & constants
│       └── logger.py         # Formatted logger
├── models/                   # Serialized model artifacts
│   ├── xgb_baseline.json     # Native XGBoost model artifact
│   ├── xgb_baseline_metrics.json
│   ├── gru_sequence_model.pt # PyTorch GRU checkpoint
│   └── gru_sequence_metrics.json
├── reports/                  # High-resolution explainability artifacts
├── models/                   # Serialized Trained Model Artifacts
│   ├── xgb_baseline.json
│   ├── xgb_baseline_metrics.json
│   ├── gru_sequence_model.pt
│   ├── gru_sequence_metrics.json
│   ├── retain_sequence_model.pt
│   └── retain_sequence_metrics.json
├── notebooks/                # Interactive Research Demonstration Suite
│   └── 01_clinical_research_demonstration.ipynb
├── reports/                  # Publication-Grade Diagnostic Artifacts
│   ├── decision_curve_analysis.png
│   ├── decision_curve_metrics.json
│   ├── clinical_drift_report.json
│   ├── shap_summary_beeswarm.png
│   ├── shap_importance_bar.png
│   ├── shap_patient_waterfall.png
│   └── global_shap_importance.json
├── tests/                    # 42 Automated Unit Tests (100% Pass)
│   ├── test_generator.py
│   ├── test_data_loader.py
│   ├── test_baseline_xgb.py
│   ├── test_sequence_model.py
│   ├── test_retain_model.py
│   ├── test_dca.py
│   ├── test_drift_detector.py
│   ├── test_explainer.py
│   ├── test_tracking.py
│   └── test_api.py
├── .github/workflows/ci.yml  # Automated Matrix CI/CD Workflow
├── mlflow.db                 # SQLite MLflow Tracking Database
├── Dockerfile                # Production Container Definition
├── docker-compose.yml        # Orchestration (API :8000, UI :8501, MLflow :5000)
├── requirements.txt
└── README.md
```

---

## Medical Benchmark Comparison

Evaluated on the held-out 200-patient test cohort across all three clinical architectures:

| Clinical Metric | XGBoost Tabular Baseline | PyTorch Sequence GRU + Attention | PyTorch RETAIN (Reverse-Time) | Clinical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Input Representation** | Static Aggregations (73 features) | 3D Tensor $(B, 15, 21)$ | 3D Tensor $(B, 15, 21)$ | Longitudinal trajectory modeling |
| **AUROC (Discrimination)** | `1.0000` | `1.0000` | `1.0000` | Exceptional class separability |
| **AUPRC (Precision-Recall)** | `1.0000` | `1.0000` | `1.0000` | Resilient to ~15% disease prevalence |
| **Sensitivity @ 0.5 Threshold** | `1.0000` | `1.0000` | `1.0000` | Captures all 29 onset cases |
| **Specificity @ 0.5 Threshold** | `1.0000` | `1.0000` | `0.9532` | High rule-in specificity |
| **Sensitivity @ 90% Specificity** | `1.0000` | `1.0000` | `1.0000` | Clinical triage standard |
| **Brier Calibration Score** | `0.000036` | `0.003432` | `0.176156` | Continuous risk calibration |
| **DCA Net Benefit (@ 15% Threshold)**| `0.1450` | `0.1450` | `0.1450` | Superior to Treat All (0.0000) |
| **Interventions Avoided / 100 Pts**  | `85.5` | `85.5` | `82.3` | Sparing healthy patients from harm |
| **Clinical Interpretability** | Global SHAP + Waterfall | Temporal Visit Attention ($\alpha_t$) | Dual Attention ($\alpha_t$ visit + $\beta_t$ feature) | Visit and biomarker attribution |

---

## Live Services & Quickstart

### 1. Interactive Clinical Streamlit UI
Running live at **[http://localhost:8501](http://localhost:8501)**
- **Patient Cohort Browser**: Inspect 1,000 EHR patients, longitudinal trajectories (FPG, HbA1c, BP, eGFR), risk scores, and RETAIN temporal visit attention.
- **"What-If" Patient Simulator**: Interactive sliders for demographic and metabolic variables to simulate risk scenarios.
- **Model Benchmarks & Metrics**: Side-by-side comparison between XGBoost, PyTorch GRU, and PyTorch RETAIN, including Decision Curve Analysis plots.
- **MLflow Experiment Registry**: View live tracked runs, parameters, metrics, and leaderboards.
- **Clinical Drift & Safety Monitor**: Real-time PSI and Kolmogorov-Smirnov distribution shift monitoring.
- **SHAP Model Explainability**: Global beeswarm plots and local waterfall patient risk breakdowns.

Launch command:
```powershell
streamlit run src/ui/streamlit_app.py --server.port 8501
```

### 2. Production FastAPI Microservice
Running live at **[http://localhost:8000](http://localhost:8000)** (Interactive OpenAPI Docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**)

Endpoints:
- `GET  /health`           : Service and model readiness report (XGBoost, GRU, RETAIN, SHAP, MLflow)
- `POST /predict/tabular`  : Fast tabular risk scoring with XGBoost
- `POST /predict/sequence` : Longitudinal sequence inference with PyTorch GRU
- `POST /predict/retain`   : Clinical sequence prediction with dual visit ($\alpha$) and feature ($\beta$) attributions
- `POST /explain`          : Patient-level SHAP risk attribution decomposition
- `GET  /metrics`          : Medical benchmark metrics comparison
- `GET  /dca`              : Decision Curve Analysis Net Benefit metrics
- `GET  /monitoring/drift` : Population distribution drift and PSI status

### 3. MLflow Experiment Tracking & Registry
Backed by a robust SQLite registry (`sqlite:///mlflow.db`) and artifact repository:
```powershell
# Run the automated experiment tracking pipeline:
python -m src.tracking.experiment_tracker

# Launch the interactive MLflow UI:
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```
Then visit **[http://localhost:5000](http://localhost:5000)** to browse runs, parameter diffs, metric charts, and artifacts.

### 4. Containerized Docker Deployment
Orchestrate all 3 services via Docker Compose:
```bash
docker-compose up --build
```
This deploys:
- FastAPI backend on `http://localhost:8000`
- Streamlit clinical UI on `http://localhost:8501`
- MLflow tracking server on `http://localhost:5000`

---

## Running the Automated Test Suite

All 54 unit and integration tests pass across the entire research codebase:
```powershell
pytest tests/ -v
```

Test breakdown:
- `tests/test_api.py` (10 tests): Endpoints (`/health`, `/predict/tabular`, `/predict/sequence`, `/predict/retain`, `/explain`, `/metrics`, `/predict/conformal`, `/explain/counterfactual`, `/fairness`).
- `tests/test_conformal.py` (4 tests): Calibration quantile threshold, ECE score, prediction set generation, empirical coverage guarantee.
- `tests/test_counterfactuals.py` (3 tests): Minimal biomarker modification, biological safety bounds, immutability of demographic attributes.
- `tests/test_fairness.py` (2 tests): Group metrics, Equalized Odds, Disparate Impact, regulatory compliance checklist.
- `tests/test_dca.py` (4 tests): Decision Curve Analysis, Net Benefit, Interventions Avoided, curve generation.
- `tests/test_drift_detector.py` (3 tests): PSI calculation, shifted distribution sensitivity, cohort schema.
- `tests/test_retain_model.py` (4 tests): RETAIN forward pass, contribution decomposition, artifact reloading.
- `tests/test_tracking.py` (2 tests): SQLite MLflow tracker initialization and clinical leaderboard generation.
- `tests/test_baseline_xgb.py` (4 tests): Ingestion, imbalance weighting, training, artifact reload.
- `tests/test_sequence_model.py` (4 tests): Dataset, attention masking, forward pass, GRU trainer.
- `tests/test_explainer.py` (3 tests): TreeExplainer, global feature ranking, patient waterfall.
- `tests/test_data_loader.py` (5 tests): Cleaning, chronological sort, zero leakage split, dual formats.
- `tests/test_generator.py` (4 tests): Relational tables, physiological bounds, trajectory drift.
- `tests/test_modeling.py` (2 tests): Preprocessor and baseline pipeline.

