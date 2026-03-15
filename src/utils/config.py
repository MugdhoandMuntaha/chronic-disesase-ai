"""Centralized configuration, paths, clinical physiological constants, and disease definitions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class ClinicalReferenceRange:
    """Represents standard clinical laboratory and vital sign physiological reference ranges."""
    name: str
    unit: str
    normal_min: float
    normal_max: float
    critical_min: float
    critical_max: float
    clinical_description: str


# Standard clinical physiology definitions based on ADA and AHA guidelines
CLINICAL_MEASUREMENT_RANGES: Dict[str, ClinicalReferenceRange] = {
    "fasting_glucose": ClinicalReferenceRange(
        name="Fasting Plasma Glucose",
        unit="mg/dL",
        normal_min=70.0,
        normal_max=99.0,
        critical_min=40.0,
        critical_max=400.0,
        clinical_description="Normal: 70-99 mg/dL; Prediabetes: 100-125 mg/dL; Diabetes: >=126 mg/dL"
    ),
    "hba1c": ClinicalReferenceRange(
        name="Hemoglobin A1c",
        unit="%",
        normal_min=4.0,
        normal_max=5.6,
        critical_min=3.5,
        critical_max=16.0,
        clinical_description="Normal: <5.7%; Prediabetes: 5.7-6.4%; Diabetes: >=6.5%"
    ),
    "systolic_bp": ClinicalReferenceRange(
        name="Systolic Blood Pressure",
        unit="mmHg",
        normal_min=90.0,
        normal_max=120.0,
        critical_min=60.0,
        critical_max=240.0,
        clinical_description="Normal: <120 mmHg; Elevated: 120-129; Stage 1: 130-139; Stage 2: >=140"
    ),
    "diastolic_bp": ClinicalReferenceRange(
        name="Diastolic Blood Pressure",
        unit="mmHg",
        normal_min=60.0,
        normal_max=80.0,
        critical_min=40.0,
        critical_max=140.0,
        clinical_description="Normal: <80 mmHg; Stage 1: 80-89; Stage 2: >=90"
    ),
    "heart_rate": ClinicalReferenceRange(
        name="Heart Rate",
        unit="bpm",
        normal_min=60.0,
        normal_max=100.0,
        critical_min=35.0,
        critical_max=180.0,
        clinical_description="Normal resting heart rate: 60-100 bpm"
    ),
    "bmi": ClinicalReferenceRange(
        name="Body Mass Index",
        unit="kg/m²",
        normal_min=18.5,
        normal_max=24.9,
        critical_min=12.0,
        critical_max=65.0,
        clinical_description="Normal: 18.5-24.9; Overweight: 25.0-29.9; Obese: >=30.0"
    ),
    "total_cholesterol": ClinicalReferenceRange(
        name="Total Cholesterol",
        unit="mg/dL",
        normal_min=125.0,
        normal_max=200.0,
        critical_min=80.0,
        critical_max=450.0,
        clinical_description="Desirable: <200 mg/dL; Borderline high: 200-239; High: >=240"
    ),
    "hdl": ClinicalReferenceRange(
        name="High-Density Lipoprotein (HDL)",
        unit="mg/dL",
        normal_min=40.0,
        normal_max=60.0,
        critical_min=15.0,
        critical_max=120.0,
        clinical_description="Protective: >=60 mg/dL; Major risk factor for heart disease: <40 mg/dL"
    ),
    "ldl": ClinicalReferenceRange(
        name="Low-Density Lipoprotein (LDL)",
        unit="mg/dL",
        normal_min=70.0,
        normal_max=100.0,
        critical_min=30.0,
        critical_max=300.0,
        clinical_description="Optimal: <100 mg/dL; Borderline: 130-159; High: 160-189; Very High: >=190"
    ),
    "triglycerides": ClinicalReferenceRange(
        name="Triglycerides",
        unit="mg/dL",
        normal_min=50.0,
        normal_max=150.0,
        critical_min=30.0,
        critical_max=700.0,
        clinical_description="Normal: <150 mg/dL; Borderline: 150-199; High: 200-499; Very High: >=500"
    ),
    "serum_creatinine": ClinicalReferenceRange(
        name="Serum Creatinine",
        unit="mg/dL",
        normal_min=0.6,
        normal_max=1.2,
        critical_min=0.2,
        critical_max=12.0,
        clinical_description="Normal range: 0.6-1.2 mg/dL (adult males ~0.7-1.3, females ~0.5-1.1)"
    ),
    "egfr": ClinicalReferenceRange(
        name="Estimated GFR",
        unit="mL/min/1.73m²",
        normal_min=90.0,
        normal_max=120.0,
        critical_min=5.0,
        critical_max=140.0,
        clinical_description="Normal: >=90; Mild decrease: 60-89; Moderate: 30-59; Severe: 15-29; Failure: <15"
    ),
}

# Relevant ICD-10 diagnostic codes for chronic disease risk & comorbidity tracking
ICD10_CODE_MAP = {
    "I10": "Essential (primary) hypertension",
    "E78.5": "Hyperlipidemia, unspecified",
    "E78.0": "Pure hypercholesterolemia",
    "E66.01": "Morbid (severe) obesity due to excess calories",
    "E66.9": "Obesity, unspecified",
    "R73.03": "Prediabetes",
    "R73.09": "Other abnormal glucose",
    "N18.2": "Chronic kidney disease, stage 2 (mild)",
    "N18.3": "Chronic kidney disease, stage 3 (moderate)",
    "Z72.0": "Tobacco use",
    "Z83.3": "Family history of diabetes mellitus",
    "I25.10": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
    "K76.0": "Fatty (change of) liver, not elsewhere classified (NAFLD)",
}

# Observation and Prediction temporal windows (in days)
DEFAULT_OBSERVATION_WINDOW_DAYS = 730  # 2 years historical data
DEFAULT_PREDICTION_WINDOW_DAYS = 365   # 1 year forward onset window
DEFAULT_RANDOM_SEED = 42
