"""Unit tests for Algorithmic Fairness and Subgroup Equity Audit."""

import numpy as np
import pytest

from src.evaluation.fairness import ClinicalFairnessAuditor, compute_group_metrics


def test_compute_group_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.2, 0.8, 0.7, 0.9])  # 1 FP, 0 FN at 0.50 threshold
    metrics = compute_group_metrics(y_true, y_prob, threshold=0.50)

    assert metrics["sample_size"] == 6
    assert metrics["disease_cases"] == 2
    assert metrics["control_cases"] == 4
    assert metrics["auroc"] is not None
    assert metrics["false_positive_rate_fpr"] == 0.25  # 1 / 4
    assert metrics["sensitivity_tpr"] == 1.0  # 2 / 2


def test_clinical_fairness_auditor():
    auditor = ClinicalFairnessAuditor()
    report = auditor.run_full_audit()

    assert "biological_sex_audit" in report
    assert "age_bracket_audit" in report
    assert "racial_ethnic_audit" in report
    assert "regulatory_summary" in report

    # Check sex audit
    sex_subgroups = report["biological_sex_audit"]["subgroups"]
    assert "Male" in sex_subgroups
    assert "Female" in sex_subgroups
    assert sex_subgroups["Male"]["sample_size"] > 0
    assert sex_subgroups["Female"]["sample_size"] > 0

    # Check age brackets
    age_subgroups = report["age_bracket_audit"]["subgroups"]
    assert len(age_subgroups) >= 3

    # Check compliance summary
    reg = report["regulatory_summary"]
    assert "audit_status" in reg
