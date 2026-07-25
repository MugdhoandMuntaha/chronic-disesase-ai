"""Clinical Decision Curve Analysis (DCA) and Clinical Net Benefit Evaluation.

Implements Decision Curve Analysis (Vickers & Elkin, BMJ / Annals of Internal Medicine):
Quantifies the clinical utility of predictive models across threshold probabilities
compared to universal intervention ("Treat All") and no intervention ("Treat None").

Formulations:
1. Clinical Net Benefit:
   Net Benefit(p_t) = (TP / N) - (FP / N) * (p_t / (1 - p_t))
2. Treat All Net Benefit:
   Net Benefit_all(p_t) = Prevalence - (1 - Prevalence) * (p_t / (1 - p_t))
3. Interventions Avoided per 100 Patients:
   Avoided(p_t) = ((Net Benefit_model - Net Benefit_all) / (p_t / (1 - p_t))) * 100
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def compute_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Computes clinical Net Benefit across threshold probabilities.

    Args:
        y_true: Binary ground truth array of shape (N,).
        y_prob: Predicted probability array of shape (N,).
        thresholds: Array of decision threshold probabilities p_t in (0, 1).

    Returns:
        net_benefits: Array of Net Benefit values for each threshold.
    """
    n = len(y_true)
    net_benefits = np.zeros(len(thresholds))

    for i, p_t in enumerate(thresholds):
        if p_t >= 1.0 or p_t <= 0.0:
            continue

        # Classify patients at operating threshold p_t
        y_pred = (y_prob >= p_t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # Weighting factor for false positives: p_t / (1 - p_t)
        weight = p_t / (1.0 - p_t)
        nb = (tp / n) - (fp / n) * weight
        net_benefits[i] = nb

    return net_benefits


def compute_treat_all_net_benefit(
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Computes Net Benefit of the default 'Treat All' clinical strategy."""
    n = len(y_true)
    prevalence = np.sum(y_true) / n
    net_benefits = np.zeros(len(thresholds))

    for i, p_t in enumerate(thresholds):
        if p_t >= 1.0 or p_t <= 0.0:
            continue
        weight = p_t / (1.0 - p_t)
        nb = prevalence - (1.0 - prevalence) * weight
        net_benefits[i] = nb

    return net_benefits


def compute_interventions_avoided(
    net_benefit_model: np.ndarray,
    net_benefit_treat_all: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Computes net reduction in unnecessary clinical interventions per 100 patients."""
    avoided = np.zeros(len(thresholds))
    for i, p_t in enumerate(thresholds):
        if p_t >= 1.0 or p_t <= 0.0:
            continue
        weight = p_t / (1.0 - p_t)
        if weight > 0:
            val = ((net_benefit_model[i] - net_benefit_treat_all[i]) / weight) * 100.0
            avoided[i] = max(0.0, val)
    return avoided


class ClinicalDecisionCurveAnalyzer:
    """Orchestrates Decision Curve Analysis for Tabular, GRU, and RETAIN models."""

    def __init__(
        self,
        threshold_min: float = 0.02,
        threshold_max: float = 0.60,
        num_thresholds: int = 100,
    ):
        self.thresholds = np.linspace(threshold_min, threshold_max, num_thresholds)
        self.results: Dict[str, Any] = {}

    def run_analysis(self) -> Dict[str, Any]:
        """Runs DCA across all available test predictions and exports publication figures."""
        # 1. Load test ground truth labels
        test_tab_path = PROCESSED_DATA_DIR / "test_tabular.csv"
        if not test_tab_path.exists():
            raise FileNotFoundError("test_tabular.csv not found. Run preprocessing first.")

        df_test = pd.read_csv(test_tab_path)
        target_col = "target_label" if "target_label" in df_test.columns else "target_disease"
        y_test = df_test[target_col].values.astype(int)
        n_samples = len(y_test)
        prevalence = float(np.mean(y_test))

        print(f"[DCA] Evaluating cohort of {n_samples} patients (Prevalence: {prevalence:.1%})")

        # Baseline strategies: Treat All & Treat None
        nb_treat_all = compute_treat_all_net_benefit(y_test, self.thresholds)
        nb_treat_none = np.zeros(len(self.thresholds))

        curves: Dict[str, Dict[str, Any]] = {
            "Treat All (Universal Intervention)": {
                "net_benefit": nb_treat_all.tolist(),
                "color": "#94a3b8",
                "linestyle": ":",
            },
            "Treat None (No Intervention)": {
                "net_benefit": nb_treat_none.tolist(),
                "color": "#475569",
                "linestyle": "--",
            },
        }

        # 2. XGBoost Baseline
        xgb_path = MODELS_DIR / "xgb_baseline.json"
        if xgb_path.exists():
            from xgboost import XGBClassifier
            from src.models.baseline_xgb import XGBoostBaselineTrainer

            trainer = XGBoostBaselineTrainer()
            X_train, y_train, X_test, _ = trainer.load_data()
            model = XGBClassifier()
            model.load_model(str(xgb_path))
            y_prob_xgb = model.predict_proba(X_test)[:, 1]

            nb_xgb = compute_net_benefit(y_test, y_prob_xgb, self.thresholds)
            avoided_xgb = compute_interventions_avoided(nb_xgb, nb_treat_all, self.thresholds)
            curves["XGBoost Tabular Baseline"] = {
                "net_benefit": nb_xgb.tolist(),
                "interventions_avoided_per_100": avoided_xgb.tolist(),
                "color": "#0284c7",
                "linestyle": "-",
            }
            print(f"[DCA] XGBoost Net Benefit computed. Mean NB: {np.mean(nb_xgb):.4f}")

        # 3. PyTorch Sequence GRU
        gru_path = MODELS_DIR / "gru_sequence_model.pt"
        if gru_path.exists():
            from src.models.sequence_model import SequenceModelTrainer
            import torch

            trainer_gru = SequenceModelTrainer()
            trainer_gru.load_model("gru_sequence_model.pt")
            test_npz = np.load(PROCESSED_DATA_DIR / "test_sequences.npz")
            X_test_seq = test_npz["X"]
            mask_test_seq = test_npz["mask"]

            valid_mask = mask_test_seq.astype(bool)
            X_test_norm = np.zeros_like(X_test_seq)
            X_test_norm[valid_mask] = (X_test_seq[valid_mask] - trainer_gru.mean) / trainer_gru.std

            t_X = torch.tensor(X_test_norm, dtype=torch.float32).to(trainer_gru.device)
            t_mask = torch.tensor(mask_test_seq, dtype=torch.float32).to(trainer_gru.device)

            with torch.no_grad():
                probs_gru, _ = trainer_gru.model.predict_proba(t_X, t_mask)
                y_prob_gru = probs_gru.cpu().numpy().flatten()

            nb_gru = compute_net_benefit(y_test, y_prob_gru, self.thresholds)
            avoided_gru = compute_interventions_avoided(nb_gru, nb_treat_all, self.thresholds)
            curves["PyTorch Sequence GRU + Attention"] = {
                "net_benefit": nb_gru.tolist(),
                "interventions_avoided_per_100": avoided_gru.tolist(),
                "color": "#8b5cf6",
                "linestyle": "-",
            }
            print(f"[DCA] PyTorch GRU Net Benefit computed. Mean NB: {np.mean(nb_gru):.4f}")

        # 4. PyTorch RETAIN
        retain_path = MODELS_DIR / "retain_sequence_model.pt"
        if retain_path.exists():
            from src.models.retain_model import RETAINSequenceTrainer
            import torch

            trainer_retain = RETAINSequenceTrainer.load_checkpoint(retain_path)
            test_npz = np.load(PROCESSED_DATA_DIR / "test_sequences.npz")
            X_test_seq = test_npz["X"]
            mask_test_seq = test_npz["mask"]

            valid_mask = mask_test_seq.astype(bool)
            X_test_norm = np.zeros_like(X_test_seq)
            if trainer_retain.norm_mean is not None and trainer_retain.norm_std is not None:
                X_test_norm[valid_mask] = (X_test_seq[valid_mask] - trainer_retain.norm_mean) / trainer_retain.norm_std

            t_X = torch.tensor(X_test_norm, dtype=torch.float32).to(trainer_retain.device)
            t_mask = torch.tensor(mask_test_seq, dtype=torch.float32).to(trainer_retain.device)

            with torch.no_grad():
                logits, _, _ = trainer_retain.model(t_X, t_mask)
                y_prob_retain = torch.sigmoid(logits).cpu().numpy().flatten()

            nb_retain = compute_net_benefit(y_test, y_prob_retain, self.thresholds)
            avoided_retain = compute_interventions_avoided(nb_retain, nb_treat_all, self.thresholds)
            curves["PyTorch RETAIN (Reverse-Time Attention)"] = {
                "net_benefit": nb_retain.tolist(),
                "interventions_avoided_per_100": avoided_retain.tolist(),
                "color": "#10b981",
                "linestyle": "-",
            }
            print(f"[DCA] PyTorch RETAIN Net Benefit computed. Mean NB: {np.mean(nb_retain):.4f}")

        self.results = {
            "thresholds": self.thresholds.tolist(),
            "prevalence": prevalence,
            "sample_count": n_samples,
            "curves": curves,
        }

        self.save_artifacts()
        return self.results

    def save_artifacts(self) -> None:
        """Exports publication-grade DCA dual plot and summary JSON."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Save JSON metrics
        json_path = REPORTS_DIR / "decision_curve_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"[DCA] Saved DCA metrics to: {json_path}")

        # 2. Plotting Dual Decision Curves
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: Net Benefit vs. Threshold Probability
        thresh_pct = np.array(self.results["thresholds"]) * 100.0

        for name, data in self.results["curves"].items():
            nb = np.array(data["net_benefit"])
            ax1.plot(
                thresh_pct,
                nb,
                label=name,
                color=data["color"],
                linestyle=data["linestyle"],
                linewidth=2.5 if "-" in data["linestyle"] else 1.8,
            )

        ax1.set_title("Clinical Decision Curve Analysis (DCA)", fontsize=14, fontweight="bold", pad=12)
        ax1.set_xlabel("Clinical Threshold Probability $p_t$ (%)", fontsize=12)
        ax1.set_ylabel("Clinical Net Benefit", fontsize=12)
        ax1.set_ylim(-0.02, max(0.20, self.results["prevalence"] * 1.2))
        ax1.set_xlim(thresh_pct.min(), thresh_pct.max())
        ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax1.legend(loc="upper right", frameon=True, fontsize=10)

        # Subplot 2: Interventions Avoided per 100 Patients
        for name, data in self.results["curves"].items():
            if "interventions_avoided_per_100" in data:
                avoided = np.array(data["interventions_avoided_per_100"])
                ax2.plot(
                    thresh_pct,
                    avoided,
                    label=f"{name} vs. Treat All",
                    color=data["color"],
                    linewidth=2.2,
                )

        ax2.set_title("Net Reduction in Unnecessary Interventions", fontsize=14, fontweight="bold", pad=12)
        ax2.set_xlabel("Clinical Threshold Probability $p_t$ (%)", fontsize=12)
        ax2.set_ylabel("Avoided Interventions / 100 Patients", fontsize=12)
        ax2.set_xlim(thresh_pct.min(), thresh_pct.max())
        ax2.legend(loc="upper right", frameon=True, fontsize=10)

        plt.tight_layout()
        plot_path = REPORTS_DIR / "decision_curve_analysis.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[DCA] Publication-grade DCA figure saved to: {plot_path}")


def run_decision_curve_pipeline() -> Dict[str, Any]:
    """CLI runner executing Decision Curve Analysis and exporting artifacts."""
    print("=" * 75)
    print("EXECUTING CLINICAL DECISION CURVE ANALYSIS (DCA)")
    print("=" * 75)

    analyzer = ClinicalDecisionCurveAnalyzer()
    results = analyzer.run_analysis()

    print("\n--- Clinical Decision Utility Highlights ---")
    thresholds = np.array(results["thresholds"])
    # Find evaluation point at 15% threshold (close to cohort prevalence)
    idx_15 = np.argmin(np.abs(thresholds - 0.15))
    t_val = thresholds[idx_15]

    for model_name, data in results["curves"].items():
        if "interventions_avoided_per_100" in data:
            nb = data["net_benefit"][idx_15]
            avoided = data["interventions_avoided_per_100"][idx_15]
            print(f"{model_name:<40} @ {t_val:.1%} threshold | Net Benefit: {nb:.4f} | Interventions Avoided: {avoided:.1f}/100 pts")

    return results


if __name__ == "__main__":
    run_decision_curve_pipeline()
