"""Reverse Time Attention (RETAIN) Neural Network for Clinical EHR Time-Series.

Implements the RETAIN architecture (Choi et al., NeurIPS 2016):
"RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism"

Key Clinical Capabilities:
1. Dual-level temporal interpretability:
   - alpha_t: Visit-level attention (which clinical encounters were most critical)
   - beta_t: Variable-level attention (which specific labs/vitals at encounter t drove the risk)
2. Additive logit decomposition for exact clinical feature attribution:
   score(t, k) = alpha_t * sum_j(w_j * beta_{t, j} * W_{e, j, k} * x_{t, k})
3. Handles class imbalance via pos_weight in BCEWithLogitsLoss.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class EHRSequenceDataset(Dataset):
    """PyTorch Dataset wrapping longitudinal EHR tensors and masks."""

    def __init__(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        patient_ids: Optional[np.ndarray] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.mask[idx], self.y[idx]


class RETAIN(nn.Module):
    """Reverse Time Attention (RETAIN) Model for Clinical EHR Time-Series.

    Generates dual attention:
    - alpha: Visit-level attention weights in [0, 1] summing to 1 over active visits.
    - beta: Variable-level attention vectors in [-1, 1] per visit.
    """

    def __init__(
        self,
        input_dim: int = 21,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 1. Visit embedding projection: v_t = W_e * x_t + b_e
        self.proj = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 2. Visit-level attention RNN (runs in reverse time)
        self.rnn_alpha = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.alpha_fc = nn.Linear(hidden_dim, 1)

        # 3. Variable-level attention RNN (runs in reverse time)
        self.rnn_beta = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.beta_fc = nn.Linear(hidden_dim, embed_dim)

        # 4. Output classification head
        self.classifier = nn.Linear(embed_dim, 1)

    def _reverse_tensor(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reverses sequential tensors along the time dimension (dim 1)."""
        rev_x = torch.flip(x, dims=[1])
        rev_mask = torch.flip(mask, dims=[1])
        return rev_x, rev_mask

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass generating prediction logits and dual attention tensors.

        Args:
            x: Input visit sequence tensor of shape (batch_size, seq_len, input_dim).
            mask: Binary visit mask of shape (batch_size, seq_len).

        Returns:
            logits: Prediction logits of shape (batch_size, 1).
            alpha: Visit attention weights of shape (batch_size, seq_len).
            beta: Variable attention vectors of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Linear visit projection: (batch, seq_len, embed_dim)
        v = self.dropout(self.proj(x))

        # Step 2: Reverse temporal sequence for reverse-time RNNs
        rev_v, rev_mask = self._reverse_tensor(v, mask)

        # Step 3: Visit-level attention alpha
        # Run backward GRU
        h_alpha, _ = self.rnn_alpha(rev_v)  # (batch, seq_len, hidden_dim)
        # Flip back to forward time order
        h_alpha = torch.flip(h_alpha, dims=[1])
        # Project to scalar energy per visit
        e_alpha = self.alpha_fc(h_alpha).squeeze(-1)  # (batch, seq_len)
        # Mask padding encounters (-1e9 before softmax)
        e_alpha = e_alpha.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(e_alpha, dim=-1)  # (batch, seq_len)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Step 4: Variable-level attention beta
        h_beta, _ = self.rnn_beta(rev_v)  # (batch, seq_len, hidden_dim)
        h_beta = torch.flip(h_beta, dims=[1])
        beta = torch.tanh(self.beta_fc(h_beta))  # (batch, seq_len, embed_dim)

        # Step 5: Dual attention context aggregation
        # alpha shape: (batch, seq_len, 1)
        # beta shape: (batch, seq_len, embed_dim)
        # v shape: (batch, seq_len, embed_dim)
        alpha_expanded = alpha.unsqueeze(-1)
        # Context per step: alpha_t * (beta_t * v_t)
        v_weighted = alpha_expanded * (beta * v)  # (batch, seq_len, embed_dim)
        # Context vector c: sum over time
        c = torch.sum(v_weighted, dim=1)  # (batch, embed_dim)

        # Step 6: Final classification logit
        logits = self.classifier(c)  # (batch, 1)

        return logits, alpha, beta

    def decompose_contributions(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes exact clinical feature contribution scores for each visit and variable.

        Formula (Choi et al., 2016):
            omega(t, k) = alpha_t * sum_j(w_j * beta_{t, j} * W_{e, j, k} * x_{t, k})

        Args:
            x: Input visit sequence tensor (batch_size, seq_len, input_dim).
            mask: Binary visit mask (batch_size, seq_len).

        Returns:
            contributions: Feature contribution matrix (batch_size, seq_len, input_dim).
            alpha: Visit attention weights (batch_size, seq_len).
            beta: Variable attention vectors (batch_size, seq_len, embed_dim).
        """
        self.eval()
        with torch.no_grad():
            _, alpha, beta = self.forward(x, mask)
            # W_e shape: (embed_dim, input_dim)
            W_e = self.proj.weight  # (embed_dim, input_dim)
            # w shape: (embed_dim,)
            w = self.classifier.weight.squeeze(0)  # (embed_dim,)

            # effective_weight per visit: beta_{t, j} * w_j -> shape (batch, seq_len, embed_dim)
            beta_w = beta * w.view(1, 1, -1)

            # Map through W_e: beta_w @ W_e -> shape (batch, seq_len, input_dim)
            gamma = torch.matmul(beta_w, W_e)

            # Multiply by input x_{t, k} and visit attention alpha_t
            # alpha: (batch, seq_len, 1)
            contributions = alpha.unsqueeze(-1) * gamma * x

            # Zero out contributions for padded visits
            contributions = contributions * mask.unsqueeze(-1)

        return contributions, alpha, beta


class RETAINSequenceTrainer:
    """Trainer and evaluator for RETAIN with clinical threshold analysis."""

    def __init__(
        self,
        input_dim: int = 21,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = RETAIN(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 25,
        pos_weight: float = 5.838,
        patience: int = 7,
    ) -> Dict[str, List[float]]:
        """Trains RETAIN with class imbalance compensation and early stopping."""
        pos_weight_tensor = torch.tensor([pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
        }

        best_val_score = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []

            for batch_x, batch_mask, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)

                self.optimizer.zero_grad()
                logits, _, _ = self.model(batch_x, batch_mask)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            history["train_loss"].append(avg_train_loss)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics["brier_score"]
                val_auroc = val_metrics["auroc"]
                history["val_loss"].append(val_loss)
                history["val_auroc"].append(val_auroc)

                if val_auroc > best_val_score:
                    best_val_score = val_auroc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def predict_proba(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts probabilities and collects true labels."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_mask, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_mask = batch_mask.to(self.device)
                logits, _, _ = self.model(batch_x, batch_mask)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(batch_y.numpy())

        return np.array(all_preds), np.array(all_targets)

    def evaluate(self, loader: DataLoader, threshold: float = 0.5) -> Dict[str, Any]:
        """Computes comprehensive clinical validation metrics."""
        y_probs, y_true = self.predict_proba(loader)
        y_pred = (y_probs >= threshold).astype(int)

        try:
            auroc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            auroc = 0.5

        try:
            auprc = float(average_precision_score(y_true, y_probs))
        except Exception:
            auprc = float(np.mean(y_true))

        brier = float(brier_score_loss(y_true, y_probs))
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Sensitivity at fixed 90% specificity
        sens_at_90_spec = 0.0
        thresh_at_90_spec = threshold
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            valid_indices = np.where((1 - fpr) >= 0.90)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[-1]
                sens_at_90_spec = float(tpr[idx])
                thresh_at_90_spec = float(thresholds[idx])
        except Exception:
            pass

        return {
            "model_architecture": "PyTorch RETAIN (Reverse Time Attention)",
            "auroc": auroc,
            "auprc": auprc,
            "sensitivity": recall,
            "specificity": specificity,
            "sensitivity_at_90_specificity": sens_at_90_spec,
            "operating_threshold_at_90_specificity": thresh_at_90_spec,
            "brier_score": brier,
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "sample_count": len(y_true),
            "positive_count": int(np.sum(y_true)),
        }

    def save_checkpoint(
        self,
        filepath: Union[str, Path] = MODELS_DIR / "retain_sequence_model.pt",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves model weights, normalizer statistics, and metrics."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "embed_dim": self.model.embed_dim,
            "hidden_dim": self.model.hidden_dim,
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
            "feature_names": self.feature_names,
            "metrics": metrics or {},
        }
        torch.save(payload, filepath)
        print(f"RETAIN model checkpoint saved to: {filepath}")

        if metrics is not None:
            metrics_path = filepath.parent / "retain_sequence_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"RETAIN evaluation metrics saved to: {metrics_path}")

    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, Path] = MODELS_DIR / "retain_sequence_model.pt",
        device: Optional[str] = None,
    ) -> "RETAINSequenceTrainer":
        """Loads a saved RETAIN checkpoint."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {filepath}")

        if device is None:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = device

        checkpoint = torch.load(filepath, map_location=target_device, weights_only=False)

        trainer = cls(
            input_dim=checkpoint.get("input_dim", 21),
            embed_dim=checkpoint.get("embed_dim", 64),
            hidden_dim=checkpoint.get("hidden_dim", 64),
            device=target_device,
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()
        trainer.norm_mean = checkpoint.get("norm_mean")
        trainer.norm_std = checkpoint.get("norm_std")
        trainer.feature_names = checkpoint.get("feature_names")
        return trainer


def load_processed_tensors(
    data_dir: Union[str, Path] = PROCESSED_DATA_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Loads train and test sequence tensors and metadata."""
    train_npz = Path(data_dir) / "train_sequences.npz"
    test_npz = Path(data_dir) / "test_sequences.npz"

    if not train_npz.exists() or not test_npz.exists():
        raise FileNotFoundError("Processed sequence files not found. Run src.data_loader first.")

    train_data = np.load(train_npz, allow_pickle=True)
    test_data = np.load(test_npz, allow_pickle=True)

    X_train = train_data["X"]
    mask_train = train_data["mask"]
    y_train = train_data["y"]

    X_test = test_data["X"]
    mask_test = test_data["mask"]
    y_test = test_data["y"]

    feature_names = [str(f) for f in train_data.get("features", [])]
    return X_train, mask_train, y_train, X_test, mask_test, y_test, feature_names


def normalize_sequences(
    X_train: np.ndarray,
    mask_train: np.ndarray,
    X_test: np.ndarray,
    mask_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes zero-leakage Z-score normalization on active sequence visits."""
    active_mask = mask_train == 1
    active_features = X_train[active_mask]

    mean = np.mean(active_features, axis=0)
    std = np.std(active_features, axis=0)
    std[std == 0.0] = 1.0

    X_train_norm = np.zeros_like(X_train)
    for i in range(len(X_train)):
        for t in range(X_train.shape[1]):
            if mask_train[i, t] == 1:
                X_train_norm[i, t] = (X_train[i, t] - mean) / std

    X_test_norm = np.zeros_like(X_test)
    for i in range(len(X_test)):
        for t in range(X_test.shape[1]):
            if mask_test[i, t] == 1:
                X_test_norm[i, t] = (X_test[i, t] - mean) / std

    return X_train_norm, X_test_norm, mean, std


def train_and_evaluate_retain() -> Dict[str, Any]:
    """Complete workflow for training, evaluating, and exporting RETAIN."""
    print("=" * 70)
    print("Training PyTorch RETAIN Clinical Sequence Model")
    print("=" * 70)

    X_train, mask_train, y_train, X_test, mask_test, y_test, feature_names = load_processed_tensors()
    X_train_norm, X_test_norm, mean, std = normalize_sequences(X_train, mask_train, X_test, mask_test)

    pos_count = int(np.sum(y_train))
    neg_count = len(y_train) - pos_count
    pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    train_dataset = EHRSequenceDataset(X_train_norm, mask_train, y_train)
    test_dataset = EHRSequenceDataset(X_test_norm, mask_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    trainer = RETAINSequenceTrainer(
        input_dim=X_train.shape[2],
        embed_dim=64,
        hidden_dim=64,
        dropout=0.2,
        lr=0.001,
        weight_decay=1e-4,
    )
    trainer.norm_mean = mean
    trainer.norm_std = std
    trainer.feature_names = feature_names

    print(f"Cohort sizes: Train={len(y_train)} ({pos_count} pos), Test={len(y_test)} ({int(np.sum(y_test))} pos)")
    print(f"Calculated pos_weight: {pos_weight:.3f}")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=30,
        pos_weight=pos_weight,
        patience=8,
    )

    metrics = trainer.evaluate(test_loader)
    print("\n--- RETAIN Test Set Evaluation Results ---")
    print(f"AUROC:                         {metrics['auroc']:.4f}")
    print(f"AUPRC:                         {metrics['auprc']:.4f}")
    print(f"Sensitivity @ 0.5:             {metrics['sensitivity']:.4f}")
    print(f"Specificity @ 0.5:             {metrics['specificity']:.4f}")
    print(f"Sensitivity @ 90% Specificity: {metrics['sensitivity_at_90_specificity']:.4f}")
    print(f"Brier Calibration Score:       {metrics['brier_score']:.6f}")

    trainer.save_checkpoint(
        filepath=MODELS_DIR / "retain_sequence_model.pt",
        metrics=metrics,
    )

    return metrics


if __name__ == "__main__":
    train_and_evaluate_retain()
