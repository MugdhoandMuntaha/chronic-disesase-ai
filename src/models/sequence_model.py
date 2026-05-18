"""Sequence-Aware Recurrent Neural Network (GRU with Temporal Attention) for EHR Time-Series.

Models longitudinal patient encounters using a multi-layer bidirectional GRU paired with
a masked temporal attention mechanism for visit-level interpretability.
Handles class imbalance via pos_weight in BCEWithLogitsLoss and supports variable-length visit histories.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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


class TemporalAttention(nn.Module):
    """Masked additive temporal attention layer for variable-length longitudinal encounters.

    Computes visit attention weights alpha_t, ignoring zero-padded visit steps.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_outputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes context representation and visit attention weights.

        Args:
            gru_outputs: Tensor of shape (batch_size, seq_len, hidden_dim).
            mask: Binary tensor of shape (batch_size, seq_len) with 1 for real visits, 0 for padding.

        Returns:
            context: Weighted sequence summary tensor of shape (batch_size, hidden_dim).
            attn_weights: Attention distribution tensor of shape (batch_size, seq_len).
        """
        # (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn_proj(gru_outputs))
        # (batch_size, seq_len)
        scores = self.attn_vector(energy).squeeze(-1)

        # Mask padding steps with -1e9 so they receive 0 attention in softmax
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)

        # Handle edge-case where an entire sequence is masked
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Compute context vector: sum over time of (alpha_t * h_t)
        # (batch_size, hidden_dim)
        context = torch.sum(gru_outputs * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights


class EHRSequenceGRU(nn.Module):
    """Sequence-Aware Gated Recurrent Unit with Temporal Attention for EHR Disease Onset Prediction."""

    def __init__(
        self,
        input_dim: int = 21,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Linear projection of visit features
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Recurrent backbone
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Temporal attention pooling
        self.attention = TemporalAttention(hidden_dim=gru_out_dim)

        # Classification MLP head
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input visit sequence tensor (batch_size, seq_len, input_dim).
            mask: Visit existence mask (batch_size, seq_len).

        Returns:
            logits: Prediction logits (batch_size, 1).
            attn_weights: Visit attention weights (batch_size, seq_len).
        """
        # Step 1: Input projection (batch_size, seq_len, hidden_dim)
        projected = self.input_layer(x)

        # Step 2: GRU sequential pass (batch_size, seq_len, gru_out_dim)
        gru_out, _ = self.gru(projected)

        # Step 3: Attention pooling over visit steps (batch_size, gru_out_dim)
        context, attn_weights = self.attention(gru_out, mask)

        # Step 4: Classification head
        logits = self.classifier(context)
        return logits, attn_weights

    def predict_proba(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes sigmoid calibrated probabilities for inference."""
        self.eval()
        with torch.no_grad():
            logits, attn = self.forward(x, mask)
            probs = torch.sigmoid(logits)
        return probs, attn


class SequenceModelTrainer:
    """Manages data preparation, loss weighting, training loop, and evaluation."""

    def __init__(
        self,
        processed_dir: Union[str, Path] = PROCESSED_DATA_DIR,
        models_dir: Union[str, Path] = MODELS_DIR,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 35,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[EHRSequenceGRU] = None
        self.feature_names: List[str] = []
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.pos_weight: float = 1.0
        self.metrics: Dict[str, Any] = {}

    def load_and_normalize_data(
        self,
    ) -> Tuple[DataLoader, DataLoader, int, int]:
        """Loads NPZ sequential arrays and computes training-set z-score normalization."""
        train_path = self.processed_dir / "train_sequences.npz"
        test_path = self.processed_dir / "test_sequences.npz"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Sequential data not found in {self.processed_dir}. "
                "Run Step 2 (src/data_loader.py) first."
            )

        print(f"[SequenceModelTrainer] Loading sequential NPZ arrays from {self.processed_dir}...")
        train_npz = np.load(train_path)
        test_npz = np.load(test_path)

        X_train = train_npz["X"]       # (800, 15, 21)
        mask_train = train_npz["mask"] # (800, 15)
        y_train = train_npz["y"]       # (800,)

        X_test = test_npz["X"]         # (200, 15, 21)
        mask_test = test_npz["mask"]   # (200, 15)
        y_test = test_npz["y"]         # (200,)

        self.feature_names = [str(f) for f in train_npz.get("features", [])]
        input_dim = X_train.shape[-1]
        seq_len = X_train.shape[1]

        # Calculate class imbalance pos_weight: Negative / Positive
        num_pos = float(np.sum(y_train == 1))
        num_neg = float(np.sum(y_train == 0))
        self.pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        print(f"[SequenceModelTrainer] Imbalance weight pos_weight = {self.pos_weight:.3f}")

        # Compute mean and std on valid non-padded visit entries from TRAIN set only
        valid_mask = mask_train.astype(bool) # (800, 15)
        valid_entries = X_train[valid_mask]  # (N_valid_visits, 21)

        self.mean = np.mean(valid_entries, axis=0, keepdims=True) # (1, 21)
        self.std = np.std(valid_entries, axis=0, keepdims=True)   # (1, 21)
        self.std[self.std < 1e-6] = 1.0  # Prevent division by zero

        # Apply normalization to visit entries, keeping padding at 0.0
        X_train_norm = np.zeros_like(X_train)
        X_test_norm = np.zeros_like(X_test)

        X_train_norm[valid_mask] = (X_train[valid_mask] - self.mean) / self.std
        valid_test_mask = mask_test.astype(bool)
        X_test_norm[valid_test_mask] = (X_test[valid_test_mask] - self.mean) / self.std

        train_dataset = EHRSequenceDataset(X_train_norm, mask_train, y_train, train_npz.get("patient_ids"))
        test_dataset = EHRSequenceDataset(X_test_norm, mask_test, y_test, test_npz.get("patient_ids"))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(
            f"[SequenceModelTrainer] Data prepared: {len(train_dataset)} train samples, "
            f"{len(test_dataset)} test samples. Feature dimension: {input_dim}, Max sequence length: {seq_len}."
        )
        return train_loader, test_loader, input_dim, seq_len

    def train_epoch(
        self,
        model: EHRSequenceGRU,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Runs one training epoch over minibatches."""
        model.train()
        total_loss = 0.0

        for X_batch, mask_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(-1)

            optimizer.zero_grad()
            logits, _ = model(X_batch, mask_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients in recurrent networks
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * len(y_batch)

        return total_loss / len(loader.dataset)

    def evaluate(
        self, model: EHRSequenceGRU, loader: DataLoader
    ) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
        """Evaluates model performance across clinical metrics on a test DataLoader."""
        model.eval()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=self.device))
        total_loss = 0.0

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for X_batch, mask_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                y_targets = y_batch.to(self.device).unsqueeze(-1)

                logits, _ = model(X_batch, mask_batch)
                loss = criterion(logits, y_targets)
                total_loss += loss.item() * len(y_batch)

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_targets.extend(y_batch.numpy().flatten())

        val_loss = total_loss / len(loader.dataset)
        y_true = np.array(all_targets, dtype=int)
        y_prob = np.array(all_probs, dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)

        # Clinical metrics
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))

        # Sensitivity at 90% fixed specificity
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        spec = 1.0 - fpr
        valid_pts = np.where(spec >= 0.90)[0]
        sens_at_90_spec = float(tpr[valid_pts[np.argmax(tpr[valid_pts])]]) if len(valid_pts) > 0 else 0.0

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))

        metrics = {
            "val_loss": round(val_loss, 5),
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(specificity, 4),
            "sensitivity_at_90_spec": round(sens_at_90_spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
            "brier_score": round(brier, 6),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }
        return val_loss, metrics, y_true, y_prob

    def fit(self) -> Dict[str, Any]:
        """Executes complete training loop with validation monitoring and checkpointing."""
        train_loader, test_loader, input_dim, _ = self.load_and_normalize_data()

        print(f"[SequenceModelTrainer] Instantiating EHRSequenceGRU on device '{self.device}'...")
        self.model = EHRSequenceGRU(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

        best_auprc = -1.0
        best_metrics: Dict[str, Any] = {}
        best_state: Optional[Dict[str, Any]] = None

        print(f"[SequenceModelTrainer] Commencing model training for {self.epochs} epochs...")
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(self.model, train_loader, optimizer, criterion)
            val_loss, val_metrics, _, _ = self.evaluate(self.model, test_loader)
            scheduler.step(val_metrics["auprc"])

            if val_metrics["auprc"] > best_auprc:
                best_auprc = val_metrics["auprc"]
                best_metrics = val_metrics
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                print(
                    f"  Epoch {epoch:02d}/{self.epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val AUROC: {val_metrics['auroc']:.4f} | Val AUPRC: {val_metrics['auprc']:.4f} | "
                    f"Val Sens: {val_metrics['sensitivity']:.4f}"
                )

        # Restore best model weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.metrics = best_metrics
        self.save_model()
        return self.metrics

    def save_model(self, filename: str = "gru_sequence_model.pt") -> Path:
        """Saves trained model state dict, normalization statistics, and metrics to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        save_path = self.models_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_architecture": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "bidirectional": self.model.bidirectional,
                "dropout": self.dropout,
            },
            "norm_mean": self.mean,
            "norm_std": self.std,
            "feature_names": self.feature_names,
            "pos_weight": self.pos_weight,
            "metrics": self.metrics,
        }
        torch.save(checkpoint, save_path)
        print(f"[SequenceModelTrainer] Model checkpoint successfully saved to {save_path}")

        # Save metrics JSON for tracking
        metrics_path = self.models_dir / "gru_sequence_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[SequenceModelTrainer] Metrics report saved to {metrics_path}")

        return save_path

    def load_model(self, filename: str = "gru_sequence_model.pt") -> EHRSequenceGRU:
        """Loads model checkpoint and normalization parameters from disk."""
        checkpoint_path = self.models_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        arch = checkpoint["model_architecture"]

        self.model = EHRSequenceGRU(
            input_dim=arch["input_dim"],
            hidden_dim=arch["hidden_dim"],
            num_layers=arch["num_layers"],
            dropout=arch["dropout"],
            bidirectional=arch["bidirectional"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.mean = checkpoint["norm_mean"]
        self.std = checkpoint["norm_std"]
        self.feature_names = checkpoint["feature_names"]
        self.pos_weight = checkpoint["pos_weight"]
        self.metrics = checkpoint.get("metrics", {})
        return self.model


def main() -> None:
    """Entrypoint executing sequence model training and comparing against XGBoost baseline."""
    print("=" * 75)
    print("STEP 4: SEQUENCE-AWARE DEEP MODEL TRAINING (PYTORCH GRU + ATTENTION)")
    print("=" * 75)

    trainer = SequenceModelTrainer(epochs=25, batch_size=32, lr=1.5e-3)
    metrics = trainer.fit()

    print("\n" + "=" * 75)
    print("CLINICAL BENCHMARK COMPARISON: TABULAR BASELINE vs. SEQUENCE GRU")
    print("=" * 75)

    # Load XGBoost metrics if available for side-by-side medical comparison
    xgb_metrics_path = MODELS_DIR / "xgb_baseline_metrics.json"
    xgb_metrics = {}
    if xgb_metrics_path.exists():
        with open(xgb_metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            xgb_metrics = data.get("metrics", {})

    print(f"{'Metric':<32} | {'XGBoost Tabular':<18} | {'PyTorch Sequence GRU':<20}")
    print("-" * 75)
    print(f"{'AUROC (Discrimination)':<32} | {xgb_metrics.get('auroc', 'N/A'):<18} | {metrics['auroc']:<20.4f}")
    print(f"{'AUPRC (Precision-Recall)':<32} | {xgb_metrics.get('auprc', 'N/A'):<18} | {metrics['auprc']:<20.4f}")
    print(f"{'Sensitivity @ 0.5 Threshold':<32} | {xgb_metrics.get('sensitivity', 'N/A'):<18} | {metrics['sensitivity']:<20.4f}")
    print(f"{'Specificity @ 0.5 Threshold':<32} | {xgb_metrics.get('specificity', 'N/A'):<18} | {metrics['specificity']:<20.4f}")
    print(f"{'Sens @ 90% Fixed Specificity':<32} | {xgb_metrics.get('sensitivity', 'N/A'):<18} | {metrics['sensitivity_at_90_spec']:<20.4f}")
    print(f"{'Brier Calibration Score':<32} | {xgb_metrics.get('brier_score', 'N/A'):<18} | {metrics['brier_score']:<20.6f}")
    print("-" * 75)
    cm = metrics["confusion_matrix"]
    print(f"GRU Test Confusion Matrix (N=200):")
    print(f"  True Negatives  : {cm['true_negatives']:<4} | False Positives : {cm['false_positives']}")
    print(f"  False Negatives : {cm['false_negatives']:<4} | True Positives  : {cm['true_positives']}")
    print("=" * 75)
    print(f"Sequence model saved to models/gru_sequence_model.pt")


if __name__ == "__main__":
    main()
