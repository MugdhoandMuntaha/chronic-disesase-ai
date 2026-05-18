"""Unit tests for Step 4: Sequence-Aware Deep Model (src/models/sequence_model.py)."""

import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.sequence_model import (
    EHRSequenceDataset,
    EHRSequenceGRU,
    TemporalAttention,
    SequenceModelTrainer,
)


def test_sequence_dataset() -> None:
    """Verifies that EHRSequenceDataset tensors and indexing behave correctly."""
    X = np.random.randn(20, 15, 21).astype(np.float32)
    mask = np.ones((20, 15), dtype=np.float32)
    mask[:, 10:] = 0.0  # Last 5 visits are padded
    y = np.random.choice([0.0, 1.0], size=20).astype(np.float32)

    dataset = EHRSequenceDataset(X, mask, y)
    assert len(dataset) == 20

    x_sample, mask_sample, y_sample = dataset[0]
    assert x_sample.shape == (15, 21)
    assert mask_sample.shape == (15,)
    assert y_sample.shape == ()


def test_temporal_attention_masking() -> None:
    """Verifies that temporal attention completely zeroes out attention on padded steps."""
    batch_size = 4
    seq_len = 10
    hidden_dim = 32

    gru_outputs = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.ones(batch_size, seq_len)
    mask[:, 6:] = 0.0  # Steps 6 through 9 are padding

    attention_layer = TemporalAttention(hidden_dim=hidden_dim)
    context, attn_weights = attention_layer(gru_outputs, mask)

    assert context.shape == (batch_size, hidden_dim)
    assert attn_weights.shape == (batch_size, seq_len)

    # Weights for padded visits must be exactly 0.0
    padded_weights = attn_weights[:, 6:].detach().numpy()
    assert np.allclose(padded_weights, 0.0, atol=1e-6), "Padded steps must receive 0 attention"

    # Weights for valid visits must sum to 1.0 per patient
    weight_sums = attn_weights.sum(dim=-1).detach().numpy()
    assert np.allclose(weight_sums, 1.0, atol=1e-5), "Attention distribution must sum to 1.0"


def test_sequence_gru_forward_and_predict() -> None:
    """Verifies forward pass and probability output shapes for EHRSequenceGRU."""
    model = EHRSequenceGRU(input_dim=21, hidden_dim=32, num_layers=2, dropout=0.1, bidirectional=True)

    x = torch.randn(8, 15, 21)
    mask = torch.ones(8, 15)
    mask[:, 12:] = 0.0

    logits, attn = model(x, mask)
    assert logits.shape == (8, 1)
    assert attn.shape == (8, 15)

    probs, _ = model.predict_proba(x, mask)
    assert probs.shape == (8, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_sequence_trainer_workflow() -> None:
    """Verifies that trainer runs for small epochs and saves valid checkpoints."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = SequenceModelTrainer(
            models_dir=tmp_dir,
            epochs=2,
            batch_size=32,
            hidden_dim=32,
            num_layers=1,
            lr=1e-3,
        )
        metrics = trainer.fit()

        assert "auroc" in metrics
        assert "auprc" in metrics
        assert "val_loss" in metrics

        # Verify checkpoint file was created
        saved_file = Path(tmp_dir) / "gru_sequence_model.pt"
        assert saved_file.exists()
        assert saved_file.stat().st_size > 0

        # Verify checkpoint can be reloaded
        reloaded = trainer.load_model("gru_sequence_model.pt")
        assert isinstance(reloaded, EHRSequenceGRU)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
