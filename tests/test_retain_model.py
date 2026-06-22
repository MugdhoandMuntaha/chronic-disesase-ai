"""Automated Unit and Integration Tests for RETAIN Clinical Sequence Model."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.retain_model import (
    EHRSequenceDataset,
    RETAIN,
    RETAINSequenceTrainer,
    load_processed_tensors,
    normalize_sequences,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@pytest.fixture
def dummy_data():
    """Generates synthetic sequence tensors for fast unit testing."""
    batch_size = 16
    seq_len = 8
    input_dim = 21

    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    mask = np.ones((batch_size, seq_len), dtype=np.float32)
    # Mask out some trailing visits to simulate variable history
    for i in range(batch_size):
        pad_count = i % 4
        if pad_count > 0:
            mask[i, -pad_count:] = 0.0

    y = (np.random.rand(batch_size) > 0.7).astype(np.float32)
    return X, mask, y


def test_retain_architecture_forward(dummy_data):
    """Verifies RETAIN model forward pass outputs and attention dimensions."""
    X, mask, _ = dummy_data
    t_X = torch.tensor(X)
    t_mask = torch.tensor(mask)

    model = RETAIN(input_dim=21, embed_dim=32, hidden_dim=32, dropout=0.1)
    model.eval()

    logits, alpha, beta = model(t_X, t_mask)

    assert logits.shape == (16, 1)
    assert alpha.shape == (16, 8)
    assert beta.shape == (16, 8, 32)

    # Check that padded encounters receive zero attention weight
    for i in range(16):
        for t in range(8):
            if mask[i, t] == 0.0:
                assert alpha[i, t].item() == pytest.approx(0.0, abs=1e-5)


def test_retain_contribution_decomposition(dummy_data):
    """Verifies that RETAIN additive contribution decomposition produces matching shapes."""
    X, mask, _ = dummy_data
    t_X = torch.tensor(X)
    t_mask = torch.tensor(mask)

    model = RETAIN(input_dim=21, embed_dim=32, hidden_dim=32)
    contributions, alpha, beta = model.decompose_contributions(t_X, t_mask)

    assert contributions.shape == (16, 8, 21)
    assert alpha.shape == (16, 8)
    assert beta.shape == (16, 8, 32)

    # Verify that padded encounters contribute 0
    for i in range(16):
        for t in range(8):
            if mask[i, t] == 0.0:
                assert torch.all(contributions[i, t] == 0.0)


def test_retain_checkpoint_save_and_reload(tmp_path, dummy_data):
    """Verifies model serialization and weight restoration."""
    X, mask, y = dummy_data
    trainer = RETAINSequenceTrainer(input_dim=21, embed_dim=32, hidden_dim=32)
    trainer.norm_mean = np.zeros((21,))
    trainer.norm_std = np.ones((21,))
    trainer.feature_names = [f"feat_{i}" for i in range(21)]

    save_file = tmp_path / "test_retain.pt"
    dummy_metrics = {"auroc": 0.985, "auprc": 0.942}
    trainer.save_checkpoint(save_file, metrics=dummy_metrics)

    assert save_file.exists()

    reloaded = RETAINSequenceTrainer.load_checkpoint(save_file)
    assert reloaded.model.input_dim == 21
    assert reloaded.model.embed_dim == 32
    assert reloaded.feature_names == trainer.feature_names


def test_retain_saved_model_artifact():
    """Verifies that the production RETAIN artifact in models/ is valid and runnable."""
    model_path = MODELS_DIR / "retain_sequence_model.pt"
    assert model_path.exists(), "models/retain_sequence_model.pt must exist"

    trainer = RETAINSequenceTrainer.load_checkpoint(model_path)
    assert trainer.model is not None
    assert trainer.norm_mean is not None
    assert trainer.norm_std is not None
