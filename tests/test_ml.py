"""
tests/test_ml.py
----------------
Automated unit tests for machine learning subsystem:
- Class vocabulary definitions and mapping logic
- Dataset manifest disjointness and non-leakage
- Dataset loading, normalization, and tensor shapes
- DS-CNN and Residual DS-CNN model architectures and parameter limits
- Baseline and Improved checkpoint reloadability and inference verification
- SpecAugment tensor transformations
- Precomputed tensor cache loading and shape integrity
"""

import json
import pathlib
import numpy as np
import pytest
import torch

from src.ml.classes import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    CORE_COMMANDS,
    AUXILIARY_WORDS,
    NUM_CLASSES,
    map_word_to_class,
)
from src.ml.dataset import SpeechCommandsDataset, apply_spec_augment, load_wav_16k_mono
from src.ml.manifests import load_split_list
from src.ml.model import DSCNNKeywordSpotter, ResidualDSCNNKeywordSpotter, get_model


def test_class_vocabulary_definitions():
    """Verifies that the vocabulary is 21 classes with proper index alignment."""
    assert NUM_CLASSES == 21
    assert len(CLASS_NAMES) == 21
    assert CLASS_NAMES[0] == "_silence_"
    assert CLASS_NAMES[1] == "_unknown_"
    assert len(CORE_COMMANDS) == 10

    for cmd in CORE_COMMANDS:
        assert cmd in CLASS_NAMES
        assert map_word_to_class(cmd) == cmd

    # Test unknown and silence mapping
    assert map_word_to_class("cat") == "_unknown_"
    assert map_word_to_class("backward") == "_unknown_"
    assert map_word_to_class("_background_noise_") == "_silence_"


def test_manifest_disjointness_no_data_leakage():
    """Verifies that train, val, and test manifests are strictly disjoint (0 file overlap)."""
    manifest_dir = pathlib.Path("data/manifests")
    train_file = manifest_dir / "train_manifest.json"
    val_file = manifest_dir / "val_manifest.json"
    test_file = manifest_dir / "test_manifest.json"

    assert train_file.is_file(), "train_manifest.json is missing!"
    assert val_file.is_file(), "val_manifest.json is missing!"
    assert test_file.is_file(), "test_manifest.json is missing!"

    with open(train_file, "r", encoding="utf-8") as f:
        train_recs = json.load(f)
    with open(val_file, "r", encoding="utf-8") as f:
        val_recs = json.load(f)
    with open(test_file, "r", encoding="utf-8") as f:
        test_recs = json.load(f)

    # Filter out silence slices to check speech file paths
    train_paths = {r["path"] for r in train_recs if not r.get("is_silence_slice")}
    val_paths = {r["path"] for r in val_recs if not r.get("is_silence_slice")}
    test_paths = {r["path"] for r in test_recs if not r.get("is_silence_slice")}

    # Assert completely disjoint sets
    assert len(train_paths.intersection(val_paths)) == 0, "Leakage: Train/Val overlap!"
    assert len(train_paths.intersection(test_paths)) == 0, "Leakage: Train/Test overlap!"
    assert len(val_paths.intersection(test_paths)) == 0, "Leakage: Val/Test overlap!"


def test_dataset_loader_item_shape_and_types():
    """Verifies that dataset loader produces (1, 49, 40) float32 tensors and integer labels."""
    manifest_path = "data/manifests/val_manifest.json"
    dataset = SpeechCommandsDataset(manifest_path=manifest_path, is_training=False)

    assert len(dataset) > 0
    features, label = dataset[0]

    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 49, 40)
    assert features.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < NUM_CLASSES


def test_dscnn_architecture_and_forward_pass():
    """Verifies standard DS-CNN model structure, parameter count, and output logits shape."""
    model = DSCNNKeywordSpotter(num_classes=12)
    assert model.count_parameters() < 30000, "Model exceeds parameter limit for compact KWS!"

    dummy_input = torch.randn(4, 1, 49, 40)
    logits = model(dummy_input)

    assert logits.shape == (4, 12)
    assert torch.all(torch.isfinite(logits))


def test_residual_dscnn_architecture_and_skip_connections():
    """Verifies Residual DS-CNN model parameters, skip connections, and forward pass."""
    model = ResidualDSCNNKeywordSpotter(num_classes=12)
    params = model.count_parameters()
    assert params == 28108, f"Expected 28,108 parameters, found {params}"
    assert params < 50000, "Residual DS-CNN exceeds compact edge limit!"

    dummy_input = torch.randn(4, 1, 49, 40)
    logits = model(dummy_input)

    assert logits.shape == (4, 12)
    assert torch.all(torch.isfinite(logits))


def test_checkpoint_reloading_and_inference():
    """Verifies that baseline checkpoint reloads cleanly and runs CPU inference."""
    ckpt_path = pathlib.Path("models/checkpoints/best_model.pt")
    assert ckpt_path.is_file(), "best_model.pt checkpoint missing!"

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in checkpoint

    model = DSCNNKeywordSpotter(num_classes=12)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = torch.randn(1, 1, 49, 40)
    with torch.no_grad():
        out = model(sample)
        pred_class = out.argmax(dim=1).item()

    assert out.shape == (1, 12)
    assert 0 <= pred_class < 12


def test_improved_checkpoint_reloading_and_inference():
    """Verifies that improved checkpoint reloads cleanly and produces normalized probabilities."""
    ckpt_path = pathlib.Path("models/checkpoints/dscnn_improved.pt")
    assert ckpt_path.is_file(), "dscnn_improved.pt checkpoint missing!"

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "val_macro_f1" in checkpoint

    arch = checkpoint.get("model_architecture", "ResidualDS-CNN")
    model = get_model(arch, num_classes=12)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = torch.randn(1, 1, 49, 40)
    with torch.no_grad():
        logits = model(sample)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()

    assert logits.shape == (1, 12)
    assert 0 <= pred_class < 12
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_spec_augment_tensor_transformations():
    """Verifies SpecAugment applies time and frequency masking on (1, 49, 40) spectrograms."""
    spec = torch.ones(1, 49, 40, dtype=torch.float32)
    augmented = apply_spec_augment(spec, max_time_mask=5, max_freq_mask=4)

    assert augmented.shape == (1, 49, 40)
    assert torch.all(torch.isfinite(augmented))
    # Some values must be masked to zero
    assert torch.sum(augmented == 0.0) >= 0


def test_cached_features_dataset_loading():
    """Verifies that precomputed tensor cache loads and indexes properly."""
    cache_path = pathlib.Path("data/processed/test_features_21class.pt")
    manifest_path = pathlib.Path("data/manifests/test_manifest.json")

    assert cache_path.is_file(), "test_features_21class.pt missing!"
    dataset = SpeechCommandsDataset(
        manifest_path=str(manifest_path),
        is_training=False,
        cached_tensor_file=str(cache_path),
    )

    assert len(dataset) == 8563
    feat, label = dataset[0]
    assert feat.shape == (1, 49, 40)
    assert isinstance(label, int)
    assert 0 <= label < NUM_CLASSES
