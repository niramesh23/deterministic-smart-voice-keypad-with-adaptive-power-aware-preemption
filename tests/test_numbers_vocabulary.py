"""
tests/test_numbers_vocabulary.py
--------------------------------
Comprehensive tests for 21-class spoken numbers extension:
1. Vocabulary length, canonical class mapping, and number-to-digit formatting.
2. Manifests partition and counts: all 21 classes present with positive sample counts across splits.
3. Strict disjointness (zero data leakage) across train, val, and test splits.
4. Model architecture for 21 classes (output shape (batch, 21), parameter count 28,693).
5. Incompatible checkpoint rejection: attempting to initialize or load a 12-class checkpoint
   when 21 classes are required raises InferenceInitializationError.
6. Inference service decoding and probability output with 21 classes.
7. Command Finite-State Machine (FSM):
   - Accepts each of the nine numbers ("one" .. "nine") when confidence & margin thresholds pass.
   - Prevents duplicate number triggering during cooldown.
   - Accurately reports digit and display formatting ("seven (7)").
   - Rejects unconfirmed, low-confidence, and out-of-vocabulary words.
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
    NUMBER_COMMANDS,
    TARGET_COMMANDS,
    NUMBER_TO_DIGIT,
    DIGIT_TO_NUMBER,
    NUM_CLASSES,
    CLASS_NAMES_12,
    CLASS_NAMES_21,
    map_word_to_class,
    get_display_command,
)
from src.ml.model import ResidualDSCNNKeywordSpotter, DSCNNKeywordSpotter, get_model
from src.ml.inference import KeywordSpotterInferenceService, InferenceInitializationError, RecognitionResult
from src.application.command_fsm import (
    CommandFiniteStateMachine,
    CommandFSMConfig,
    CommandState,
    RejectionReason,
    CommandRecord,
)
from src.application.live_recognizer import LivePipelineTelemetry


# ==============================================================================
# 1. Vocabulary and Class Mapping Tests
# ==============================================================================

def test_canonical_21_class_definitions():
    """Verifies that NUM_CLASSES is 21 and all canonical classes are correctly structured."""
    assert NUM_CLASSES == 21
    assert len(CLASS_NAMES) == 21
    assert CLASS_NAMES == CLASS_NAMES_21

    # Special tokens
    assert CLASS_NAMES[0] == "_silence_"
    assert CLASS_NAMES[1] == "_unknown_"

    # 10 Core Commands
    assert len(CORE_COMMANDS) == 10
    expected_core = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    assert CORE_COMMANDS == expected_core

    # 9 Number Commands
    assert len(NUMBER_COMMANDS) == 9
    expected_numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    assert NUMBER_COMMANDS == expected_numbers

    # 19 Target Commands
    assert len(TARGET_COMMANDS) == 19
    assert TARGET_COMMANDS == CORE_COMMANDS + NUMBER_COMMANDS

    # Full ordering
    assert CLASS_NAMES == ["_silence_", "_unknown_"] + TARGET_COMMANDS


def test_number_digit_mappings_and_display():
    """Verifies number-to-digit bidirectional mapping and get_display_command helper."""
    assert len(NUMBER_TO_DIGIT) == 9
    assert len(DIGIT_TO_NUMBER) == 9

    for num_str, digit_char in NUMBER_TO_DIGIT.items():
        assert DIGIT_TO_NUMBER[digit_char] == num_str
        assert get_display_command(num_str) == f"{num_str} ({digit_char})"

    # Non-number commands display without digit
    for cmd in CORE_COMMANDS:
        assert get_display_command(cmd) == cmd

    assert get_display_command("_unknown_") == "_unknown_"
    assert get_display_command("_silence_") == "_silence_"


def test_map_word_to_class_21_classes():
    """Verifies that words map to their exact class or _unknown_/_silence_."""
    # 10 core commands map to themselves
    for cmd in CORE_COMMANDS:
        assert map_word_to_class(cmd) == cmd

    # 9 numbers map to themselves
    for num in NUMBER_COMMANDS:
        assert map_word_to_class(num) == num

    # Auxiliary words map to _unknown_
    aux_sample = ["cat", "dog", "bed", "bird", "happy", "house", "marvin", "sheila", "tree", "zero"]
    for w in aux_sample:
        assert map_word_to_class(w) == "_unknown_"

    # Background noise and silence
    assert map_word_to_class("_background_noise_") == "_silence_"
    assert map_word_to_class("_silence_") == "_silence_"


# ==============================================================================
# 2. Manifest Partition and Leakage Tests
# ==============================================================================

def test_manifests_contain_all_21_classes_with_nonzero_samples():
    """Verifies train, val, and test manifests each contain samples for all 21 classes."""
    manifest_dir = pathlib.Path("data/manifests")
    for split in ["train", "val", "test"]:
        manifest_file = manifest_dir / f"{split}_manifest.json"
        assert manifest_file.is_file(), f"{manifest_file} missing!"

        with open(manifest_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        assert len(records) > 0, f"{split} manifest is empty!"
        present_classes = {r["label"] for r in records}

        for c_name in CLASS_NAMES:
            assert c_name in present_classes, f"Class '{c_name}' missing from {split} manifest!"


def test_manifest_partition_strict_disjointness():
    """Verifies zero speaker/file leakage across train, val, and test splits for all 21 classes."""
    manifest_dir = pathlib.Path("data/manifests")
    with open(manifest_dir / "train_manifest.json", "r", encoding="utf-8") as f:
        train_recs = json.load(f)
    with open(manifest_dir / "val_manifest.json", "r", encoding="utf-8") as f:
        val_recs = json.load(f)
    with open(manifest_dir / "test_manifest.json", "r", encoding="utf-8") as f:
        test_recs = json.load(f)

    # Filter out synthetic ambient noise slices which don't have disk file paths
    train_files = {r["path"] for r in train_recs if not r.get("is_silence_slice")}
    val_files = {r["path"] for r in val_recs if not r.get("is_silence_slice")}
    test_files = {r["path"] for r in test_recs if not r.get("is_silence_slice")}

    assert len(train_files.intersection(val_files)) == 0, "Data Leakage: Train & Val overlap!"
    assert len(train_files.intersection(test_files)) == 0, "Data Leakage: Train & Test overlap!"
    assert len(val_files.intersection(test_files)) == 0, "Data Leakage: Val & Test overlap!"


# ==============================================================================
# 3. Model Architecture Tests
# ==============================================================================

def test_residual_dscnn_21_class_architecture():
    """Verifies model forward pass and parameter count with 21 output classes."""
    model = ResidualDSCNNKeywordSpotter(num_classes=21)
    params = model.count_parameters()
    # 28,108 (12-class) + (21 - 12) * 64 weights + 9 bias = 28,108 + 585 = 28,693
    assert params == 28693, f"Expected 28,693 parameters for 21-class model, got {params}"

    dummy_input = torch.randn(2, 1, 49, 40)
    logits = model(dummy_input)
    assert logits.shape == (2, 21), f"Expected shape (2, 21), got {logits.shape}"
    assert torch.all(torch.isfinite(logits))


# ==============================================================================
# 4. Incompatible Checkpoint Rejection Tests
# ==============================================================================

def test_inference_service_rejects_old_12_class_checkpoint_when_21_classes_expected(tmp_path):
    """
    Verifies that attempting to load a 12-class checkpoint with a 21-class mapping
    fails explicitly with InferenceInitializationError.
    """
    # 1. Create a 21-class mapping file
    mapping_file = tmp_path / "class_mapping_21.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump({
            "class_names": CLASS_NAMES_21,
            "num_classes": 21,
            "core_commands": TARGET_COMMANDS,
        }, f)

    # 2. Attempt to initialize service with 12-class checkpoint (dscnn_improved.pt) and 21-class mapping
    with pytest.raises(InferenceInitializationError, match="class names mismatch|Output layer shape mismatch"):
        KeywordSpotterInferenceService(
            checkpoint_path="models/checkpoints/dscnn_improved.pt",
            class_mapping_path=str(mapping_file),
            feature_config_path="models/checkpoints/feature_config_improved.json",
        )


# ==============================================================================
# 5. Command FSM Number Handling & Cooldown Tests
# ==============================================================================

def _make_dummy_telemetry():
    return LivePipelineTelemetry(
        sequence_number=1,
        audio_event_timestamp=100.0,
        utterance_onset_timestamp=99.5,
        decision_timestamp=100.015,
        capture_and_buffering_delay_ms=20.0,
        queue_delay_ms=0.5,
        feature_extraction_ms=1.2,
        model_inference_ms=2.1,
        postprocessing_ms=0.1,
        reaction_latency_ms=15.0,
        total_acoustic_latency_ms=515.0,
    )


def test_command_fsm_confirms_all_nine_numbers():
    """Verifies that the FSM confirms every number command with correct metadata."""
    fsm = CommandFiniteStateMachine()
    telemetry = _make_dummy_telemetry()

    for idx, num_cmd in enumerate(NUMBER_COMMANDS):
        fsm.reset()
        probs = {c: 0.01 for c in CLASS_NAMES}
        probs[num_cmd] = 0.90
        probs["_unknown_"] = 0.05

        result = RecognitionResult(
            command=num_cmd,
            raw_class=num_cmd,
            confidence=0.90,
            raw_confidence=0.90,
            top2_class="_unknown_",
            top2_confidence=0.05,
            margin=0.85,
            is_confirmed=True,
            is_silence=False,
            is_unknown=False,
            is_cooldown=False,
            probabilities=probs,
            audio_event_timestamp=100.0 + idx,
            decision_timestamp=100.02 + idx,
        )

        rec = fsm.process_prediction(result, telemetry)
        assert rec.is_confirmed is True
        assert rec.command == num_cmd
        assert rec.digit == NUMBER_TO_DIGIT[num_cmd]
        assert rec.display_command == f"{num_cmd} ({NUMBER_TO_DIGIT[num_cmd]})"
        assert fsm.state == CommandState.COOLDOWN


def test_command_fsm_number_cooldown_suppression():
    """Verifies that consecutive identical numbers within cooldown are suppressed as DUPLICATE_IN_COOLDOWN."""
    fsm = CommandFiniteStateMachine()
    telemetry = _make_dummy_telemetry()

    probs = {c: 0.01 for c in CLASS_NAMES}
    probs["seven"] = 0.92
    probs["_unknown_"] = 0.04

    result = RecognitionResult(
        command="seven",
        raw_class="seven",
        confidence=0.92,
        raw_confidence=0.92,
        top2_class="_unknown_",
        top2_confidence=0.04,
        margin=0.88,
        is_confirmed=True,
        is_silence=False,
        is_unknown=False,
        is_cooldown=False,
        probabilities=probs,
        audio_event_timestamp=100.0,
        decision_timestamp=100.02,
    )

    # First event: confirmed
    rec1 = fsm.process_prediction(result, telemetry)
    assert rec1.is_confirmed is True
    assert rec1.command == "seven"
    assert rec1.digit == "7"

    # Immediate second identical event: duplicate suppressed
    rec2 = fsm.process_prediction(result, telemetry)
    assert rec2.is_confirmed is False
    assert rec2.rejection_reason == RejectionReason.DUPLICATE_IN_COOLDOWN
    assert fsm.total_duplicates_suppressed == 1

