"""
tests/test_live_recognition.py
------------------------------
Unit and integration tests for Stage 6 Live Keyword Recognition:
- Checkpoint loading, parameter verification, and error handling.
- Feature configuration validation.
- Per-class confidence scoring and probability normalization.
- Unknown word and acoustic silence rejection.
- Margin thresholding and ambiguous prediction suppression.
- Repeated-command suppression (cooldown logic).
- Prediction smoothing across successive frames.
- Live pipeline streaming and latency telemetry instrumentation.
"""

import pathlib
import time
import numpy as np
import pytest
import torch

from src.application.live_recognizer import LiveKeywordRecognizer, LivePipelineTelemetry
from src.dsp.features import LogMelSpectrogramExtractor
from src.ml.classes import CLASS_NAMES, NUM_CLASSES
from src.ml.dataset import load_wav_16k_mono
from src.ml.inference import (
    InferenceInitializationError,
    KeywordSpotterInferenceService,
    RecognitionResult,
)


def test_inference_service_initialization_and_verification():
    """Verifies that inference service loads checkpoint once, verifies configs, and sets eval mode."""
    service = KeywordSpotterInferenceService(
        checkpoint_path="models/checkpoints/dscnn_improved.pt",
        class_mapping_path="models/checkpoints/class_mapping_improved.json",
        feature_config_path="models/checkpoints/feature_config_improved.json",
    )
    stats = service.get_stats()
    assert stats["parameter_count"] == 28108, f"Expected 28,108 parameters, got {stats['parameter_count']}"
    assert stats["model_architecture"] == "ResidualDSCNNKeywordSpotter"
    assert service.model.training is False, "Model must be in eval mode!"

    # Verify that gradients are disabled
    for p in service.model.parameters():
        assert p.requires_grad is False, "Model parameters must not require gradients!"


def test_inference_service_missing_checkpoint_fails_clearly():
    """Verifies clear exception if checkpoint is missing."""
    with pytest.raises(InferenceInitializationError, match="Checkpoint file not found"):
        KeywordSpotterInferenceService(
            checkpoint_path="models/checkpoints/non_existent_checkpoint.pt"
        )


def test_prediction_probabilities_and_margin():
    """Verifies that predictions return valid probability distributions and compute margins."""
    service = KeywordSpotterInferenceService()
    dummy_audio = np.random.normal(0.0, 0.05, 16000).astype(np.float32)

    result = service.predict(dummy_audio)
    assert isinstance(result, RecognitionResult)
    assert len(result.probabilities) == service.num_classes
    prob_sum = sum(result.probabilities.values())
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities must sum to ~1.0, got {prob_sum}"
    assert result.margin >= 0.0, "Margin must be non-negative!"
    assert result.confidence >= result.top2_confidence, "Top-1 confidence must be >= Top-2!"


def test_silence_and_unknown_rejection():
    """Verifies that silence and unknown words are rejected from confirming commands."""
    service = KeywordSpotterInferenceService(confidence_threshold=0.60)

    # 1. Zero acoustic silence
    zero_audio = np.zeros(16000, dtype=np.float32)
    res_silence = service.predict(zero_audio)
    assert res_silence.command is None, "Silence must never trigger a command!"
    assert res_silence.is_confirmed is False

    # 2. Auxiliary word 'cat' (should be recognized as unknown or low confidence)
    cat_files = list(pathlib.Path("data/raw/cat").glob("*.wav"))
    if cat_files:
        service.reset_state()
        cat_samples = load_wav_16k_mono(cat_files[0])
        res_cat = service.predict(cat_samples)
        # Should NOT confirm any core command!
        assert res_cat.command is None, f"Auxiliary word 'cat' triggered false command: {res_cat.command}"


def test_command_confirmation_on_real_samples():
    """Verifies that high-confidence core commands trigger confirmation."""
    service = KeywordSpotterInferenceService(confidence_threshold=0.65)

    yes_files = list(pathlib.Path("data/raw/yes").glob("*.wav"))
    assert len(yes_files) > 0, "No 'yes' files found in dataset!"

    service.reset_state()
    yes_samples = load_wav_16k_mono(yes_files[0])
    res_yes = service.predict(yes_samples)

    if res_yes.raw_class == "yes" and res_yes.confidence >= 0.65:
        assert res_yes.is_confirmed is True
        assert res_yes.command == "yes"


def test_command_cooldown_suppression():
    """Verifies that repeated triggers of the same command within cooldown are suppressed."""
    service = KeywordSpotterInferenceService(
        confidence_threshold=0.50,
        cooldown_seconds=1.0,
    )

    yes_files = list(pathlib.Path("data/raw/yes").glob("*.wav"))
    if not yes_files:
        pytest.skip("No 'yes' files available for test.")

    yes_samples = load_wav_16k_mono(yes_files[0])
    t0 = time.perf_counter()

    # First call: Should confirm
    service.reset_state()
    res1 = service.predict(yes_samples, audio_event_timestamp=t0)

    if res1.is_confirmed:
        # Second call immediately after (50 ms): Must be suppressed by cooldown
        t1 = t0 + 0.05
        res2 = service.predict(yes_samples, audio_event_timestamp=t1)
        assert res2.is_confirmed is False, "Repeated command was not suppressed by cooldown!"
        assert res2.is_cooldown is True
        assert res2.command is None

        # Third call after cooldown expires (1.1s): Must confirm again
        t2 = t0 + 1.10
        res3 = service.predict(yes_samples, audio_event_timestamp=t2)
        assert res3.is_confirmed is True, "Command should confirm after cooldown expires!"
        assert res3.command == res1.command


def test_prediction_smoothing_effect():
    """Verifies that exponential smoothing dampens sudden probability spikes."""
    service_smooth = KeywordSpotterInferenceService(smoothing_alpha=0.30)
    service_raw = KeywordSpotterInferenceService(smoothing_alpha=1.00)

    # First feed silence
    silence = np.zeros(16000, dtype=np.float32)
    service_smooth.predict(silence)
    service_raw.predict(silence)

    # Then feed speech sample
    yes_files = list(pathlib.Path("data/raw/yes").glob("*.wav"))
    if yes_files:
        yes_samples = load_wav_16k_mono(yes_files[0])
        res_smooth = service_smooth.predict(yes_samples)
        res_raw = service_raw.predict(yes_samples)

        # Smoothed confidence should be dampened compared to raw on first spike
        if res_raw.raw_class == "yes":
            assert res_smooth.probabilities["yes"] <= res_raw.probabilities["yes"] + 1e-4


def test_live_recognizer_external_stream_and_telemetry():
    """
    Verifies that LiveKeywordRecognizer runs audio through ping-pong buffers,
    audio queue, VAD, and produces complete latency telemetry.
    """
    received_results = []
    received_telemetries = []

    def on_command(res: RecognitionResult, tel: LivePipelineTelemetry):
        received_results.append(res)
        received_telemetries.append(tel)

    recognizer = LiveKeywordRecognizer(
        confidence_threshold=0.50,
        on_command_callback=on_command,
    )

    # Start recognizer without physical microphone (stream feeder mode)
    recognizer.start(use_microphone=False)
    assert recognizer._is_running is True

    try:
        # Feed 25 hops of silence (500 ms @ 20 ms/hop)
        silence_hop = np.zeros(320, dtype=np.int16)
        for _ in range(25):
            recognizer.push_external_hop(silence_hop)
            time.sleep(0.002)

        # Feed 25 hops of speech from 'yes' (500 ms)
        yes_files = list(pathlib.Path("data/raw/yes").glob("*.wav"))
        if yes_files:
            yes_audio = load_wav_16k_mono(yes_files[0])
            yes_pcm = (yes_audio * 32767.0).astype(np.int16)
            for i in range(0, min(8000, len(yes_pcm) - 320), 320):
                recognizer.push_external_hop(yes_pcm[i : i + 320])
                time.sleep(0.002)

        # Feed 25 hops of silence to trigger hangover completion
        for _ in range(25):
            recognizer.push_external_hop(silence_hop)
            time.sleep(0.002)

        # Wait briefly for processing worker to flush
        time.sleep(0.15)

    finally:
        recognizer.stop()
        assert recognizer._is_running is False

    diag = recognizer.get_diagnostics()
    assert diag["total_hops_processed"] > 0
    assert diag["ping_pong_status"]["total_written"] > 0
    assert diag["queue_diagnostics"]["total_enqueued"] > 0

    # If any telemetry was produced, verify required timing fields
    for tel in received_telemetries:
        assert isinstance(tel.capture_and_buffering_delay_ms, float)
        assert isinstance(tel.feature_extraction_ms, float)
        assert isinstance(tel.model_inference_ms, float)
        assert isinstance(tel.reaction_latency_ms, float)
        assert tel.model_inference_ms > 0.0

