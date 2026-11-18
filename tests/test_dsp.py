"""
tests/test_dsp.py
-----------------
Automated test suite for Digital Signal Processing (DSP) subsystem:
- Audio framing (hop 320 vs frame 640, incomplete block handling)
- RMS energy (silence, constant, sine tone A/sqrt(2), noise)
- Adaptive noise floor tracking (quiet, noisy, freeze on speech)
- Pre-roll buffer onset preservation
- Voice Activity Detection (VAD) and hangover gap bridging
- Transient noise rejection
- Feature extraction determinism, finiteness, and shapes
"""

import math
import numpy as np
import pytest

from src.audio.framing import SlidingWindowFramer, pcm16_to_float32
from src.dsp.energy import calculate_rms, calculate_frame_energy, rms_to_db
from src.dsp.noise import AdaptiveNoiseFloor
from src.dsp.preroll import PreRollBuffer
from src.dsp.vad import VADState, VoiceActivityDetector, UtteranceSegment
from src.dsp.features import (
    LogMelSpectrogramExtractor,
    MFCCExtractor,
    benchmark_feature_extractors,
)


# ==============================================================================
# Task A: Audio Framing Tests
# ==============================================================================

def test_audio_framing_hop_vs_frame():
    """
    Verifies 20 ms hop (320 samples) and 40 ms frame (640 samples) sliding window.
    Documents that hop controls update frequency while frame controls spectral resolution.
    """
    framer = SlidingWindowFramer(frame_size=640, hop_size=320)

    hop1 = np.arange(320, dtype=np.int16)
    hop2 = np.arange(320, 640, dtype=np.int16)

    # Frame 1: [zero_history (320) + hop1 (320)]
    f1 = framer.process_hop(hop1)
    assert len(f1) == 640
    assert f1.dtype == np.int16
    np.testing.assert_array_equal(f1[320:], hop1)

    # Frame 2: [hop1 (320) + hop2 (320)]
    f2 = framer.process_hop(hop2)
    assert len(f2) == 640
    np.testing.assert_array_equal(f2[:320], hop1)
    np.testing.assert_array_equal(f2[320:], hop2)


def test_audio_framing_invalid_hop_size():
    """Verifies that invalid hop lengths raise ValueError."""
    framer = SlidingWindowFramer(frame_size=640, hop_size=320)
    with pytest.raises(ValueError, match="Expected hop size 320"):
        framer.process_hop(np.zeros(200, dtype=np.int16))


# ==============================================================================
# Task B: RMS Energy Tests
# ==============================================================================

def test_rms_silence():
    """Verifies that pure silence yields exactly 0.0 RMS."""
    silence = np.zeros(640, dtype=np.float32)
    assert calculate_rms(silence) == 0.0
    assert calculate_frame_energy(silence) == 0.0
    assert rms_to_db(0.0) == -120.0


def test_rms_constant_signal():
    """Verifies that a constant DC signal of value C yields RMS = |C|."""
    val = 0.42
    dc_signal = np.full(640, val, dtype=np.float32)
    assert pytest.approx(calculate_rms(dc_signal), rel=1e-5) == val


def test_rms_synthetic_sine_tone():
    """
    Verifies that a pure sine wave of amplitude A has theoretical RMS = A / sqrt(2).
    """
    sample_rate = 16000
    freq = 440.0  # 440 Hz (A4)
    amplitude = 0.8
    num_samples = 16000  # 1.0 second (integer number of cycles)

    t = np.arange(num_samples) / sample_rate
    sine_wave = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    expected_rms = amplitude / math.sqrt(2.0)  # ~0.565685
    measured_rms = calculate_rms(sine_wave)

    assert pytest.approx(measured_rms, rel=1e-3) == expected_rms


def test_rms_generated_gaussian_noise():
    """Verifies that Gaussian noise with standard deviation sigma yields RMS ~ sigma."""
    sigma = 0.15
    np.random.seed(42)
    noise = np.random.normal(loc=0.0, scale=sigma, size=32000).astype(np.float32)
    measured_rms = calculate_rms(noise)
    assert pytest.approx(measured_rms, rel=0.05) == sigma


# ==============================================================================
# Task C: Adaptive Noise Floor Tests
# ==============================================================================

def test_noise_floor_quiet_convergence():
    """Verifies that noise floor converges downwards in quiet conditions."""
    estimator = AdaptiveNoiseFloor(initial_noise_floor=0.02, threshold_factor=2.5)

    # Feed quiet frames with RMS = 0.002
    for _ in range(50):
        estimator.update(0.002, is_speech_active=False)

    assert estimator.noise_floor < 0.01
    assert estimator.noise_floor >= estimator.min_noise_floor
    assert estimator.threshold == pytest.approx(estimator.noise_floor * 2.5)


def test_noise_floor_speech_protection_freeze():
    """Verifies that noise floor NEVER adapts on active speech energy."""
    estimator = AdaptiveNoiseFloor(initial_noise_floor=0.005, threshold_factor=2.5)
    initial_floor = estimator.noise_floor

    # Feed high energy speech (RMS = 0.25) with is_speech_active=True
    for _ in range(20):
        estimator.update(0.25, is_speech_active=True)

    # Noise floor must not have moved up into the speech signal
    assert estimator.noise_floor == initial_floor


def test_noise_floor_bounded_rate_of_change():
    """Verifies that sudden noise step changes are rate-limited per hop."""
    estimator = AdaptiveNoiseFloor(
        initial_noise_floor=0.005,
        threshold_factor=2.5,
        max_delta_per_hop=0.001,
    )
    initial_floor = estimator.noise_floor

    # One hop of slightly elevated noise within threshold (e.g. 0.010, while thresh is 0.0125)
    estimator.update(0.010, is_speech_active=False)

    # Change cannot exceed max_delta_per_hop
    assert abs(estimator.noise_floor - initial_floor) <= 0.001001


# ==============================================================================
# Task E: Pre-Roll Buffer Tests
# ==============================================================================

def test_preroll_buffer_preserves_onset_samples():
    """
    Verifies that the pre-roll buffer retains exactly 200 ms (3,200 samples)
    of preceding audio in correct chronological order.
    """
    preroll = PreRollBuffer(sample_rate=16000, duration_ms=200.0, hop_size=320)
    assert preroll.capacity_samples == 3200

    # Push 15 hops tagged with index 0..14
    for i in range(15):
        hop_data = np.full(320, i, dtype=np.int16)
        preroll.push_hop(hop_data)

    # Retrieve pre-roll: should contain the latest 10 hops (hops 5..14)
    samples = preroll.get_preroll_samples()
    assert len(samples) == 3200

    # Verify first 320 samples are from hop 5, and last 320 are from hop 14
    assert np.all(samples[:320] == 5)
    assert np.all(samples[-320:] == 14)


# ==============================================================================
# Task D: VAD and Hangover Tests
# ==============================================================================

def test_vad_speech_activation_and_hangover_gap():
    """
    Tests complete VAD lifecycle:
    Silence -> Onset Confirmation -> Active Speech -> Brief Gap (Hangover) -> Resumed Speech -> Termination.
    """
    vad = VoiceActivityDetector(
        sample_rate=16000,
        hop_size=320,
        threshold_factor=2.5,
        min_speech_duration_ms=100.0,  # 5 hops min
        hangover_duration_ms=80.0,     # 4 hops hangover
        pre_roll_duration_ms=200.0,    # 10 hops preroll
        onset_confirmation_hops=2,
        initial_noise_floor=0.005,
    )

    quiet_hop = np.full(320, 50, dtype=np.int16)     # RMS ~ 0.0015 (quiet)
    speech_hop = np.full(320, 5000, dtype=np.int16)  # RMS ~ 0.1526 (loud speech)

    # 1. Feed 12 quiet hops to populate pre-roll and settle noise floor
    for _ in range(12):
        seg = vad.process_hop(quiet_hop)
        assert seg is None
        assert vad.state == VADState.SILENCE

    # 2. Feed 1st speech hop -> triggers ONSET_CANDIDATE
    seg = vad.process_hop(speech_hop)
    assert seg is None
    assert vad.state == VADState.ONSET_CANDIDATE

    # 3. Feed 2nd speech hop -> confirms onset, enters SPEECH_ACTIVE
    seg = vad.process_hop(speech_hop)
    assert seg is None
    assert vad.state == VADState.SPEECH_ACTIVE

    # 4. Feed 4 more speech hops
    for _ in range(4):
        assert vad.process_hop(speech_hop) is None
        assert vad.state == VADState.SPEECH_ACTIVE

    # 5. Brief 2-hop silence gap (e.g. stop consonant closure) -> enters HANGOVER
    vad.process_hop(quiet_hop)
    assert vad.state == VADState.HANGOVER
    vad.process_hop(quiet_hop)
    assert vad.state == VADState.HANGOVER

    # 6. Speech resumes -> immediately returns to SPEECH_ACTIVE! (Hangover prevented split)
    vad.process_hop(speech_hop)
    assert vad.state == VADState.SPEECH_ACTIVE

    # 7. Final silence exceeding hangover duration (4 hops)
    seg = None
    for _ in range(5):
        res = vad.process_hop(quiet_hop)
        if res is not None:
            seg = res

    # 8. Utterance must have finalized cleanly!
    assert seg is not None
    assert isinstance(seg, UtteranceSegment)
    assert seg.pre_roll_samples == 3200  # 200 ms preroll prepended
    assert len(seg.audio) > 3200
    assert seg.duration_ms >= 200.0
    assert vad.state == VADState.SILENCE


def test_vad_single_click_transient_rejection():
    """
    Verifies that a 1-hop click followed by silence is rejected as a transient
    and does NOT produce an utterance.
    """
    vad = VoiceActivityDetector(
        sample_rate=16000,
        hop_size=320,
        onset_confirmation_hops=2,  # requires 2 consecutive hops
    )

    quiet_hop = np.zeros(320, dtype=np.int16)
    click_hop = np.full(320, 10000, dtype=np.int16)  # 1 loud click

    # Warmup
    for _ in range(5):
        vad.process_hop(quiet_hop)

    # 1 single loud hop
    seg = vad.process_hop(click_hop)
    assert seg is None
    assert vad.state == VADState.ONSET_CANDIDATE

    # Next hop is quiet -> false trigger cancelled!
    seg = vad.process_hop(quiet_hop)
    assert seg is None
    assert vad.state == VADState.SILENCE
    assert vad.transients_rejected >= 1


# ==============================================================================
# Task F: Feature Extraction Tests
# ==============================================================================

def test_feature_extractor_shapes_and_finiteness():
    """
    Verifies Log-Mel and MFCC feature extractors on 1.0 second 16 kHz audio.
    Expected frames: 49 frames (hop 320, frame 640), 40 Mel bins.
    """
    audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)

    log_mel_ext = LogMelSpectrogramExtractor(
        sample_rate=16000, frame_size=640, hop_size=320, num_mel_bins=40
    )
    mfcc_ext = MFCCExtractor(
        sample_rate=16000, frame_size=640, hop_size=320, num_mel_bins=40, num_mfcc=40
    )

    mel_features = log_mel_ext.extract(audio)
    mfcc_features = mfcc_ext.extract(audio)

    assert mel_features.shape == (49, 40)
    assert mfcc_features.shape == (49, 40)

    # Assert 100% finite values (no NaN, no Inf)
    assert np.all(np.isfinite(mel_features))
    assert np.all(np.isfinite(mfcc_features))


def test_feature_extractor_determinism():
    """Verifies that feature extraction is strictly deterministic under fixed inputs."""
    audio = np.sin(np.linspace(0, 100 * np.pi, 16000)).astype(np.float32)

    ext = LogMelSpectrogramExtractor()
    feat1 = ext.extract(audio)
    feat2 = ext.extract(audio)

    np.testing.assert_array_equal(feat1, feat2)


def test_feature_extractor_benchmark_report():
    """Verifies the comparative benchmark function between Log-Mel and MFCC."""
    results = benchmark_feature_extractors(num_runs=10)
    assert "log_mel" in results
    assert "mfcc" in results
    assert results["log_mel"]["shape"] == [49, 40]
    assert results["mfcc"]["shape"] == [49, 40]
    assert results["log_mel"]["latency_ms"] < 10.0
    assert results["mfcc"]["latency_ms"] < 10.0
    assert "recommendation" in results

