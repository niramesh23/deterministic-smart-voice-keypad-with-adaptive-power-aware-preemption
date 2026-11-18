"""
dsp Subsystem
=============
Responsibilities:
- Short-time frame energy, Root Mean Square (RMS), and power spectral density computation.
- Background acoustic noise estimation and adaptive noise floor tracking.
- Voice Activity Detection (VAD) state machine (SPEECH_START, SPEECH_ACTIVE, SPEECH_END).
- Pre-roll circular ring buffering to preserve speech onset syllables prior to trigger.
- Time-to-frequency transformations and feature extraction (e.g. Log-Mel Filterbanks, MFCCs, Spectrograms).

Design Principles:
- Pure functional/stateless DSP primitives where feasible for testability.
- State machines encapsulated within dedicated VAD and buffer tracker classes.
- Zero copy array slicing to minimize latency during frame analysis.
"""

from src.dsp.energy import calculate_rms, calculate_frame_energy, rms_to_db
from src.dsp.noise import AdaptiveNoiseFloor
from src.dsp.preroll import PreRollBuffer
from src.dsp.vad import VADState, UtteranceSegment, VoiceActivityDetector
from src.dsp.features import (
    LogMelSpectrogramExtractor,
    MFCCExtractor,
    create_mel_filterbank,
    benchmark_feature_extractors,
)

__all__ = [
    "calculate_rms",
    "calculate_frame_energy",
    "rms_to_db",
    "AdaptiveNoiseFloor",
    "PreRollBuffer",
    "VADState",
    "UtteranceSegment",
    "VoiceActivityDetector",
    "LogMelSpectrogramExtractor",
    "MFCCExtractor",
    "create_mel_filterbank",
    "benchmark_feature_extractors",
]
