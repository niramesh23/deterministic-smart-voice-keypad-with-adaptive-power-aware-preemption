#!/usr/bin/env python3
"""
scripts/visualize_dsp.py
------------------------
Visualizes waveform, short-time RMS energy, adaptive noise floor,
VAD state transitions with pre-roll boundaries, and extracted Log-Mel features.

Saves visualization plot to results/dsp_visualization.png
and feature benchmark report to results/dsp_feature_comparison.json.
"""

import argparse
import json
import pathlib
import sys
from typing import Tuple
import wave
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless saving
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.energy import calculate_rms
from src.dsp.features import (
    LogMelSpectrogramExtractor,
    MFCCExtractor,
    benchmark_feature_extractors,
)
from src.dsp.vad import VADState, VoiceActivityDetector


def load_wav_file(wav_path: pathlib.Path) -> Tuple[np.ndarray, int]:
    """Loads a 16-bit mono WAV file."""
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        frames = w.readframes(w.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)
    return samples, sr


def run_dsp_visualization(
    wav_path_str: str = "data/raw/yes/004ae714_nohash_0.wav",
    output_image_path: str = "results/dsp_visualization.png",
    output_json_path: str = "results/dsp_feature_comparison.json",
):
    print("=" * 80)
    print("  STAGE 4: DIGITAL SIGNAL PROCESSING (DSP) VISUALIZATION & BENCHMARK")
    print("=" * 80)

    wav_path = pathlib.Path(wav_path_str)
    if not wav_path.is_file():
        # Fallback search if exact path differs
        found = list(pathlib.Path("data/raw").glob("yes/*.wav"))
        if found:
            wav_path = found[0]
        else:
            print(f"[ERROR] Audio file not found: {wav_path}")
            sys.exit(1)

    print(f"Input Speech Audio:   {wav_path}")
    raw_samples, sr = load_wav_file(wav_path)
    float_samples = raw_samples.astype(np.float32) / 32768.0
    duration_s = len(raw_samples) / sr
    print(f"Sample Rate:          {sr} Hz")
    print(f"Duration:             {duration_s:.3f} s ({len(raw_samples)} samples)")

    # 1. Feature Extraction Comparison Benchmark
    print("\n[1/3] Benchmarking feature extractors (Log-Mel vs MFCC)...")
    comparison_metrics = benchmark_feature_extractors(num_runs=30, sample_length=len(raw_samples))
    print(f"  Log-Mel Spectrogram:")
    print(f"    Shape:            {comparison_metrics['log_mel']['shape']}")
    print(f"    Mean Latency:     {comparison_metrics['log_mel']['latency_ms']:.3f} ms")
    print(f"    Memory Footprint: {comparison_metrics['log_mel']['memory_bytes']} bytes")
    print(f"  MFCC:")
    print(f"    Shape:            {comparison_metrics['mfcc']['shape']}")
    print(f"    Mean Latency:     {comparison_metrics['mfcc']['latency_ms']:.3f} ms")
    print(f"    Memory Footprint: {comparison_metrics['mfcc']['memory_bytes']} bytes")
    print(f"\n  Recommendation: {comparison_metrics['recommendation']}")

    # Save feature comparison metrics
    json_path = pathlib.Path(output_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_metrics, f, indent=2)

    # 2. Simulate Streaming VAD Processing with Hop=320
    print("\n[2/3] Simulating streaming VAD across audio hops...")
    hop_size = 320
    vad = VoiceActivityDetector(
        sample_rate=sr,
        hop_size=hop_size,
        threshold_factor=2.5,
        min_speech_duration_ms=150.0,
        hangover_duration_ms=100.0,
        pre_roll_duration_ms=200.0,
        onset_confirmation_hops=2,
        initial_noise_floor=0.005,
    )

    hop_times = []
    hop_rms = []
    noise_floors = []
    thresholds = []
    vad_states = []
    completed_utterances = []

    num_hops = len(raw_samples) // hop_size
    for i in range(num_hops):
        hop = raw_samples[i * hop_size : (i + 1) * hop_size]
        t = (i * hop_size + hop_size // 2) / sr
        hop_times.append(t)

        rms = calculate_rms(hop)
        hop_rms.append(rms)
        noise_floors.append(vad.noise_floor.noise_floor)
        thresholds.append(vad.noise_floor.threshold)

        seg = vad.process_hop(hop)
        vad_states.append(vad.state.value)
        if seg is not None:
            completed_utterances.append(seg)

    # Flush any remaining utterance at EOF
    eof_seg = vad.flush()
    if eof_seg is not None:
        completed_utterances.append(eof_seg)

    print(f"Processed {num_hops} hops.")
    print(f"Completed Utterances Detected: {len(completed_utterances)}")
    for idx, u in enumerate(completed_utterances):
        print(f"  Utterance {idx+1}: Duration={u.duration_ms:.1f} ms, Peak RMS={u.peak_rms:.4f}, Pre-roll samples={u.pre_roll_samples}")

    # 3. Generate Multi-panel Visualization
    print("\n[3/3] Generating multi-panel DSP visualization plot...")
    img_path = pathlib.Path(output_image_path)
    img_path.parent.mkdir(parents=True, exist_ok=True)

    log_mel_ext = LogMelSpectrogramExtractor(sample_rate=sr, frame_size=640, hop_size=320, num_mel_bins=40)
    log_mel_spec = log_mel_ext.extract(float_samples)

    time_axis = np.linspace(0, duration_s, len(float_samples))

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False, gridspec_kw={"height_ratios": [1, 1, 0.7, 1.3]})
    fig.suptitle(f"Real-Time DSP Pipeline Analysis: {wav_path.name}", fontsize=14, fontweight="bold")

    # Panel 1: Waveform
    axes[0].plot(time_axis, float_samples, color="#2b5c8f", lw=0.9, label="Raw Audio Signal")
    axes[0].set_ylabel("Amplitude", fontsize=10)
    axes[0].set_title("1. Acoustic Waveform (16 kHz Mono)", fontsize=11, fontweight="semibold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, duration_s])
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 2: Short-time RMS Energy, Adaptive Noise Floor, and Threshold
    axes[1].plot(hop_times, hop_rms, color="#e65100", lw=1.8, label="Frame RMS Energy")
    axes[1].plot(hop_times, thresholds, color="#c62828", lw=1.5, ls="--", label="Adaptive Trigger Threshold (2.5x)")
    axes[1].plot(hop_times, noise_floors, color="#2e7d32", lw=1.5, ls=":", label="Background Noise Floor")
    axes[1].set_ylabel("RMS Amplitude", fontsize=10)
    axes[1].set_title("2. Short-Time RMS Energy & Adaptive Noise Floor Tracking", fontsize=11, fontweight="semibold")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, duration_s])
    axes[1].legend(loc="upper right", fontsize=9)

    # Panel 3: VAD State Transitions
    state_names = ["SILENCE", "ONSET_CANDIDATE", "SPEECH_ACTIVE", "HANGOVER", "COMPLETE"]
    axes[2].step(hop_times, vad_states, where="mid", color="#6a1b9a", lw=1.8, label="VAD State Machine")
    axes[2].set_yticks(range(len(state_names)))
    axes[2].set_yticklabels(state_names, fontsize=8)
    axes[2].set_ylabel("VAD State", fontsize=10)
    axes[2].set_title("3. Voice Activity Detection (VAD) & Hangover States", fontsize=11, fontweight="semibold")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, duration_s])

    # Panel 4: Log-Mel Filterbank Spectrogram
    im = axes[3].imshow(
        log_mel_spec.T,
        origin="lower",
        aspect="auto",
        extent=[0, duration_s, 0, 40],
        cmap="magma",
    )
    axes[3].set_ylabel("Mel Frequency Bin", fontsize=10)
    axes[3].set_xlabel("Time (seconds)", fontsize=10)
    axes[3].set_title("4. Log-Mel Filterbank Energy Spectrogram (40 Bins x 49 Frames)", fontsize=11, fontweight="semibold")
    fig.colorbar(im, ax=axes[3], orientation="horizontal", pad=0.25, shrink=0.5, label="Log Energy (dB)")

    plt.tight_layout()
    plt.savefig(img_path, dpi=180)
    plt.close()

    print(f"\nVisualization figure saved to: {img_path.resolve()}")
    print(f"Feature comparison JSON saved to: {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSP pipeline visualization and feature benchmark.")
    parser.add_argument("--audio-file", type=str, default="data/raw/yes/004ae714_nohash_0.wav", help="Input audio file")
    args = parser.parse_args()

    run_dsp_visualization(wav_path_str=args.audio_file)
