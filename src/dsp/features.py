"""
src/dsp/features.py
-------------------
Audio feature extraction module providing Log-Mel Spectrogram and MFCC representations.

Guarantees:
- Deterministic, finite outputs (no NaNs, no Infs).
- Identical pipeline for offline dataset training and real-time live inference.
- Pure NumPy + SciPy implementation for reproducible low-latency execution.
"""

import math
import time
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.fftpack import dct


def create_mel_filterbank(
    num_filters: int = 40,
    fft_size: int = 640,
    sample_rate: int = 16000,
    low_freq: float = 20.0,
    high_freq: float = 8000.0,
) -> np.ndarray:
    """
    Constructs a triangular Mel-spaced filterbank matrix.

    Returns:
        np.ndarray of shape (num_filters, fft_size // 2 + 1)
    """
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    # Convert Hz points to FFT bin indices
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    num_bins = fft_size // 2 + 1
    filterbank = np.zeros((num_filters, num_bins), dtype=np.float32)

    for i in range(1, num_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        # Upward slope
        if center > left:
            for j in range(left, center):
                if 0 <= j < num_bins:
                    filterbank[i - 1, j] = (j - left) / (center - left)

        # Downward slope
        if right > center:
            for j in range(center, right):
                if 0 <= j < num_bins:
                    filterbank[i - 1, j] = (right - j) / (right - center)

    return filterbank


class LogMelSpectrogramExtractor:
    """
    Extracts Log-Mel Filterbank Energy Spectrograms.

    Converts 1.0s (16,000 samples) of audio into a 2D matrix of shape (num_frames, num_mel_bins).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 640,
        hop_size: int = 320,
        num_mel_bins: int = 40,
        fft_size: int = 640,
        low_freq: float = 20.0,
        high_freq: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.num_mel_bins = num_mel_bins
        self.fft_size = fft_size

        # Precompute Hann window and Mel filterbank
        self.window = np.hanning(frame_size).astype(np.float32)
        self.filterbank = create_mel_filterbank(
            num_filters=num_mel_bins,
            fft_size=fft_size,
            sample_rate=sample_rate,
            low_freq=low_freq,
            high_freq=high_freq,
        )

    def extract(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extracts log-Mel spectrogram from 1D audio samples.

        Parameters:
            audio_data: 1D array of float32 samples in [-1.0, 1.0] or int16.

        Returns:
            np.ndarray of shape (num_frames, num_mel_bins) in float32.
        """
        if audio_data.dtype == np.int16:
            samples = audio_data.astype(np.float32) / 32768.0
        else:
            samples = audio_data.astype(np.float32)

        if len(samples) < self.frame_size:
            # Pad with zeros if shorter than one frame
            pad = np.zeros(self.frame_size - len(samples), dtype=np.float32)
            samples = np.concatenate([samples, pad])

        # Step 1: Framing with overlap
        num_frames = 1 + int((len(samples) - self.frame_size) // self.hop_size)
        frames = np.lib.stride_tricks.sliding_window_view(samples, self.frame_size)[::self.hop_size]

        # Step 2: Apply Hann Window
        windowed = frames * self.window

        # Step 3: Compute Real FFT power spectrum
        spectra = np.abs(rfft(windowed, n=self.fft_size, axis=-1)) ** 2

        # Step 4: Apply Mel Filterbank
        # spectra shape: (num_frames, num_bins), filterbank shape: (num_mel_bins, num_bins)
        mel_energies = np.dot(spectra, self.filterbank.T)

        # Step 5: Logarithmic compression (add eps to prevent log(0))
        eps = 1e-6
        log_mel = np.log(mel_energies + eps).astype(np.float32)

        # Sanity check: Ensure finite values
        if not np.all(np.isfinite(log_mel)):
            log_mel = np.nan_to_num(log_mel, nan=-13.8, posinf=0.0, neginf=-13.8)

        return log_mel


class MFCCExtractor:
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCC).
    Applies Discrete Cosine Transform (DCT Type-II) to Log-Mel Spectrogram.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 640,
        hop_size: int = 320,
        num_mel_bins: int = 40,
        num_mfcc: int = 40,
    ):
        self.num_mfcc = num_mfcc
        self.log_mel_extractor = LogMelSpectrogramExtractor(
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            num_mel_bins=num_mel_bins,
        )

    def extract(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extracts MFCC feature matrix.

        Returns:
            np.ndarray of shape (num_frames, num_mfcc) in float32.
        """
        log_mel = self.log_mel_extractor.extract(audio_data)
        # Apply DCT-II across the filterbank dimension
        mfcc = dct(log_mel, type=2, axis=-1, norm="ortho")
        return mfcc[:, :self.num_mfcc].astype(np.float32)


def benchmark_feature_extractors(
    num_runs: int = 50, sample_length: int = 16000
) -> Dict[str, any]:
    """
    Compares Log-Mel Spectrogram vs MFCC computational latency, memory footprint,
    and output dimensions on 1.0 second 16 kHz audio.
    """
    synth_audio = np.random.uniform(-0.5, 0.5, sample_length).astype(np.float32)

    logmel_ext = LogMelSpectrogramExtractor()
    mfcc_ext = MFCCExtractor()

    # Warmup
    _ = logmel_ext.extract(synth_audio)
    _ = mfcc_ext.extract(synth_audio)

    # Benchmark Log-Mel
    t0 = time.perf_counter()
    for _ in range(num_runs):
        mel_feat = logmel_ext.extract(synth_audio)
    t_mel = (time.perf_counter() - t0) / num_runs * 1000.0

    # Benchmark MFCC
    t0 = time.perf_counter()
    for _ in range(num_runs):
        mfcc_feat = mfcc_ext.extract(synth_audio)
    t_mfcc = (time.perf_counter() - t0) / num_runs * 1000.0

    recommendation = (
        "Log-Mel Spectrogram is recommended for the initial 2D Convolutional Neural Network (DS-CNN). "
        "Log-Mel filterbanks preserve 2D time-frequency locality and formant structure that 2D convolutional kernels exploit. "
        "The additional DCT step in MFCC decorrelates coefficients (historically required for GMMs), but disrupts spatial "
        "correlation and incurs extra computational overhead in real-time execution."
    )

    return {
        "log_mel": {
            "shape": list(mel_feat.shape),
            "latency_ms": t_mel,
            "memory_bytes": mel_feat.nbytes,
            "is_finite": bool(np.all(np.isfinite(mel_feat))),
        },
        "mfcc": {
            "shape": list(mfcc_feat.shape),
            "latency_ms": t_mfcc,
            "memory_bytes": mfcc_feat.nbytes,
            "is_finite": bool(np.all(np.isfinite(mfcc_feat))),
        },
        "recommendation": recommendation,
        "selected_feature": "log_mel",
    }

