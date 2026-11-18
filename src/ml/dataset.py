"""
src/ml/dataset.py
-----------------
PyTorch Dataset and DataLoader for Keyword Spotting (KWS).

Guarantees:
- Safe 16-bit PCM reading and length normalization (16,000 samples = 1.000s @ 16 kHz).
- Extracts Log-Mel Filterbanks using the exact frozen Stage 4 DSP configuration.
- Identical preprocessing path for training, validation, testing, and live inference.
- Documented training data augmentation (time-shift, volume scaling, SpecAugment).
- Fast feature caching / pre-extraction for high-throughput CPU training.
"""

import json
import math
import os
import pathlib
import random
import sys
import time
import wave
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.features import LogMelSpectrogramExtractor
from src.ml.classes import CLASS_NAMES, NUM_CLASSES


def load_wav_16k_mono(file_path: pathlib.Path) -> np.ndarray:
    """
    Reads a WAV file, converts to 16 kHz mono float32 in [-1.0, 1.0].
    """
    with wave.open(str(file_path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        nframes = w.getnframes()
        raw_bytes = w.readframes(nframes)

    # Read samples
    if sw == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Downmix to mono if multi-channel
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)

    return samples


def apply_spec_augment(
    spec: torch.Tensor,
    max_time_mask: int = 5,
    max_freq_mask: int = 4,
) -> torch.Tensor:
    """
    Applies SpecAugment (time and frequency masking) to a Log-Mel spectrogram tensor.
    Tensor shape: (1, Frames=49, Mel=40) or (Frames=49, Mel=40).
    """
    augmented = spec.clone()
    is_3d = (augmented.dim() == 3)
    t_dim = 1 if is_3d else 0
    f_dim = 2 if is_3d else 1

    num_frames = augmented.shape[t_dim]
    num_mel = augmented.shape[f_dim]

    # Time mask
    t_len = random.randint(0, max_time_mask)
    if t_len > 0 and num_frames > t_len:
        t_start = random.randint(0, num_frames - t_len)
        if is_3d:
            augmented[:, t_start : t_start + t_len, :] = 0.0
        else:
            augmented[t_start : t_start + t_len, :] = 0.0

    # Frequency mask
    f_len = random.randint(0, max_freq_mask)
    if f_len > 0 and num_mel > f_len:
        f_start = random.randint(0, num_mel - f_len)
        if is_3d:
            augmented[:, :, f_start : f_start + f_len] = 0.0
        else:
            augmented[:, f_start : f_start + f_len] = 0.0

    return augmented


class SpeechCommandsDataset(Dataset):
    """
    Dataset representing Google Speech Commands v0.02 keyword spotting records.

    Preprocessing:
        1. Audio is padded with zeros or truncated to exactly 16,000 samples (1.000s).
        2. Scaled to normalized float32 in [-1.0, 1.0].
        3. Optional training-only augmentation:
           - Random time shift: uniformly shifts audio by up to +/- 1,600 samples (+/- 100 ms).
           - Random amplitude scaling: scales volume by factor in [0.8, 1.2].
           - SpecAugment: random time and frequency masking on Log-Mel features.
        4. Log-Mel Spectrogram extraction:
           - Frame: 640 samples (40 ms), Hop: 320 samples (20 ms), Mel bins: 40.
           - Yields tensor of shape (1, 49, 40).
    """

    # In-memory cache for long background noise recordings to prevent redundant disk I/O
    _noise_audio_cache: Dict[str, np.ndarray] = {}

    def __init__(
        self,
        manifest_path: str,
        dataset_root: str = "data/raw",
        target_length: int = 16000,
        is_training: bool = False,
        augment: bool = False,
        time_shift_max_samples: int = 1600,
        cache_features: bool = False,
        cached_tensor_file: Optional[str] = None,
    ):
        self.manifest_path = pathlib.Path(manifest_path).resolve()
        self.dataset_root = pathlib.Path(dataset_root).resolve()
        self.target_length = target_length
        self.is_training = is_training
        self.augment = augment and is_training
        self.time_shift_max_samples = time_shift_max_samples
        self.cache_features = cache_features

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        # Frozen Stage 4 feature extractor
        self.feature_extractor = LogMelSpectrogramExtractor(
            sample_rate=16000,
            frame_size=640,
            hop_size=320,
            num_mel_bins=40,
        )

        # Optional pre-loaded tensor cache
        self.cached_features: Optional[torch.Tensor] = None
        self.cached_labels: Optional[torch.Tensor] = None

        if cached_tensor_file and pathlib.Path(cached_tensor_file).is_file():
            cached_data = torch.load(cached_tensor_file, map_location="cpu")
            cache_ver = cached_data.get("cache_version", 1)
            cache_classes = cached_data.get("class_names")
            cache_num_classes = cached_data.get("num_classes")
            self.cached_features = cached_data.get("features")
            self.cached_labels = cached_data.get("labels")

            # Validate cache version and class alignment
            is_valid = True
            reason = ""
            if self.cached_features is None or self.cached_labels is None:
                is_valid = False
                reason = "missing 'features' or 'labels' tensor"
            elif cache_ver < 2:
                is_valid = False
                reason = f"stale cache version ({cache_ver} < 2)"
            elif cache_classes is not None and cache_classes != CLASS_NAMES:
                is_valid = False
                reason = f"class names mismatch (cached: {len(cache_classes)} classes vs expected: {len(CLASS_NAMES)})"
            elif cache_num_classes is not None and cache_num_classes != NUM_CLASSES:
                is_valid = False
                reason = f"num_classes mismatch (cached: {cache_num_classes} vs expected: {NUM_CLASSES})"
            elif len(self.cached_features) != len(self.records):
                is_valid = False
                reason = f"sample count mismatch (cached: {len(self.cached_features)} vs manifest: {len(self.records)})"
            elif len(self.cached_labels) > 0 and (int(torch.max(self.cached_labels).item()) >= NUM_CLASSES or int(torch.min(self.cached_labels).item()) < 0):
                is_valid = False
                reason = f"cached labels outside valid class range [0, {NUM_CLASSES - 1}]"

            if not is_valid:
                print(
                    f"[WARNING] Cached tensor file '{cached_tensor_file}' rejected ({reason}). "
                    f"Ignoring stale cache and reading from raw audio."
                )
                self.cached_features = None
                self.cached_labels = None

        # Per-item lazy in-memory cache
        self._cache: Dict[int, Tuple[torch.Tensor, int]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_audio_sample(self, record: Dict[str, Any]) -> np.ndarray:
        """Loads and normalizes raw audio sample for a record."""
        # 1. Handle synthetic ambient silence matching real room acoustics (RMS ~ 0.00089)
        if record.get("is_ambient_synth", False):
            # Realistic low-energy room silence with faint Gaussian noise
            synth_noise = np.random.normal(0.0, 0.00089, self.target_length).astype(np.float32)
            return np.clip(synth_noise, -1.0, 1.0)

        file_path = self.dataset_root / record["path"]

        # 2. Handle sliced background noise recordings
        if record.get("is_silence_slice", False):
            cache_key = str(file_path)
            if cache_key not in SpeechCommandsDataset._noise_audio_cache:
                SpeechCommandsDataset._noise_audio_cache[cache_key] = load_wav_16k_mono(file_path)
            noise_full = SpeechCommandsDataset._noise_audio_cache[cache_key]

            # Read exact slice if specified in manifest
            if "start_sample" in record and "end_sample" in record:
                s_start = int(record["start_sample"])
                s_end = int(record["end_sample"])
                if s_end <= len(noise_full):
                    audio = noise_full[s_start:s_end].copy()
                else:
                    audio = noise_full[s_start:].copy()
            else:
                slice_idx = record.get("slice_idx", 0)
                offset = (slice_idx * self.target_length) % max(1, len(noise_full) - self.target_length)
                audio = noise_full[offset : offset + self.target_length].copy()

            if len(audio) < self.target_length:
                pad = np.zeros(self.target_length - len(audio), dtype=np.float32)
                audio = np.concatenate([audio, pad])
            elif len(audio) > self.target_length:
                audio = audio[: self.target_length]

            # Scale background noise to realistic ambient level (0.25x)
            audio = audio * 0.25
        else:
            # 3. Regular audio file loading
            audio = load_wav_16k_mono(file_path)
            if len(audio) < self.target_length:
                pad_len = self.target_length - len(audio)
                audio = np.pad(audio, (0, pad_len), mode="constant")
            elif len(audio) > self.target_length:
                audio = audio[: self.target_length]

        return audio

    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Applies training-only time shift and volume scaling."""
        if not self.augment:
            return audio

        # Random time shift (+/- 100 ms)
        shift = random.randint(-self.time_shift_max_samples, self.time_shift_max_samples)
        if shift > 0:
            audio = np.pad(audio, (shift, 0), mode="constant")[: self.target_length]
        elif shift < 0:
            audio = np.pad(audio, (0, -shift), mode="constant")[-shift:]
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode="constant")

        # Random volume perturbation (+/- 20%)
        volume_factor = random.uniform(0.8, 1.2)
        audio = np.clip(audio * volume_factor, -1.0, 1.0)

        return audio

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Fast path 1: Loaded from pre-extracted tensor file
        if self.cached_features is not None:
            feat = self.cached_features[idx]
            label = int(self.cached_labels[idx].item())
            if self.augment:
                feat = apply_spec_augment(feat)
            return feat, label

        # Fast path 2: In-memory lazy cache
        if self.cache_features and idx in self._cache and not self.augment:
            return self._cache[idx]

        record = self.records[idx]
        audio = self._load_audio_sample(record)

        if self.augment:
            audio = self._apply_augmentation(audio)

        # Extract features (shape: 49 frames x 40 Mel bins)
        spec = self.feature_extractor.extract(audio)

        # Convert to PyTorch Tensor: (Channels=1, Frames=49, Mel=40)
        feature_tensor = torch.from_numpy(spec).unsqueeze(0).float()
        label_idx = int(record["label_idx"])

        if self.augment:
            feature_tensor = apply_spec_augment(feature_tensor)

        if self.cache_features and not self.augment:
            self._cache[idx] = (feature_tensor, label_idx)

        return feature_tensor, label_idx


def precompute_dataset_features(
    manifest_path: str,
    output_file: str,
    dataset_root: str = "data/raw",
    num_workers: int = 8,
) -> str:
    """
    Precomputes Log-Mel features for an entire manifest using parallel threads
    and saves as a single PyTorch tensor file.
    Enables orders of magnitude faster CPU training without disk bottlenecks.

    Parameters:
        manifest_path: Path to manifest JSON.
        output_file: Target .pt file path.
        dataset_root: Path to raw dataset root.
        num_workers: Number of parallel extraction threads.

    Returns:
        Path to generated .pt cache file.
    """
    import concurrent.futures

    out_path = pathlib.Path(output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Precomputing features for: {manifest_path}")
    print(f"Output target:              {out_path}")

    dataset = SpeechCommandsDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        is_training=False,
        augment=False,
        cache_features=False,
    )
    total = len(dataset)
    all_features = torch.zeros((total, 1, 49, 40), dtype=torch.float32)
    all_labels = torch.zeros(total, dtype=torch.long)

    t0 = time.perf_counter()

    def _extract_item(idx: int) -> Tuple[int, torch.Tensor, int]:
        feat, label = dataset[idx]
        return idx, feat, label

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, feat, label in executor.map(_extract_item, range(total)):
            all_features[idx] = feat
            all_labels[idx] = label
            if (idx + 1) % 5000 == 0 or (idx + 1) == total:
                elapsed = time.perf_counter() - t0
                rate = (idx + 1) / max(0.001, elapsed)
                print(f"  Processed {idx + 1:>5d}/{total} samples ({rate:.1f} samples/s)")

    torch.save(
        {
            "cache_version": 2,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "manifest_path": str(manifest_path),
            "manifest_records": total,
            "features": all_features,
            "labels": all_labels,
        },
        out_path,
    )
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {total:,} features to {out_path} ({file_size_mb:.2f} MB) in {time.perf_counter() - t0:.2f}s\n")
    return str(out_path)
