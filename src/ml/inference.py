"""
src/ml/inference.py
-------------------
Keyword spotting inference service with model loading, verification,
per-class confidence scoring, prediction smoothing, margin checking,
and command cooldown management.

Guarantees:
- Model loaded once at initialization.
- Validates checkpoint, class mapping, and feature configs before starting.
- Inference-only mode with disabled autograd (torch.no_grad).
- Strict separation from audio acquisition callback.
- Robust rejection of silence, unknown words, and low-confidence/ambiguous predictions.
"""

from dataclasses import dataclass, field
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

# Ensure project root in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.features import LogMelSpectrogramExtractor
from src.ml.classes import CLASS_NAMES, NUM_CLASSES, NUMBER_COMMANDS
from src.ml.model import get_model


class InferenceInitializationError(RuntimeError):
    """Raised when checkpoint or configuration files are missing or incompatible."""
    pass


@dataclass
class RecognitionResult:
    """
    Structured outcome of a live keyword recognition evaluation.

    Attributes:
        command: Confirmed keyword command name, or None if rejected/silence/unknown.
        raw_class: Highest probability class from the model.
        confidence: Probability of the top-1 class (after smoothing).
        raw_confidence: Unsmoothed softmax probability of the top-1 class.
        top2_class: Second-highest probability class.
        top2_confidence: Probability of the second-highest class.
        margin: Difference between top-1 and top-2 probabilities.
        is_confirmed: True if all criteria (threshold, margin, cooldown, VAD) are satisfied.
        is_silence: True if classified as _silence_ or rejected as acoustic silence.
        is_unknown: True if classified as _unknown_ word.
        is_cooldown: True if a valid command was detected but suppressed by the cooldown timer.
        probabilities: Dictionary mapping all 12 class names to their smoothed probabilities.
        audio_event_timestamp: Timestamp when the audio segment completed.
        decision_timestamp: Timestamp when inference completed.
        timings: Execution duration breakdown (ms) for feature extraction, inference, and postprocessing.
    """
    command: Optional[str]
    raw_class: str
    confidence: float
    raw_confidence: float
    top2_class: str
    top2_confidence: float
    margin: float
    is_confirmed: bool
    is_silence: bool
    is_unknown: bool
    is_cooldown: bool
    probabilities: Dict[str, float]
    audio_event_timestamp: float
    decision_timestamp: float
    timings: Dict[str, float] = field(default_factory=dict)


class KeywordSpotterInferenceService:
    """
    Live Keyword Spotting Inference Engine.

    Parameters:
        checkpoint_path: Path to trained PyTorch checkpoint (.pt).
        class_mapping_path: Path to class mapping JSON file.
        feature_config_path: Path to feature configuration JSON file.
        confidence_threshold: Minimum probability required to confirm a command (default: 0.65).
        margin_threshold: Minimum margin between top-1 and top-2 class (default: 0.15).
        smoothing_alpha: Exponential smoothing factor in [0.0, 1.0] (default: 0.70; 1.0 = no smoothing).
        cooldown_seconds: Minimum cooldown period between consecutive confirmed commands (default: 0.8s).
        device: PyTorch device ('cpu').
    """

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/dscnn_numbers_v1.pt",
        class_mapping_path: Optional[str] = None,
        feature_config_path: Optional[str] = None,
        confidence_threshold: float = 0.65,
        margin_threshold: float = 0.15,
        smoothing_alpha: float = 0.70,
        cooldown_seconds: float = 0.80,
        device: str = "cpu",
    ):
        self.checkpoint_path = pathlib.Path(checkpoint_path).resolve()

        # Derive tag for companion config resolution
        stem = self.checkpoint_path.stem
        tag = stem[len("dscnn_"):] if stem.startswith("dscnn_") else (stem[len("resdscnn_"):] if stem.startswith("resdscnn_") else stem)

        if class_mapping_path is None:
            candidate = self.checkpoint_path.parent / f"class_mapping_{tag}.json"
            if candidate.is_file():
                self.class_mapping_path = candidate
            else:
                self.class_mapping_path = (self.checkpoint_path.parent / "class_mapping_improved.json").resolve()
        else:
            self.class_mapping_path = pathlib.Path(class_mapping_path).resolve()

        if feature_config_path is None:
            candidate = self.checkpoint_path.parent / f"feature_config_{tag}.json"
            if candidate.is_file():
                self.feature_config_path = candidate
            else:
                self.feature_config_path = (self.checkpoint_path.parent / "feature_config_improved.json").resolve()
        else:
            self.feature_config_path = pathlib.Path(feature_config_path).resolve()

        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.smoothing_alpha = smoothing_alpha
        self.cooldown_seconds = cooldown_seconds
        self.device = torch.device(device)

        # Optimize PyTorch CPU threading: 1 thread avoids OpenMP context-switch contention
        if self.device.type == "cpu":
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        # 1. Verify existence of required files
        self._verify_file_existence()

        # 2. Verify and load feature configuration
        self.feature_config = self._load_and_verify_feature_config()

        # 3. Load class mapping
        self.class_mapping = self._load_and_verify_class_mapping()
        self.class_names = list(self.class_mapping["class_names"])
        self.num_classes = len(self.class_names)

        # 4. Initialize Feature Extractor matching Stage 4 & 5
        self.extractor = LogMelSpectrogramExtractor(
            sample_rate=self.feature_config["sample_rate"],
            frame_size=self.feature_config["frame_size"],
            hop_size=self.feature_config["hop_size"],
            num_mel_bins=self.feature_config["num_mel_bins"],
        )

        # 5. Load Model Checkpoint
        self.model = self._load_model_checkpoint()

        # 6. State management: smoothing and cooldown
        self._smoothed_probs = np.zeros(self.num_classes, dtype=np.float32)
        self._last_confirmed_command: Optional[str] = None
        self._last_confirmed_timestamp: float = 0.0
        self._total_inferences: int = 0
        self._total_confirmed: int = 0
        self._is_available: bool = True

    @property
    def is_available(self) -> bool:
        """Returns True if the ML model is currently healthy and online."""
        return self._is_available

    def set_available(self, available: bool):
        """Marks the ML model available or unavailable during fault/recovery."""
        self._is_available = available

    def _verify_file_existence(self):
        """Verifies that checkpoint and config files exist on disk."""
        if not self.checkpoint_path.is_file():
            raise InferenceInitializationError(
                f"Checkpoint file not found: {self.checkpoint_path}\n"
                f"Please ensure Stage 5 model has been trained and saved."
            )
        if not self.class_mapping_path.is_file():
            raise InferenceInitializationError(
                f"Class mapping file not found: {self.class_mapping_path}"
            )
        if not self.feature_config_path.is_file():
            raise InferenceInitializationError(
                f"Feature config file not found: {self.feature_config_path}"
            )

    def _load_and_verify_feature_config(self) -> Dict[str, Any]:
        """Loads and verifies DSP feature parameters."""
        try:
            with open(self.feature_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            raise InferenceInitializationError(f"Failed to parse feature config JSON: {e}") from e

        required_keys = ["sample_rate", "frame_size", "hop_size", "num_mel_bins", "target_length"]
        for k in required_keys:
            if k not in cfg:
                raise InferenceInitializationError(f"Missing required feature config key: '{k}'")

        if cfg["sample_rate"] != 16000:
            raise InferenceInitializationError(f"Unsupported sample rate: {cfg['sample_rate']}. Expected 16,000 Hz.")
        if cfg["frame_size"] != 640 or cfg["hop_size"] != 320 or cfg["num_mel_bins"] != 40:
            raise InferenceInitializationError(f"Incompatible framing/mel configuration: {cfg}")

        return cfg

    def _load_and_verify_class_mapping(self) -> Dict[str, Any]:
        """Loads and verifies vocabulary class mapping."""
        try:
            with open(self.class_mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        except Exception as e:
            raise InferenceInitializationError(f"Failed to parse class mapping JSON: {e}") from e

        if "class_names" not in mapping or not isinstance(mapping["class_names"], list):
            raise InferenceInitializationError(
                f"Invalid class mapping '{self.class_mapping_path}': missing 'class_names' list."
            )
        classes = mapping["class_names"]
        if len(classes) != len(set(classes)):
            raise InferenceInitializationError(
                f"Invalid class mapping '{self.class_mapping_path}': contains duplicate class names."
            )
        if len(classes) < 2:
            raise InferenceInitializationError(
                f"Invalid class mapping '{self.class_mapping_path}': expected at least 2 classes, got {len(classes)}."
            )
        return mapping

    def _load_model_checkpoint(self) -> nn.Module:
        """Loads PyTorch model checkpoint into CPU memory in evaluation mode."""
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        except Exception as e:
            raise InferenceInitializationError(f"Failed to load checkpoint torch weights: {e}") from e

        arch = ckpt.get("model_architecture", "ResidualDS-CNN")
        ckpt_classes = ckpt.get("class_names")
        ckpt_num_classes = ckpt.get("num_classes")

        # Explicit validation of class layout compatibility
        if ckpt_classes is not None and ckpt_classes != self.class_names:
            raise InferenceInitializationError(
                f"Checkpoint class names mismatch between '{self.checkpoint_path.name}' and '{self.class_mapping_path.name}'!\n"
                f"Checkpoint classes ({len(ckpt_classes)}): {ckpt_classes}\n"
                f"Expected class mapping ({len(self.class_names)}): {self.class_names}"
            )
        if ckpt_num_classes is not None and ckpt_num_classes != len(self.class_names):
            raise InferenceInitializationError(
                f"Checkpoint class count mismatch: checkpoint has {ckpt_num_classes} classes, "
                f"but class mapping specifies {len(self.class_names)} classes."
            )

        model = get_model(arch, num_classes=self.num_classes)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except Exception as e:
            raise InferenceInitializationError(
                f"Failed to load model weights (incompatible model architecture or class count mismatch): {e}"
            ) from e

        model.to(self.device)
        model.eval()

        # Disable gradients globally for model parameters
        for p in model.parameters():
            p.requires_grad = False

        # Run dummy forward passes to warm up runtime and verify shapes
        dummy = torch.randn(1, 1, 49, 40, device=self.device)
        with torch.inference_mode():
            for _ in range(5):
                out = model(dummy)
            if out.shape != (1, self.num_classes) or not torch.all(torch.isfinite(out)):
                raise InferenceInitializationError(
                    f"Model sanity check failed! Output shape: {out.shape}, expected: (1, {self.num_classes})"
                )

        return model

    def reset_state(self):
        """Resets smoothed probability state and cooldown timers."""
        self._smoothed_probs = np.zeros(self.num_classes, dtype=np.float32)
        self._last_confirmed_command = None
        self._last_confirmed_timestamp = 0.0

    def extract_features(self, audio_samples: np.ndarray) -> np.ndarray:
        """
        Extracts Log-Mel filterbank spectrogram from raw audio samples.
        Normalizes/pads audio to 16,000 samples and returns (49, 40) float32 array.
        """
        target_len = self.feature_config["target_length"]
        if len(audio_samples) < target_len:
            pad = np.zeros(target_len - len(audio_samples), dtype=audio_samples.dtype)
            audio_norm = np.concatenate([audio_samples, pad])
        elif len(audio_samples) > target_len:
            audio_norm = audio_samples[:target_len]
        else:
            audio_norm = audio_samples

        log_mel = self.extractor.extract(audio_norm)
        if log_mel.shape != (49, 40):
            raise ValueError(f"Feature shape mismatch: expected (49, 40), got {log_mel.shape}")
        return log_mel

    def predict_from_features(
        self,
        log_mel: np.ndarray,
        audio_event_timestamp: Optional[float] = None,
        dsp_time_ms: float = 0.0,
        reset_smoothing: bool = False,
    ) -> RecognitionResult:
        """
        Executes model inference and prediction decision logic over pre-extracted features.
        """
        t_start = time.perf_counter()
        if audio_event_timestamp is None:
            audio_event_timestamp = t_start

        if not self._is_available:
            return RecognitionResult(
                command=None,
                raw_class="_silence_",
                confidence=0.0,
                raw_confidence=0.0,
                top2_class="_unknown_",
                top2_confidence=0.0,
                margin=0.0,
                is_confirmed=False,
                is_silence=True,
                is_unknown=False,
                is_cooldown=False,
                probabilities={name: 0.0 for name in self.class_names},
                audio_event_timestamp=audio_event_timestamp,
                decision_timestamp=t_start,
                nn_time_ms=0.0,
                dsp_time_ms=dsp_time_ms,
                post_time_ms=0.0,
            )

        # 1. High-Performance Model Inference (inference_mode)
        t_nn_0 = time.perf_counter()
        tensor_in = torch.as_tensor(log_mel, dtype=torch.float32, device=self.device).view(1, 1, 49, 40)
        with torch.inference_mode():
            logits = self.model(tensor_in)
            probs_raw = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        t_nn_1 = time.perf_counter()
        nn_time_ms = (t_nn_1 - t_nn_0) * 1000.0

        # 2. Prediction Smoothing across successive windows
        t_post_0 = time.perf_counter()
        if reset_smoothing or np.sum(self._smoothed_probs) == 0.0 or self.smoothing_alpha >= 1.0:
            self._smoothed_probs = probs_raw.copy()
        else:
            self._smoothed_probs = (
                self.smoothing_alpha * probs_raw + (1.0 - self.smoothing_alpha) * self._smoothed_probs
            )
            sum_p = np.sum(self._smoothed_probs)
            if sum_p > 0:
                self._smoothed_probs /= sum_p

        # Rank classes by smoothed probability
        sorted_indices = np.argsort(self._smoothed_probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1])

        raw_top1_idx = int(np.argmax(probs_raw))
        raw_top1_class = self.class_names[raw_top1_idx]
        raw_conf = float(probs_raw[raw_top1_idx])
        raw_top2_conf = float(np.partition(probs_raw, -2)[-2])
        raw_margin = raw_conf - raw_top2_conf

        top1_class = self.class_names[top1_idx]
        top2_class = self.class_names[top2_idx]
        top1_conf = float(self._smoothed_probs[top1_idx])
        top2_conf = float(self._smoothed_probs[top2_idx])

        # Peak detection: if instantaneous prediction is confident or matches smoothed top-1,
        # adopt the higher raw confidence and margin so speech isn't penalized by smoothing lag
        if top1_class == raw_top1_class:
            eval_conf = max(top1_conf, raw_conf)
            eval_margin = max(top1_conf - top2_conf, raw_margin)
        elif raw_top1_class not in ("_silence_", "_unknown_") and raw_conf >= self.confidence_threshold and raw_margin >= self.margin_threshold:
            # Instantaneous confident detection of a real keyword overrides lingering silence
            top1_class = raw_top1_class
            eval_conf = raw_conf
            eval_margin = raw_margin
            self._smoothed_probs[raw_top1_idx] = max(self._smoothed_probs[raw_top1_idx], raw_conf)
        else:
            eval_conf = top1_conf
            eval_margin = top1_conf - top2_conf

        margin = eval_margin
        is_silence = (top1_class == "_silence_")
        is_unknown = (top1_class == "_unknown_")

        passed_threshold = (eval_conf >= self.confidence_threshold)
        passed_margin = (margin >= self.margin_threshold)
        is_command_candidate = (not is_silence) and (not is_unknown) and passed_threshold and passed_margin

        is_cooldown = False
        is_confirmed = False
        confirmed_command: Optional[str] = None

        if is_command_candidate:
            elapsed_since_last = audio_event_timestamp - self._last_confirmed_timestamp
            eff_cooldown = 0.40 if (top1_class in NUMBER_COMMANDS) else self.cooldown_seconds
            if top1_class == self._last_confirmed_command and elapsed_since_last < eff_cooldown:
                is_cooldown = True
            else:
                is_confirmed = True
                confirmed_command = top1_class
                self._last_confirmed_command = top1_class
                self._last_confirmed_timestamp = audio_event_timestamp
                self._total_confirmed += 1

        t_post_1 = time.perf_counter()
        post_time_ms = (t_post_1 - t_post_0) * 1000.0

        self._total_inferences += 1
        probs_dict = {name: float(self._smoothed_probs[i]) for i, name in enumerate(self.class_names)}

        return RecognitionResult(
            command=confirmed_command,
            raw_class=top1_class,
            confidence=eval_conf,
            raw_confidence=raw_conf,
            top2_class=top2_class,
            top2_confidence=top2_conf,
            margin=margin,
            is_confirmed=is_confirmed,
            is_silence=is_silence,
            is_unknown=is_unknown,
            is_cooldown=is_cooldown,
            probabilities=probs_dict,
            audio_event_timestamp=audio_event_timestamp,
            decision_timestamp=t_post_1,
            timings={
                "feature_extraction_ms": round(dsp_time_ms, 3),
                "model_inference_ms": round(nn_time_ms, 3),
                "postprocessing_ms": round(post_time_ms, 3),
            },
        )

    def predict(
        self,
        audio_samples: np.ndarray,
        audio_event_timestamp: Optional[float] = None,
        reset_smoothing: bool = False,
    ) -> RecognitionResult:
        """
        Executes feature extraction, model inference, and prediction decision logic
        over a 1.0-second (16,000 samples) audio buffer.

        Parameters:
            audio_samples: 1D NumPy array of audio samples (float32 [-1, 1] or int16).
            audio_event_timestamp: Monotonic clock timestamp when audio event finished.

        Returns:
            RecognitionResult dataclass containing decision, confidence, and timings.
        """
        t_start = time.perf_counter()
        if audio_event_timestamp is None:
            audio_event_timestamp = t_start

        # 1 & 2. Feature Extraction
        t_dsp_0 = time.perf_counter()
        log_mel = self.extract_features(audio_samples)
        t_dsp_1 = time.perf_counter()
        dsp_time_ms = (t_dsp_1 - t_dsp_0) * 1000.0

        # 3, 4, 5. Inference and Decision
        return self.predict_from_features(
            log_mel=log_mel,
            audio_event_timestamp=audio_event_timestamp,
            dsp_time_ms=dsp_time_ms,
            reset_smoothing=reset_smoothing,
        )

    def reset_state(self):
        """Resets smoothed probabilities, cooldown state, and restores availability."""
        self._smoothed_probs = np.zeros(self.num_classes, dtype=np.float32)
        self._last_confirmed_command = None
        self._last_confirmed_timestamp = 0.0
        self._is_available = True

    def get_stats(self) -> Dict[str, Any]:
        """Returns service operating statistics."""
        return {
            "total_inferences": self._total_inferences,
            "total_confirmed_commands": self._total_confirmed,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "smoothing_alpha": self.smoothing_alpha,
            "cooldown_seconds": self.cooldown_seconds,
            "model_architecture": self.model.__class__.__name__,
            "parameter_count": self.model.count_parameters(),
        }

