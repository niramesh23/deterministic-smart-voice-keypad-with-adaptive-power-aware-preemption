"""
src/dsp/noise.py
----------------
Adaptive acoustic noise floor estimator and dynamic threshold tracking.

Guarantees:
- Updates conservatively during verified non-speech periods.
- Freezes or heavily attenuates updates during active speech to avoid absorbing word energy.
- Enforces strict upper/lower bounds and maximum per-frame delta limits.
- Implements asymmetric exponential moving average (fast descent, slow ascent).
"""

from typing import Dict, List, Optional
import numpy as np


class AdaptiveNoiseFloor:
    """
    Estimates background acoustic noise floor and sets adaptive VAD thresholds.

    Formula:
        noise_floor[k] = alpha * noise_floor[k-1] + (1 - alpha) * frame_rms
        threshold[k]   = noise_floor[k] * threshold_factor
    """

    def __init__(
        self,
        initial_noise_floor: float = 0.005,
        threshold_factor: float = 2.5,
        up_alpha: float = 0.98,
        down_alpha: float = 0.90,
        min_noise_floor: float = 0.0005,
        max_noise_floor: float = 0.08,
        max_delta_per_hop: float = 0.002,
    ):
        """
        Parameters:
            initial_noise_floor: Starting estimate of RMS background noise floor.
            threshold_factor: Multiplier above noise floor defining VAD trigger threshold.
            up_alpha: EMA weight when noise floor is rising (slow rise: 0.98).
            down_alpha: EMA weight when noise floor is falling (faster drop: 0.90).
            min_noise_floor: Absolute floor clamp (prevents zero division / hyper-sensitivity).
            max_noise_floor: Absolute ceiling clamp (prevents runaway noise tracking).
            max_delta_per_hop: Maximum permitted single-frame change in noise floor.
        """
        self.noise_floor = float(np.clip(initial_noise_floor, min_noise_floor, max_noise_floor))
        self.threshold_factor = threshold_factor
        self.up_alpha = up_alpha
        self.down_alpha = down_alpha
        self.min_noise_floor = min_noise_floor
        self.max_noise_floor = max_noise_floor
        self.max_delta_per_hop = max_delta_per_hop

        self._update_count = 0
        self._freeze_count = 0
        self._history: List[float] = [self.noise_floor]

    @property
    def threshold(self) -> float:
        """Current Voice Activity Detection (VAD) activation threshold."""
        return float(self.noise_floor * self.threshold_factor)

    def set_floor(self, new_floor: float):
        """Calibrates/sets background noise floor to an empirical acoustic baseline."""
        self.noise_floor = float(np.clip(new_floor, self.min_noise_floor, self.max_noise_floor))
        self._freeze_count = 0

    def update(self, frame_rms: float, is_speech_active: bool = False) -> float:
        """
        Updates the background noise floor estimate using the incoming frame RMS.

        Parameters:
            frame_rms: Current frame RMS energy.
            is_speech_active: True if VAD is currently in an active speech or hangover state.

        Returns:
            float: Updated noise floor estimate.
        """
        # Rule 1: Never adapt on active speech or strong transients above current threshold
        if is_speech_active or frame_rms > self.threshold:
            self._freeze_count += 1
            # If frozen for > 60 consecutive hops (~1.2s), slow upward tracking prevents deadlock on stationary room noise
            if self._freeze_count > 60 and frame_rms <= self.max_noise_floor:
                target_floor = (0.995 * self.noise_floor) + (0.005 * frame_rms)
                self.noise_floor = float(np.clip(target_floor, self.min_noise_floor, self.max_noise_floor))
            return self.noise_floor

        self._freeze_count = 0

        # Rule 2: Asymmetric exponential moving average
        if frame_rms > self.noise_floor:
            # Rising noise floor: update conservatively with high up_alpha
            alpha = self.up_alpha
        else:
            # Falling noise floor: adapt quicker to quietness with down_alpha
            alpha = self.down_alpha

        target_floor = (alpha * self.noise_floor) + ((1.0 - alpha) * frame_rms)

        # Rule 3: Bound rate of change per hop
        delta = target_floor - self.noise_floor
        bounded_delta = float(np.clip(delta, -self.max_delta_per_hop, self.max_delta_per_hop))
        updated_floor = self.noise_floor + bounded_delta

        # Rule 4: Enforce absolute minimum and maximum bounds
        self.noise_floor = float(np.clip(updated_floor, self.min_noise_floor, self.max_noise_floor))
        self._update_count += 1

        if len(self._history) < 2000:
            self._history.append(self.noise_floor)

        return self.noise_floor

    def reset(self, initial_noise_floor: Optional[float] = None):
        """Resets the noise floor estimator state."""
        if initial_noise_floor is not None:
            self.noise_floor = float(np.clip(initial_noise_floor, self.min_noise_floor, self.max_noise_floor))
        self._update_count = 0
        self._freeze_count = 0
        self._history = [self.noise_floor]

    def get_diagnostics(self) -> Dict[str, any]:
        return {
            "noise_floor": self.noise_floor,
            "threshold": self.threshold,
            "threshold_factor": self.threshold_factor,
            "total_updates": self._update_count,
            "frozen_updates": self._freeze_count,
            "min_noise_floor": self.min_noise_floor,
            "max_noise_floor": self.max_noise_floor,
        }

