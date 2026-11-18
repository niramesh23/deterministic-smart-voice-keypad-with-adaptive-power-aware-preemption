"""
src/dsp/preroll.py
------------------
Circular pre-roll buffer to preserve acoustic word onsets prior to VAD activation.

Standard Specification:
- Duration: 200 ms (default)
- At 16,000 Hz: 3,200 samples (10 hops @ 320 samples / hop).
"""

from collections import deque
from typing import Optional
import numpy as np


class PreRollBuffer:
    """
    Fixed-capacity circular FIFO buffer holding pre-trigger audio history.

    When Voice Activity Detection triggers on speech energy, the initial
    consonants (e.g. unvoiced plosives 'p', 't', 'k', or fricatives 's', 'f')
    precede the high-energy vowel core. Pre-roll guarantees these onset
    phonemes are captured.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration_ms: float = 200.0,
        hop_size: int = 320,
        dtype: np.dtype = np.int16,
    ):
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.hop_size = hop_size
        self.dtype = np.dtype(dtype)

        self.target_samples = int(round((duration_ms / 1000.0) * sample_rate))
        self.num_hops = int(round(self.target_samples / hop_size))

        self._hops_queue: deque[np.ndarray] = deque(maxlen=self.num_hops)
        self._total_hops_pushed = 0

    @property
    def capacity_samples(self) -> int:
        return self.num_hops * self.hop_size

    @property
    def current_samples(self) -> int:
        return len(self._hops_queue) * self.hop_size

    def push_hop(self, hop_data: np.ndarray):
        """
        Pushes a new audio hop block into the circular pre-roll buffer.
        """
        if len(hop_data) != self.hop_size:
            raise ValueError(f"Expected hop size {self.hop_size}, got {len(hop_data)}.")

        self._hops_queue.append(np.asarray(hop_data, dtype=self.dtype).copy())
        self._total_hops_pushed += 1

    def get_preroll_samples(self) -> np.ndarray:
        """
        Returns all buffered pre-roll audio samples in chronological order.

        If fewer hops have been pushed than capacity, returns available samples
        preceded by zero padding to guarantee exact target length.
        """
        if not self._hops_queue:
            return np.zeros(self.capacity_samples, dtype=self.dtype)

        available_hops = list(self._hops_queue)
        concatenated = np.concatenate(available_hops)

        if len(concatenated) < self.capacity_samples:
            padding_needed = self.capacity_samples - len(concatenated)
            pad = np.zeros(padding_needed, dtype=self.dtype)
            return np.concatenate([pad, concatenated])

        return concatenated

    def reset(self):
        """Clears the pre-roll buffer."""
        self._hops_queue.clear()
        self._total_hops_pushed = 0

