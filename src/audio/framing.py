"""
src/audio/framing.py
--------------------
Audio frame sliding window buffer and numerical format converters.

Implements hop-to-frame sliding window generation:
- Target Hop: 320 samples (20.0 ms @ 16 kHz)
- Target Frame: 640 samples (40.0 ms @ 16 kHz, 50% overlap)
- Explicit conversion from 16-bit PCM to normalized float32 [-1.0, 1.0].
"""

from typing import Generator, List, Optional
import numpy as np


def pcm16_to_float32(pcm_data: np.ndarray) -> np.ndarray:
    """
    Explicitly converts 16-bit signed integer PCM audio [-32768, 32767]
    to standard IEEE 754 32-bit floating point [-1.0, 1.0].

    Numerical Representation:
        y[n] = x[n] / 32768.0, where x[n] in int16.
        Produces full dynamic range symmetric scaling with zero DC bias.
    """
    if pcm_data.dtype != np.int16:
        if pcm_data.dtype == np.float32:
            return pcm_data.copy()
        pcm_data = pcm_data.astype(np.int16)
    return pcm_data.astype(np.float32) / 32768.0


def float32_to_pcm16(float_data: np.ndarray) -> np.ndarray:
    """
    Converts normalized float32 [-1.0, 1.0] back to 16-bit PCM with clipping guard.
    """
    clipped = np.clip(float_data, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


class SlidingWindowFramer:
    """
    Transforms a continuous stream of incoming hop blocks (e.g. 320 samples)
    into overlapping analysis frames (e.g. 640 samples).

    Maintains internal history state to construct full window frames with 50% overlap.
    """

    def __init__(self, frame_size: int = 640, hop_size: int = 320):
        if hop_size <= 0 or frame_size <= 0:
            raise ValueError("frame_size and hop_size must be positive.")
        if hop_size > frame_size:
            raise ValueError("hop_size cannot exceed frame_size.")

        self.frame_size = frame_size
        self.hop_size = hop_size
        self.overlap_size = frame_size - hop_size

        # Internal buffer to store overlap history
        self._history = np.zeros(self.overlap_size, dtype=np.int16)
        self._initialized = False

    def reset(self):
        """Resets the history buffer."""
        self._history = np.zeros(self.overlap_size, dtype=np.int16)
        self._initialized = False

    def process_hop(self, hop_data: np.ndarray) -> np.ndarray:
        """
        Accepts a hop block (length = hop_size) and returns a complete frame (length = frame_size).

        Parameters:
            hop_data: 1D NumPy array of length hop_size.

        Returns:
            1D NumPy array of length frame_size formed by [history + new_hop].
        """
        if len(hop_data) != self.hop_size:
            raise ValueError(f"Expected hop size {self.hop_size}, got {len(hop_data)}.")

        # Construct frame: history (320 samples) + current hop (320 samples) = 640 samples
        frame = np.concatenate([self._history, hop_data])

        # Update history for next iteration with the trailing overlap_size samples
        self._history = frame[-self.overlap_size:].copy()
        self._initialized = True

        return frame

