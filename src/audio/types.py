"""
src/audio/types.py
------------------
Data structures, enumerations, and type definitions for audio acquisition
and buffering subsystem.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional
import numpy as np


class QueueOverflowPolicy(Enum):
    """Policy executed when BoundedAudioQueue reaches capacity."""
    DROP_OLDEST = auto()  # Evict oldest queued block to accommodate newest (best for real-time)
    DROP_NEWEST = auto()  # Reject incoming block (preserves historical continuity)


class BufferState(Enum):
    """Lifecycle state of an individual software ping-pong buffer slot."""
    EMPTY = 0    # Buffer contains no valid audio data and is available for filling
    FILLING = 1  # Buffer is currently being populated by audio producer
    READY = 2    # Buffer is fully populated and pending consumer read
    READING = 3  # Buffer is currently being read by downstream consumer


@dataclass
class AudioBlock:
    """
    Encapsulates a single discrete audio frame/hop with metadata and timing instrumentation.

    Attributes:
        data: Audio sample array (1D NumPy array).
        sequence_number: Monotonically increasing block counter.
        capture_timestamp: Monotonic clock timestamp (perf_counter) upon callback ingress.
        sample_rate: Sampling frequency in Hz (e.g. 16,000 Hz).
        channels: Number of audio channels (e.g. 1 = mono).
        buffer_id: ID of ping-pong buffer used (0 = Ping, 1 = Pong, -1 = synthetic/direct).
        enqueue_timestamp: Monotonic clock timestamp when block entered bounded queue.
        dequeue_timestamp: Monotonic clock timestamp when block was retrieved by consumer.
        status_flags: Optional hardware/driver status flags (e.g. input overflow).
        time_info: PortAudio timing dictionary (inputBufferAdcTime, currentTime).
    """
    data: np.ndarray
    sequence_number: int
    capture_timestamp: float
    sample_rate: int = 16000
    channels: int = 1
    buffer_id: int = -1
    enqueue_timestamp: Optional[float] = None
    dequeue_timestamp: Optional[float] = None
    status_flags: Optional[str] = None
    time_info: Optional[Dict[str, float]] = None

    @property
    def num_samples(self) -> int:
        return len(self.data)

    @property
    def duration_ms(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return (len(self.data) / self.sample_rate) * 1000.0

    @property
    def queue_latency_ms(self) -> Optional[float]:
        """Time spent sitting in the bounded queue before consumption."""
        if self.enqueue_timestamp is not None and self.dequeue_timestamp is not None:
            return (self.dequeue_timestamp - self.enqueue_timestamp) * 1000.0
        return None

    def to_float32(self) -> np.ndarray:
        """
        Converts audio data to standardized normalized 32-bit floating point [-1.0, 1.0].
        If data is already float32, returns a copy. If int16, divides by 32768.0.
        """
        if self.data.dtype == np.float32:
            return self.data.copy()
        elif self.data.dtype == np.int16:
            return self.data.astype(np.float32) / 32768.0
        elif self.data.dtype == np.int32:
            return self.data.astype(np.float32) / 2147483648.0
        else:
            arr = self.data.astype(np.float32)
            max_val = np.max(np.abs(arr)) if len(arr) > 0 else 1.0
            return arr / (max_val if max_val > 0 else 1.0)


@dataclass
class AudioDeviceInfo:
    """Representation of an audio hardware device on the host system."""
    device_id: int
    name: str
    hostapi_name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool = False

    def __str__(self) -> str:
        tag = " [DEFAULT INPUT]" if self.is_default_input else ""
        return (
            f"Device [{self.device_id}] {self.name} ({self.hostapi_name}) "
            f"- {self.max_input_channels} In, {self.max_output_channels} Out, "
            f"Default Rate: {self.default_sample_rate:.0f} Hz{tag}"
        )

