"""
src/audio/buffers.py
--------------------
Software-managed ping-pong double buffer implementation.

NOTE: These are software-managed application buffers, NOT hardware DMA buffers.
In an embedded DSP or microcontroller, ping-pong buffering is typically executed
by hardware Direct Memory Access (DMA) controllers swapping memory addresses without
CPU overhead. In this Python system, ping-pong buffering is implemented in software
using two pre-allocated NumPy buffers, thread synchronization, and atomic state
transitions to emulate the double-buffering paradigm, preventing race conditions
between audio acquisition callbacks and processing threads.
"""

import threading
import time
from typing import Dict, Optional, Tuple
import numpy as np

from src.audio.types import BufferState


class BufferOverrunError(RuntimeError):
    """Raised or recorded when a buffer is overwritten before consumption."""
    pass


class SoftwarePingPongBuffer:
    """
    Two application-managed reusable memory buffers (Ping [0] and Pong [1]).

    Alternates between Buffer 0 and Buffer 1 on successive audio chunks.
    Tracks state machine (EMPTY, FILLING, READY, READING), sequence numbers,
    and detects overwritten or delayed buffer consumption.
    """

    def __init__(self, buffer_size: int = 320, dtype: np.dtype = np.int16):
        """
        Initializes the software ping-pong buffer with pre-allocated arrays.

        Parameters:
            buffer_size: Number of samples per buffer slot (default 320 = 20 ms @ 16 kHz).
            dtype: Data type of samples (np.int16 for 16-bit PCM, np.float32 for float).
        """
        self.buffer_size = buffer_size
        self.dtype = np.dtype(dtype)

        # Pre-allocate the two internal buffers in memory
        self._buffers = [
            np.zeros(buffer_size, dtype=self.dtype),  # Slot 0: Ping
            np.zeros(buffer_size, dtype=self.dtype),  # Slot 1: Pong
        ]

        # State tracking per slot
        self._states = [BufferState.EMPTY, BufferState.EMPTY]
        self._sequence_numbers = [-1, -1]
        self._write_timestamps = [0.0, 0.0]

        # Active producer slot (0 or 1)
        self._active_write_slot = 0

        # Instrumentation counters
        self._reuse_counts = [0, 0]
        self._overrun_count = 0
        self._underrun_count = 0
        self._total_written = 0
        self._total_read = 0

        # Synchronization strategy: Reentrant Lock with Condition variable
        self._lock = threading.RLock()
        self._ready_condition = threading.Condition(self._lock)

    @property
    def overrun_count(self) -> int:
        """Total number of times a buffer was overwritten while still unread (consumer too slow)."""
        with self._lock:
            return self._overrun_count

    @property
    def underrun_count(self) -> int:
        """Total number of times the consumer requested a buffer when none was ready."""
        with self._lock:
            return self._underrun_count

    @property
    def reuse_counts(self) -> Tuple[int, int]:
        """Number of times Slot 0 (Ping) and Slot 1 (Pong) have been filled/reused."""
        with self._lock:
            return (self._reuse_counts[0], self._reuse_counts[1])

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    @property
    def total_read(self) -> int:
        with self._lock:
            return self._total_read

    def write_block(self, data: np.ndarray, sequence_number: int) -> Tuple[int, bool]:
        """
        Fills the active buffer slot with incoming audio samples and alternates slot.

        Parameters:
            data: 1D NumPy array of audio samples (length must match buffer_size).
            sequence_number: Monotonic block sequence counter.

        Returns:
            Tuple of (buffer_id: int, was_overrun: bool)
            buffer_id: 0 for Ping, 1 for Pong.
            was_overrun: True if target buffer was still unread (overwritten).
        """
        if len(data) != self.buffer_size:
            raise ValueError(f"Data length ({len(data)}) does not match buffer size ({self.buffer_size}).")

        with self._lock:
            target_slot = self._active_write_slot
            was_overrun = False

            # Check if target slot is still pending consumer read or being read
            current_state = self._states[target_slot]
            if current_state in (BufferState.READY, BufferState.READING):
                was_overrun = True
                self._overrun_count += 1

            # Transition state to FILLING
            self._states[target_slot] = BufferState.FILLING

            # Copy samples into preallocated buffer without reallocating
            np.copyto(self._buffers[target_slot], data)
            self._sequence_numbers[target_slot] = sequence_number
            self._write_timestamps[target_slot] = time.perf_counter()
            self._reuse_counts[target_slot] += 1
            self._total_written += 1

            # Transition state to READY
            self._states[target_slot] = BufferState.READY

            # Alternate active write slot: 0 -> 1 -> 0
            self._active_write_slot = 1 - target_slot

            # Wake up any blocked consumers waiting on condition
            self._ready_condition.notify_all()

            return target_slot, was_overrun

    def read_ready_buffer(
        self, timeout: Optional[float] = None
    ) -> Optional[Tuple[np.ndarray, int, int, float]]:
        """
        Retrieves the next ready buffer for consumer processing.

        Parameters:
            timeout: Maximum seconds to wait for a ready buffer.
                     If None, non-blocking poll. If > 0, waits up to timeout.

        Returns:
            Tuple of (data_copy: np.ndarray, buffer_id: int, sequence_number: int, timestamp: float)
            or None if no buffer is available.
        """
        with self._ready_condition:
            # Find an available READY buffer
            target_slot = self._find_ready_slot()

            if target_slot is None and timeout is not None and timeout > 0:
                # Wait until producer signals a ready buffer
                def predicate():
                    return self._find_ready_slot() is not None

                self._ready_condition.wait_for(predicate, timeout=timeout)
                target_slot = self._find_ready_slot()

            if target_slot is None:
                self._underrun_count += 1
                return None

            # Transition state: READY -> READING
            self._states[target_slot] = BufferState.READING

            # Retrieve data copy to ensure complete isolation from producer
            data_copy = self._buffers[target_slot].copy()
            seq_num = self._sequence_numbers[target_slot]
            ts = self._write_timestamps[target_slot]
            self._total_read += 1

            # Transition state: READING -> EMPTY
            self._states[target_slot] = BufferState.EMPTY

            return data_copy, target_slot, seq_num, ts

    def _find_ready_slot(self) -> Optional[int]:
        """Identifies which slot is READY, prioritizing the oldest sequence number."""
        ready_slots = [i for i, state in enumerate(self._states) if state == BufferState.READY]
        if not ready_slots:
            return None
        if len(ready_slots) == 1:
            return ready_slots[0]
        # Both are ready (consumer lag): pick the one with lowest (earliest) sequence number
        return min(ready_slots, key=lambda i: self._sequence_numbers[i])

    def get_status(self) -> Dict[str, any]:
        """Returns diagnostic status snapshot of both buffers."""
        with self._lock:
            return {
                "active_write_slot": self._active_write_slot,
                "slot_0_state": self._states[0].name,
                "slot_1_state": self._states[1].name,
                "slot_0_seq": self._sequence_numbers[0],
                "slot_1_seq": self._sequence_numbers[1],
                "slot_0_reuses": self._reuse_counts[0],
                "slot_1_reuses": self._reuse_counts[1],
                "overrun_count": self._overrun_count,
                "underrun_count": self._underrun_count,
                "total_written": self._total_written,
                "total_read": self._total_read,
            }

