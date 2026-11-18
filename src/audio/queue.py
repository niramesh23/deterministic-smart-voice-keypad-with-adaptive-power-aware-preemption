"""
src/audio/queue.py
------------------
Bounded, thread-safe producer-consumer queue for audio blocks.
Implements explicit overflow policies, non-blocking producer semantics,
and detailed timing/latency accounting.
"""

from collections import deque
import threading
import time
from typing import Any, Dict, List, Optional

from src.audio.types import AudioBlock, QueueOverflowPolicy


class BoundedAudioQueue:
    """
    Thread-safe bounded queue between audio acquisition producer and DSP/RTS consumer.

    Features:
    - Strict capacity enforcement.
    - Explicit overflow policy (DROP_OLDEST or DROP_NEWEST).
    - Producer never blocks indefinitely (O(1) non-blocking push).
    - Detailed accounting of drops, wait times, and queue occupancy.
    - Clean shutdown signaling.
    """

    def __init__(
        self,
        capacity: int = 100,
        overflow_policy: QueueOverflowPolicy = QueueOverflowPolicy.DROP_OLDEST,
    ):
        """
        Initializes the bounded audio queue.

        Parameters:
            capacity: Maximum number of AudioBlocks allowed in queue.
            overflow_policy: QueueOverflowPolicy.DROP_OLDEST or DROP_NEWEST.
        """
        if capacity <= 0:
            raise ValueError(f"Queue capacity must be positive, got {capacity}.")

        self.capacity = capacity
        self.overflow_policy = overflow_policy

        self._queue: deque[AudioBlock] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False

        # Instrumentation metrics (bounded deques for O(1) eviction without list memory copies)
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._overflow_count = 0
        self._dropped_blocks_count = 0
        self._peak_occupancy = 0
        self._dropped_log: List[Dict[str, Any]] = []
        self._wait_times_ms: deque[float] = deque(maxlen=2000)

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def push(self, block: AudioBlock) -> bool:
        """
        Pushes an AudioBlock into the queue (Non-blocking).
        Never blocks the producer indefinitely.

        Returns:
            True if block was enqueued (or enqueued after dropping oldest),
            False if block itself was dropped (under DROP_NEWEST) or queue is closed.
        """
        with self._lock:
            if self._closed:
                return False

            now = time.perf_counter()
            current_len = len(self._queue)

            if current_len >= self.capacity:
                self._overflow_count += 1
                self._dropped_blocks_count += 1

                if self.overflow_policy == QueueOverflowPolicy.DROP_OLDEST:
                    # Evict the oldest block sitting in front of queue
                    evicted = self._queue.popleft()
                    self._dropped_log.append({
                        "sequence_number": evicted.sequence_number,
                        "capture_timestamp": evicted.capture_timestamp,
                        "drop_timestamp": now,
                        "reason": "OVERFLOW_DROP_OLDEST",
                    })
                    # Enqueue the new block
                    block.enqueue_timestamp = now
                    self._queue.append(block)
                    self._total_enqueued += 1
                    self._not_empty.notify()
                    return True
                else:  # DROP_NEWEST
                    self._dropped_log.append({
                        "sequence_number": block.sequence_number,
                        "capture_timestamp": block.capture_timestamp,
                        "drop_timestamp": now,
                        "reason": "OVERFLOW_DROP_NEWEST",
                    })
                    return False

            # Queue has available capacity
            block.enqueue_timestamp = now
            self._queue.append(block)
            self._total_enqueued += 1

            if len(self._queue) > self._peak_occupancy:
                self._peak_occupancy = len(self._queue)

            self._not_empty.notify()
            return True

    def pop(self, timeout: Optional[float] = None) -> Optional[AudioBlock]:
        """
        Retrieves the next AudioBlock from the queue.

        Parameters:
            timeout: Maximum seconds to wait if queue is empty.
                     If None, blocks until an item is available or queue closed.
                     If 0, non-blocking check.

        Returns:
            AudioBlock if available, or None if timeout expired or queue closed.
        """
        with self._not_empty:
            while len(self._queue) == 0:
                if self._closed:
                    return None
                if timeout is not None and timeout <= 0:
                    return None

                success = self._not_empty.wait(timeout=timeout)
                if not success and len(self._queue) == 0:
                    return None

            block = self._queue.popleft()
            now = time.perf_counter()
            block.dequeue_timestamp = now

            if block.enqueue_timestamp is not None:
                wait_ms = (now - block.enqueue_timestamp) * 1000.0
                self._wait_times_ms.append(wait_ms)

            self._total_dequeued += 1
            return block

    def close(self):
        """Signals shutdown to all waiting threads and closes queue."""
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()

    def clear(self):
        """Clears all buffered audio blocks."""
        with self._lock:
            self._queue.clear()

    def qsize(self) -> int:
        """Returns current number of buffered audio blocks."""
        with self._lock:
            return len(self._queue)

    def __len__(self) -> int:
        return self.qsize()

    def get_metrics(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic and timing metrics for the queue."""
        with self._lock:
            current_occ = len(self._queue)
            wait_times = list(self._wait_times_ms)

            mean_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
            p95_wait = float(sorted(wait_times)[int(len(wait_times) * 0.95)]) if wait_times else 0.0

            return {
                "capacity": self.capacity,
                "current_occupancy": current_occ,
                "peak_occupancy": self._peak_occupancy,
                "total_enqueued": self._total_enqueued,
                "total_dequeued": self._total_dequeued,
                "overflow_count": self._overflow_count,
                "dropped_blocks_count": self._dropped_blocks_count,
                "overflow_policy": self.overflow_policy.name,
                "mean_queue_wait_ms": mean_wait,
                "p95_queue_wait_ms": p95_wait,
                "recent_drops_sample": list(self._dropped_log[-5:]),
            }

