"""
tests/test_audio_queue.py
-------------------------
Automated synthetic tests for bounded audio queue.
Verifies FIFO ordering, bounded capacity, DROP_OLDEST and DROP_NEWEST policies,
detailed drop accounting, and clean shutdown unblocking.
"""

import threading
import time
import numpy as np
import pytest

from src.audio.queue import BoundedAudioQueue
from src.audio.types import AudioBlock, QueueOverflowPolicy


def _make_block(seq: int) -> AudioBlock:
    return AudioBlock(
        data=np.full(320, seq, dtype=np.int16),
        sequence_number=seq,
        capture_timestamp=time.perf_counter(),
    )


def test_queue_fifo_ordering():
    """Verifies that blocks are retrieved in the exact order they were enqueued."""
    q = BoundedAudioQueue(capacity=10)

    for i in range(5):
        q.push(_make_block(i))

    for i in range(5):
        blk = q.pop(timeout=0.1)
        assert blk is not None
        assert blk.sequence_number == i
        assert blk.dequeue_timestamp is not None
        assert blk.queue_latency_ms is not None
        assert blk.queue_latency_ms >= 0.0


def test_queue_overflow_drop_oldest():
    """Verifies DROP_OLDEST policy evicts the oldest items to accommodate newest."""
    q = BoundedAudioQueue(capacity=3, overflow_policy=QueueOverflowPolicy.DROP_OLDEST)

    # Push 5 items into capacity-3 queue
    for i in range(1, 6):
        ok = q.push(_make_block(i))
        assert ok is True

    metrics = q.get_metrics()
    assert metrics["capacity"] == 3
    assert metrics["current_occupancy"] == 3
    assert metrics["overflow_count"] == 2
    assert metrics["dropped_blocks_count"] == 2

    # Oldest (1, 2) were dropped; remaining must be (3, 4, 5)
    popped_seqs = []
    for _ in range(3):
        blk = q.pop(timeout=0.1)
        popped_seqs.append(blk.sequence_number)

    assert popped_seqs == [3, 4, 5]


def test_queue_overflow_drop_newest():
    """Verifies DROP_NEWEST policy rejects incoming items when queue is full."""
    q = BoundedAudioQueue(capacity=3, overflow_policy=QueueOverflowPolicy.DROP_NEWEST)

    assert q.push(_make_block(1)) is True
    assert q.push(_make_block(2)) is True
    assert q.push(_make_block(3)) is True
    # 4th and 5th should be rejected
    assert q.push(_make_block(4)) is False
    assert q.push(_make_block(5)) is False

    metrics = q.get_metrics()
    assert metrics["overflow_count"] == 2
    assert metrics["dropped_blocks_count"] == 2

    # Retained items are 1, 2, 3
    popped = [q.pop(timeout=0.1).sequence_number for _ in range(3)]
    assert popped == [1, 2, 3]


def test_never_silently_discard_data():
    """Verifies that every dropped block is recorded in the dropped log with timestamp and reason."""
    q = BoundedAudioQueue(capacity=2, overflow_policy=QueueOverflowPolicy.DROP_OLDEST)

    q.push(_make_block(101))
    q.push(_make_block(102))
    q.push(_make_block(103))  # Evicts 101

    metrics = q.get_metrics()
    drops = metrics["recent_drops_sample"]
    assert len(drops) == 1
    assert drops[0]["sequence_number"] == 101
    assert drops[0]["reason"] == "OVERFLOW_DROP_OLDEST"
    assert "drop_timestamp" in drops[0]


def test_queue_clean_shutdown():
    """Verifies that closing the queue immediately unblocks waiting consumers."""
    q = BoundedAudioQueue(capacity=5)
    consumer_result = []

    def consumer():
        blk = q.pop(timeout=5.0)  # Long wait
        consumer_result.append(blk)

    t = threading.Thread(target=consumer)
    t.start()

    time.sleep(0.05)
    q.close()
    t.join(timeout=1.0)

    assert not t.is_alive()
    assert len(consumer_result) == 1
    assert consumer_result[0] is None
    assert q.is_closed is True

