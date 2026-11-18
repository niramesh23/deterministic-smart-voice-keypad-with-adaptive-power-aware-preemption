"""
tests/test_audio_buffers.py
---------------------------
Automated synthetic tests for software-managed ping-pong double buffer.
Verifies alternating transitions, sequence numbers, overrun/underrun detection,
and thread safety.
"""

import threading
import time
import numpy as np
import pytest

from src.audio.buffers import SoftwarePingPongBuffer
from src.audio.types import BufferState


def test_ping_pong_buffer_initialization():
    """Verifies that buffers are allocated with correct size and initial empty states."""
    pp = SoftwarePingPongBuffer(buffer_size=320, dtype=np.int16)
    assert pp.buffer_size == 320
    assert pp.dtype == np.int16
    assert pp.total_written == 0
    assert pp.total_read == 0
    assert pp.overrun_count == 0
    assert pp.underrun_count == 0
    assert pp.reuse_counts == (0, 0)


def test_ping_pong_alternation_and_states():
    """Verifies that successive writes strictly alternate between Ping (0) and Pong (1)."""
    pp = SoftwarePingPongBuffer(buffer_size=320, dtype=np.int16)

    data1 = np.full(320, 100, dtype=np.int16)
    data2 = np.full(320, 200, dtype=np.int16)
    data3 = np.full(320, 300, dtype=np.int16)

    # First write -> goes to slot 0 (Ping)
    slot1, overrun1 = pp.write_block(data1, sequence_number=1)
    assert slot1 == 0
    assert not overrun1
    assert pp.reuse_counts == (1, 0)

    # Second write -> goes to slot 1 (Pong)
    slot2, overrun2 = pp.write_block(data2, sequence_number=2)
    assert slot2 == 1
    assert not overrun2
    assert pp.reuse_counts == (1, 1)

    # Consume slot 0
    read_data, read_slot, seq, _ = pp.read_ready_buffer()
    assert read_slot == 0
    assert seq == 1
    np.testing.assert_array_equal(read_data, data1)

    # Third write -> alternates back to slot 0 (Ping)
    slot3, overrun3 = pp.write_block(data3, sequence_number=3)
    assert slot3 == 0
    assert not overrun3
    assert pp.reuse_counts == (2, 1)


def test_ping_pong_overrun_detection():
    """Verifies that writing without consuming triggers overrun detection."""
    pp = SoftwarePingPongBuffer(buffer_size=320, dtype=np.int16)

    # Write Slot 0 (Ping)
    s0, o0 = pp.write_block(np.ones(320, dtype=np.int16), 1)
    assert not o0

    # Write Slot 1 (Pong)
    s1, o1 = pp.write_block(np.ones(320, dtype=np.int16), 2)
    assert not o1

    # Write Slot 0 again while Slot 0 was NEVER read (still READY!)
    s2, o2 = pp.write_block(np.ones(320, dtype=np.int16), 3)
    assert s2 == 0
    assert o2 is True  # Overwritten detected!
    assert pp.overrun_count >= 1


def test_ping_pong_underrun_reporting():
    """Verifies that reading when no buffer is ready increments underrun counter."""
    pp = SoftwarePingPongBuffer(buffer_size=320, dtype=np.int16)

    # Read immediately without any write
    result = pp.read_ready_buffer(timeout=0)
    assert result is None
    assert pp.underrun_count == 1


def test_ping_pong_concurrent_producer_consumer():
    """Tests thread-safety with concurrent producer and consumer threads."""
    pp = SoftwarePingPongBuffer(buffer_size=320, dtype=np.int16)
    num_blocks = 100
    consumed_seqs = []

    def producer():
        for i in range(num_blocks):
            data = np.full(320, i, dtype=np.int16)
            pp.write_block(data, sequence_number=i)
            time.sleep(0.001)

    def consumer():
        while len(consumed_seqs) < num_blocks:
            res = pp.read_ready_buffer(timeout=0.1)
            if res is not None:
                _, _, seq, _ = res
                consumed_seqs.append(seq)

    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)

    t_cons.start()
    t_prod.start()

    t_prod.join(timeout=3.0)
    t_cons.join(timeout=3.0)

    assert len(consumed_seqs) == num_blocks
    assert consumed_seqs == list(range(num_blocks))
    assert pp.reuse_counts[0] + pp.reuse_counts[1] == num_blocks

