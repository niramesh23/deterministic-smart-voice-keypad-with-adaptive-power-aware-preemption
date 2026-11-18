"""
tests/test_audio_framing.py
---------------------------
Automated synthetic tests for numerical audio representation and sliding window framing.
"""

import numpy as np
import pytest

from src.audio.framing import pcm16_to_float32, float32_to_pcm16, SlidingWindowFramer


def test_pcm16_to_float32_conversion():
    """Verifies that 16-bit PCM maps symmetrically to [-1.0, 1.0]."""
    pcm_samples = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
    float_samples = pcm16_to_float32(pcm_samples)

    assert float_samples.dtype == np.float32
    assert pytest.approx(float_samples[0], rel=1e-4) == -1.0
    assert pytest.approx(float_samples[1], rel=1e-4) == -0.5
    assert pytest.approx(float_samples[2], rel=1e-4) == 0.0
    assert pytest.approx(float_samples[3], rel=1e-4) == 0.5
    assert float_samples[4] <= 1.0


def test_float32_to_pcm16_clipping():
    """Verifies clipping guard when converting float32 to int16."""
    floats = np.array([-1.5, -1.0, 0.0, 1.0, 1.5], dtype=np.float32)
    pcm = float32_to_pcm16(floats)

    assert pcm.dtype == np.int16
    assert pcm[0] == -32767
    assert pcm[1] == -32767
    assert pcm[2] == 0
    assert pcm[3] == 32767
    assert pcm[4] == 32767


def test_sliding_window_framer():
    """Verifies that 320-sample hops produce 640-sample frames with 50% overlap."""
    framer = SlidingWindowFramer(frame_size=640, hop_size=320)

    hop1 = np.ones(320, dtype=np.int16) * 10
    hop2 = np.ones(320, dtype=np.int16) * 20
    hop3 = np.ones(320, dtype=np.int16) * 30

    frame1 = framer.process_hop(hop1)
    assert len(frame1) == 640
    # First 320 is initial zero history, next 320 is hop1
    np.testing.assert_array_equal(frame1[:320], np.zeros(320, dtype=np.int16))
    np.testing.assert_array_equal(frame1[320:], hop1)

    frame2 = framer.process_hop(hop2)
    assert len(frame2) == 640
    # First 320 is hop1 (history), second 320 is hop2
    np.testing.assert_array_equal(frame2[:320], hop1)
    np.testing.assert_array_equal(frame2[320:], hop2)

    frame3 = framer.process_hop(hop3)
    assert len(frame3) == 640
    # First 320 is hop2 (history), second 320 is hop3
    np.testing.assert_array_equal(frame3[:320], hop2)
    np.testing.assert_array_equal(frame3[320:], hop3)

