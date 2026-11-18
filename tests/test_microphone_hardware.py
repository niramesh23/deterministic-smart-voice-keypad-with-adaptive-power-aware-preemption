"""
tests/test_microphone_hardware.py
---------------------------------
Hardware-dependent test executing real microphone capture on Windows PC.
Distinguished from synthetic tests via pytest marker 'hardware'.
"""

import json
import pathlib
import time
import pytest

from src.audio.buffers import SoftwarePingPongBuffer
from src.audio.microphone import (
    MicrophoneCapture,
    list_audio_input_devices,
    get_default_input_device,
)
from src.audio.queue import BoundedAudioQueue
from src.audio.types import QueueOverflowPolicy
from src.dsp.vad import VoiceActivityDetector


@pytest.mark.hardware
def test_real_microphone_hardware_live():
    """
    Executes live audio acquisition on the physical host microphone.
    Validates:
    1. Audio device discovery and default selection.
    2. Actual sample rate (16,000 Hz) and 1-channel mono negotiation.
    3. Block acquisition (320 samples / 20 ms hop).
    4. Software ping-pong buffer alternation and reuse counts.
    5. Producer-consumer queue operation and timing telemetry.
    6. Clean stream shutdown and release.
    """
    # 1. Device discovery
    devices = list_audio_input_devices()
    assert len(devices) > 0, "No audio input devices found on system!"

    default_dev = get_default_input_device()
    assert default_dev is not None
    assert default_dev.max_input_channels > 0

    # 2. Setup subsystems
    ping_pong = SoftwarePingPongBuffer(buffer_size=320, dtype="int16")
    queue = BoundedAudioQueue(capacity=100, overflow_policy=QueueOverflowPolicy.DROP_OLDEST)

    mic = MicrophoneCapture(
        device_id=default_dev.device_id,
        sample_rate=16000,
        channels=1,
        hop_size=320,
        dtype="int16",
        queue=queue,
        ping_pong=ping_pong,
    )

    # 3. Execute live capture for 1.0 second
    captured_blocks = []
    capture_start = time.perf_counter()

    with mic:
        assert mic.is_running is True
        # Run live acquisition for 1.0 second (expecting ~50 hops)
        while (time.perf_counter() - capture_start) < 1.0:
            block = queue.pop(timeout=0.05)
            if block is not None:
                # Read matching slot from software ping-pong buffer
                _ = ping_pong.read_ready_buffer(timeout=0)
                captured_blocks.append(block)

    # 4. Stream should shut down cleanly
    assert mic.is_running is False

    # 5. Verify captured audio and actual parameters
    diag = mic.get_diagnostics()
    assert diag["actual_sample_rate"] == 16000.0, f"Expected 16000 Hz, got {diag['actual_sample_rate']}"
    assert diag["actual_channels"] == 1, f"Expected 1 channel, got {diag['actual_channels']}"
    assert len(captured_blocks) > 0, "Failed to capture any audio blocks from live microphone!"

    # 6. Verify each block properties
    for blk in captured_blocks:
        assert len(blk.data) == 320, f"Expected 320 samples per block, got {len(blk.data)}"
        assert blk.sample_rate == 16000
        assert blk.channels == 1
        assert blk.buffer_id in (0, 1)

    # 7. Verify ping-pong buffer alternation
    pp_status = ping_pong.get_status()
    assert pp_status["slot_0_reuses"] > 0, "Slot 0 (Ping) was never written!"
    assert pp_status["slot_1_reuses"] > 0, "Slot 1 (Pong) was never written!"

    # 8. Save live test report to disk
    results_dir = pathlib.Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "device_name": diag["device_name"],
        "device_id": diag["device_id"],
        "actual_sample_rate": diag["actual_sample_rate"],
        "actual_channels": diag["actual_channels"],
        "actual_latency_s": diag["actual_latency_s"],
        "total_blocks_captured": len(captured_blocks),
        "total_samples": diag["total_samples_captured"],
        "mean_callback_duration_ms": diag["mean_callback_duration_ms"],
        "max_callback_duration_ms": diag["max_callback_duration_ms"],
        "mean_callback_interval_ms": diag["mean_callback_interval_ms"],
        "p95_callback_interval_ms": diag["p95_callback_interval_ms"],
        "ping_pong_status": pp_status,
        "queue_metrics": queue.get_metrics(),
    }
    with open(results_dir / "audio_live_capture.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)


@pytest.mark.hardware
def test_live_microphone_with_dsp_vad():
    """
    Verifies live microphone capture integrated with DSP VAD pipeline in consumer thread.
    Confirms callback execution duration is completely unaffected by DSP processing.
    """
    default_dev = get_default_input_device()
    queue = BoundedAudioQueue(capacity=50)
    ping_pong = SoftwarePingPongBuffer(buffer_size=320, dtype="int16")
    vad = VoiceActivityDetector(sample_rate=16000, hop_size=320, threshold_factor=2.5)

    mic = MicrophoneCapture(
        device_id=default_dev.device_id,
        sample_rate=16000,
        channels=1,
        hop_size=320,
        dtype="int16",
        queue=queue,
        ping_pong=ping_pong,
    )

    t_start = time.perf_counter()
    hops_processed = 0

    with mic:
        while (time.perf_counter() - t_start) < 1.0:
            block = queue.pop(timeout=0.05)
            if block is not None:
                _ = ping_pong.read_ready_buffer(timeout=0)
                # Run DSP in consumer thread (outside audio callback)
                _ = vad.process_hop(block.data)
                hops_processed += 1

    assert hops_processed > 0
    diag = mic.get_diagnostics()
    # Ensure audio callback duration remained non-blocking (< 1.0 ms budget)
    assert diag["mean_callback_duration_ms"] < 0.5
    assert vad.total_hops_processed == hops_processed
