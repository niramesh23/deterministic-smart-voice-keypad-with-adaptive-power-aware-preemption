#!/usr/bin/env python3
"""
scripts/test_microphone.py
--------------------------
Demonstration and benchmarking script for physical microphone acquisition,
software ping-pong double buffering, bounded queue telemetry, and overflow detection.

Saves measured benchmark metrics to results/microphone_benchmark.txt.
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.buffers import SoftwarePingPongBuffer
from src.audio.framing import pcm16_to_float32, SlidingWindowFramer
from src.audio.microphone import (
    MicrophoneCapture,
    list_audio_input_devices,
    get_default_input_device,
    verify_device_support,
)
from src.audio.queue import BoundedAudioQueue
from src.audio.types import QueueOverflowPolicy


def run_microphone_benchmark(
    device_id: int = None,
    duration_s: float = 2.0,
    simulate_overflow: bool = True,
    output_path: str = "results/microphone_benchmark.txt",
):
    print("=" * 80)
    print("  STAGE 3: REAL MICROPHONE & BUFFERING SUBSYSTEM BENCHMARK")
    print("=" * 80)

    # 1. Device enumeration
    devices = list_audio_input_devices()
    print(f"Discovered {len(devices)} audio input device(s):")
    for d in devices:
        print(f"  {d}")

    default_dev = get_default_input_device()
    target_device_id = device_id if device_id is not None else default_dev.device_id
    print(f"\nSelected Target Device ID: {target_device_id}")

    # 2. Format verification
    target_sr = 16000
    target_ch = 1
    target_dtype = "int16"
    hop_size = 320   # 20.0 ms @ 16 kHz
    frame_size = 640  # 40.0 ms @ 16 kHz

    is_supported, err = verify_device_support(target_device_id, target_sr, target_ch, target_dtype)
    print(f"Requested Format: {target_sr} Hz, {target_ch} ch (mono), 16-bit PCM, hop={hop_size} ({hop_size/target_sr*1000:.1f} ms)")
    print(f"Hardware Format Negotiation: {'SUCCESSFUL' if is_supported else f'FAILED: {err}'}")
    if not is_supported:
        print("[ERROR] Device cannot support requested format directly.")
        sys.exit(1)

    # 3. Setup Buffers & Queue
    queue_cap = 50
    ping_pong = SoftwarePingPongBuffer(buffer_size=hop_size, dtype=np.int16)
    queue = BoundedAudioQueue(capacity=queue_cap, overflow_policy=QueueOverflowPolicy.DROP_OLDEST)
    framer = SlidingWindowFramer(frame_size=frame_size, hop_size=hop_size)

    mic = MicrophoneCapture(
        device_id=target_device_id,
        sample_rate=target_sr,
        channels=target_ch,
        hop_size=hop_size,
        dtype=target_dtype,
        queue=queue,
        ping_pong=ping_pong,
    )

    print("\nStarting live microphone acquisition...")
    processed_blocks = []
    processed_frames = []
    rms_values = []

    t_start = time.perf_counter()

    with mic:
        print(f"Audio stream opened successfully.")
        print(f"  Verified Device Name:  {mic.actual_device_name}")
        print(f"  Verified Sample Rate:  {mic.actual_sample_rate} Hz")
        print(f"  Verified Channels:     {mic.actual_channels}")
        print(f"  Reported HW Latency:   {mic.actual_latency * 1000:.2f} ms")
        print("\nCapturing live audio stream (real-time consumer running)...")

        while (time.perf_counter() - t_start) < duration_s:
            block = queue.pop(timeout=0.05)
            if block is not None:
                processed_blocks.append(block)

                # Consume the ready software ping-pong buffer slot to complete lifecycle
                _ = ping_pong.read_ready_buffer(timeout=0)

                # Convert to float32 and calculate frame RMS
                float_data = block.to_float32()
                frame = framer.process_hop(block.data)
                processed_frames.append(frame)

                rms = float(np.sqrt(np.mean(float_data ** 2)))
                rms_values.append(rms)

    print(f"Live capture complete: {len(processed_blocks)} blocks retrieved.")

    # 4. Optional Demonstration of Queue Overflow Detection
    overflow_demo_result = None
    if simulate_overflow:
        print("\nDemonstrating Queue Overflow Detection under consumer lag...")
        small_queue = BoundedAudioQueue(capacity=5, overflow_policy=QueueOverflowPolicy.DROP_OLDEST)
        mic_overflow = MicrophoneCapture(
            device_id=target_device_id,
            sample_rate=target_sr,
            channels=target_ch,
            hop_size=hop_size,
            dtype=target_dtype,
            queue=small_queue,
            ping_pong=None,
        )

        with mic_overflow:
            # Let producer fill and overflow capacity-5 queue without consumer reading for 0.4s
            time.sleep(0.4)  # ~20 blocks produced, capacity is 5 -> ~15 drops expected

        overflow_demo_result = small_queue.get_metrics()
        print(f"  Queue Capacity:         {overflow_demo_result['capacity']}")
        print(f"  Enqueued Count:         {overflow_demo_result['total_enqueued']}")
        print(f"  Overflow Count:         {overflow_demo_result['overflow_count']}")
        print(f"  Dropped Blocks Count:   {overflow_demo_result['dropped_blocks_count']}")
        print(f"  Dropped Log Entries:    {len(overflow_demo_result['recent_drops_sample'])}")

    # 5. Compile Measurements
    diag = mic.get_diagnostics()
    pp_status = ping_pong.get_status()
    q_metrics = queue.get_metrics()

    mean_rms = sum(rms_values) / len(rms_values) if rms_values else 0.0
    peak_rms = max(rms_values) if rms_values else 0.0

    # 6. Format Detailed Report
    out_file = pathlib.Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 80)
    lines.append("STAGE 3: REAL MICROPHONE & BUFFERING SUBSYSTEM BENCHMARK REPORT")
    lines.append("Project: Real-Time Adaptive Voice Recognition Using Python")
    lines.append("=" * 80)
    lines.append(f"Audit Date / Timestamp:       {now_str}")
    lines.append(f"Operating System:             {sys.platform} (Windows)")
    lines.append(f"Python Environment:           Python {sys.version.split()[0]}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("1. AUDIO HARDWARE & FORMAT VERIFICATION")
    lines.append("-" * 80)
    lines.append(f"Selected Device Index:        {diag['device_id']}")
    lines.append(f"Selected Device Name:         {diag['device_name']}")
    lines.append(f"Target Sampling Rate:         {target_sr} Hz")
    lines.append(f"Actual Verified Rate:         {diag['actual_sample_rate']} Hz")
    lines.append(f"Target Channels:              {target_ch} (Mono)")
    lines.append(f"Actual Verified Channels:     {diag['actual_channels']}")
    lines.append(f"Sample Data Format:           16-bit signed PCM (int16), converted to float32 [-1.0, 1.0]")
    lines.append(f"Hop Size:                     {hop_size} samples ({hop_size/target_sr*1000:.1f} ms)")
    lines.append(f"Frame Size:                   {frame_size} samples ({frame_size/target_sr*1000:.1f} ms, 50% overlap)")
    lines.append(f"Hardware Stream Latency:      {diag['actual_latency_s']*1000:.2f} ms")
    lines.append(f"Stream Status Flags:          {diag['stream_status_flags'] if diag['stream_status_flags'] else 'None (Clean stream)'}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("2. SOFTWARE-MANAGED PING-PONG BUFFER TELEMETRY")
    lines.append("-" * 80)
    lines.append("NOTE: These are software-managed application buffers, NOT hardware DMA buffers.")
    lines.append(f"Buffer Slot 0 (Ping) Reuses:  {pp_status['slot_0_reuses']}")
    lines.append(f"Buffer Slot 1 (Pong) Reuses:  {pp_status['slot_1_reuses']}")
    lines.append(f"Total Writes to Ping-Pong:    {pp_status['total_written']}")
    lines.append(f"Total Overwritten / Overruns: {pp_status['overrun_count']}")
    lines.append(f"Total Underruns Reported:     {pp_status['underrun_count']}")
    lines.append(f"Final Slot 0 State:           {pp_status['slot_0_state']}")
    lines.append(f"Final Slot 1 State:           {pp_status['slot_1_state']}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("3. BOUNDED PRODUCER-CONSUMER QUEUE METRICS")
    lines.append("-" * 80)
    lines.append(f"Configured Queue Capacity:    {q_metrics['capacity']} blocks")
    lines.append(f"Overflow Policy:              {q_metrics['overflow_policy']}")
    lines.append(f"Total Blocks Enqueued:        {q_metrics['total_enqueued']}")
    lines.append(f"Total Blocks Dequeued:        {q_metrics['total_dequeued']}")
    lines.append(f"Peak Queue Occupancy:         {q_metrics['peak_occupancy']} blocks")
    lines.append(f"Final Queue Occupancy:        {q_metrics['current_occupancy']} blocks")
    lines.append(f"Live Stream Overflows:        {q_metrics['overflow_count']}")
    lines.append(f"Live Stream Dropped Blocks:   {q_metrics['dropped_blocks_count']}")
    lines.append(f"Mean Queue Wait Latency:      {q_metrics['mean_queue_wait_ms']:.3f} ms")
    lines.append(f"P95 Queue Wait Latency:       {q_metrics['p95_queue_wait_ms']:.3f} ms")
    lines.append("")

    if overflow_demo_result:
        lines.append("-" * 80)
        lines.append("4. QUEUE OVERFLOW DETECTION EXPERIMENT (ARTIFICIAL CONSUMER LAG)")
        lines.append("-" * 80)
        lines.append(f"Experimental Small Capacity:  {overflow_demo_result['capacity']} blocks")
        lines.append(f"Produced Blocks under Stall:  {overflow_demo_result['total_enqueued'] + overflow_demo_result['dropped_blocks_count']}")
        lines.append(f"Detected Overflow Events:     {overflow_demo_result['overflow_count']}")
        lines.append(f"Total Dropped Blocks:         {overflow_demo_result['dropped_blocks_count']}")
        lines.append("Sample Dropped Block Log (Never Silently Discarded):")
        for drop in overflow_demo_result['recent_drops_sample']:
            lines.append(f"  Seq: {drop['sequence_number']}, Timestamp: {drop['capture_timestamp']:.4f}, Reason: {drop['reason']}")
        lines.append("")

    lines.append("-" * 80)
    lines.append("5. REAL-TIME CALLBACK TIMING & JITTER")
    lines.append("-" * 80)
    lines.append(f"Total Audio Callbacks:        {diag['total_callbacks']}")
    lines.append(f"Total PCM Samples Ingested:   {diag['total_samples_captured']:,}")
    lines.append(f"Mean Callback Execution Time: {diag['mean_callback_duration_ms']:.4f} ms (Deadline budget: ~1.0 ms)")
    lines.append(f"Max Callback Execution Time:  {diag['max_callback_duration_ms']:.4f} ms")
    lines.append(f"Target Callback Interval:     20.000 ms (320 samples @ 16 kHz)")
    lines.append(f"Mean Callback Interval:       {diag['mean_callback_interval_ms']:.3f} ms")
    lines.append(f"P95 Callback Interval:        {diag['p95_callback_interval_ms']:.3f} ms")
    lines.append(f"Signal Mean RMS (Acoustic):   {mean_rms:.5f}")
    lines.append(f"Signal Peak RMS (Acoustic):   {peak_rms:.5f}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("6. WINDOWS SCHEDULING & PYTHON CALLBACK LIMITATIONS")
    lines.append("-" * 80)
    lines.append("A. Windows Non-RTOS Pre-emption:")
    lines.append("   Windows is a general-purpose operating system. Default thread scheduling quantum is")
    lines.append("   approximately 15.625 ms (or 1.0 ms with high-resolution timer requests). High system load,")
    lines.append("   Deferred Procedure Calls (DPCs), or Windows background services can cause transient thread delays.")
    lines.append("")
    lines.append("B. Python Global Interpreter Lock (GIL):")
    lines.append("   PortAudio invokes the Python callback from an unmanaged native C thread, acquiring the GIL")
    lines.append("   via PyGILState_Ensure. If another Python thread is executing a long-running CPU-bound bytecode")
    lines.append("   instruction, the callback may experience latency jitter before gaining GIL ownership.")
    lines.append("   Mitigation: The callback performs strictly O(1) memory copy and non-blocking queue push,")
    lines.append("   leaving all DSP, feature extraction, and ML inference to downstream consumer threads.")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save JSON summary
    json_path = out_file.parent / "audio_live_capture.json"
    summary_dict = {
        "device_id": diag["device_id"],
        "device_name": diag["device_name"],
        "actual_sample_rate": diag["actual_sample_rate"],
        "actual_channels": diag["actual_channels"],
        "actual_latency_s": diag["actual_latency_s"],
        "total_callbacks": diag["total_callbacks"],
        "total_samples": diag["total_samples_captured"],
        "mean_callback_duration_ms": diag["mean_callback_duration_ms"],
        "max_callback_duration_ms": diag["max_callback_duration_ms"],
        "mean_callback_interval_ms": diag["mean_callback_interval_ms"],
        "p95_callback_interval_ms": diag["p95_callback_interval_ms"],
        "mean_rms": mean_rms,
        "peak_rms": peak_rms,
        "ping_pong_status": pp_status,
        "queue_metrics": q_metrics,
        "overflow_demo": overflow_demo_result,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nBenchmark completed successfully.")
    print(f"Report written to: {out_file.resolve()}")
    print(f"JSON metrics to:   {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and benchmark real microphone acquisition.")
    parser.add_argument("--device-id", type=int, default=None, help="Input device ID (default: system default)")
    parser.add_argument("--duration", type=float, default=2.0, help="Test duration in seconds (default: 2.0)")
    parser.add_argument("--no-overflow-demo", action="store_true", help="Skip the artificial overflow demonstration")
    args = parser.parse_args()

    run_microphone_benchmark(
        device_id=args.device_id,
        duration_s=args.duration,
        simulate_overflow=not args.no_overflow_demo,
    )
