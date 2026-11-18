"""
scripts/evaluate_realtime_scheduling.py
---------------------------------------
Stage 7 Evaluation & Benchmarking:
Application-Level Real-Time Scheduling (Baseline FIFO, Rate Monotonic, Earliest Deadline First).

Execution Steps:
1. Synthetic Ordering & Contention Verification:
   - Evaluates FIFO, RM, and EDF under controlled job contention and burst arrivals.
   - Measures dispatch sequence, decision overhead, response times, and deadline misses.
2. Real Pipeline Benchmark on Identical Audio Workload:
   - Streams 10 verified test recordings (yes, no, up, down, left, right, on, off, stop, go)
     through ScheduledKeywordRecognizer under:
     * Mode A: Baseline (FIFO)
     * Mode B: Rate Monotonic (RM)
     * Mode C: Earliest Deadline First (EDF)
   - Measures per-job execution time, response time, queue delay, jitter, overhead, and deadline misses.
3. Live Physical Microphone Hardware Verification:
   - Tests ScheduledKeywordRecognizer with real Windows microphone hardware.
4. Saves structured JSON to results/realtime_scheduling_results.json and generates Markdown report.
"""

from collections import defaultdict
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.scheduled_recognizer import ScheduledKeywordRecognizer
from src.audio.microphone import list_audio_input_devices, get_default_input_device
from src.ml.classes import CLASS_NAMES, CORE_COMMANDS
from src.ml.dataset import load_wav_16k_mono
from src.ml.inference import RecognitionResult
from src.application.live_recognizer import LivePipelineTelemetry
from src.rts.tasks import (
    RealTimeJob,
    JobType,
    JobStatus,
    create_audio_block_job,
    create_feature_extraction_job,
    create_keyword_inference_job,
    create_monitoring_job,
)
from src.rts.scheduler import (
    RealTimeScheduler,
    SchedulingPolicy,
    LateJobStrategy,
    SchedulingDecision,
)
from src.rts.monitor import RealTimeMonitor, DistributionStats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_rts")


def synthetic_cpu_work(duration_ms: float):
    """Simulates active CPU work for a given number of milliseconds."""
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1000.0 < duration_ms:
        pass


def run_synthetic_contention_benchmark() -> Dict[str, Any]:
    """
    Executes controlled synthetic job suites under FIFO, RM, and EDF.
    Contention Scenario:
    Simultaneous release of 16 mixed jobs:
    - 8 AudioBlock jobs (T=20ms, D=20ms, C=0.2ms, P=1)
    - 4 FeatureExtraction jobs (T=40ms, D=40ms, C=0.5ms, P=2)
    - 2 KeywordInference jobs (T=100ms, D=100ms, C=3.5ms, P=3)
    - 2 Monitoring jobs (T=200ms, D=200ms, C=0.3ms, P=4)
    """
    results: Dict[str, Any] = {}

    for policy in [SchedulingPolicy.FIFO, SchedulingPolicy.RM, SchedulingPolicy.EDF]:
        sched = RealTimeScheduler(policy=policy, max_queue_capacity=100)
        t_base = time.perf_counter()

        # Submit jobs in interleaved arrival order to test reordering
        jobs_to_submit = []

        # Interleave submission
        for i in range(2):
            jobs_to_submit.append(
                RealTimeJob(
                    job_id=f"mon_{i}",
                    name=f"Monitoring_{i}",
                    job_type=JobType.MONITORING,
                    work_fn=synthetic_cpu_work,
                    args=(0.3,),
                    period_ms=200.0,
                    relative_deadline_ms=200.0,
                    release_time=t_base,
                    fixed_priority=4,
                )
            )
            jobs_to_submit.append(
                RealTimeJob(
                    job_id=f"infer_{i}",
                    name=f"KeywordInference_{i}",
                    job_type=JobType.KEYWORD_INFERENCE,
                    work_fn=synthetic_cpu_work,
                    args=(3.5,),
                    period_ms=100.0,
                    relative_deadline_ms=100.0,
                    release_time=t_base + (i * 0.0001),
                    fixed_priority=3,
                )
            )

        for i in range(4):
            jobs_to_submit.append(
                RealTimeJob(
                    job_id=f"feat_{i}",
                    name=f"FeatureExtraction_{i}",
                    job_type=JobType.FEATURE_EXTRACTION,
                    work_fn=synthetic_cpu_work,
                    args=(0.5,),
                    period_ms=40.0,
                    relative_deadline_ms=40.0,
                    release_time=t_base + (i * 0.0001),
                    fixed_priority=2,
                )
            )

        for i in range(8):
            jobs_to_submit.append(
                RealTimeJob(
                    job_id=f"audio_{i}",
                    name=f"AudioBlock_{i}",
                    job_type=JobType.AUDIO_BLOCK,
                    work_fn=synthetic_cpu_work,
                    args=(0.2,),
                    period_ms=20.0,
                    relative_deadline_ms=20.0,
                    release_time=t_base + (i * 0.0001),
                    fixed_priority=1,
                )
            )

        # Submit all jobs
        for j in jobs_to_submit:
            sched.submit_job(j)

        # Execute all pending jobs sequentially
        executed_count = sched.run_all_pending()

        monitor = RealTimeMonitor()
        analysis = monitor.analyze_scheduler_run(sched)

        decisions = sched.get_decision_history()
        execution_order = [d.selected_job_id for d in decisions]
        job_types_order = [d.selected_job_type for d in decisions]

        results[policy.value] = {
            "execution_order": execution_order,
            "job_types_order": job_types_order,
            "analysis": analysis,
            "decision_samples": [d.to_dict() for d in decisions[:6]],
        }

    return results


def run_overload_contention_benchmark() -> Dict[str, Any]:
    """
    Overload scenario demonstrating deadline miss protection under RM and EDF vs FIFO.
    Scenario:
    3 Heavy compute jobs (C=8.0ms each, D=100ms, P=3) arrive just before
    5 urgent AudioBlock jobs (C=0.2ms each, D=20ms, P=1).
    Total heavy execution = 24.0 ms (> 20 ms audio deadline).
    - Under FIFO: Heavy jobs run first; all 5 audio jobs miss their 20ms deadline!
    - Under RM: Audio jobs (P=1) preempt the queue and finish in ~1ms (0% audio misses).
    - Under EDF: Audio jobs (d=now+20ms) execute before heavy jobs (d=now+100ms) (0% audio misses).
    """
    results: Dict[str, Any] = {}

    for policy in [SchedulingPolicy.FIFO, SchedulingPolicy.RM, SchedulingPolicy.EDF]:
        sched = RealTimeScheduler(policy=policy, late_job_strategy=LateJobStrategy.EXECUTE_AND_RECORD)

        # Submit 3 heavy jobs first
        for i in range(3):
            now = time.perf_counter()
            sched.submit_job(
                RealTimeJob(
                    job_id=f"heavy_{i}",
                    name=f"HeavyCompute_{i}",
                    job_type=JobType.KEYWORD_INFERENCE,
                    work_fn=synthetic_cpu_work,
                    args=(8.0,),
                    period_ms=100.0,
                    relative_deadline_ms=100.0,
                    release_time=now,
                    fixed_priority=3,
                )
            )

        # Allow slight physical progression (0.5 ms) before audio blocks arrive
        time.sleep(0.0005)

        # Submit 5 urgent audio jobs slightly after
        for i in range(5):
            now = time.perf_counter()
            sched.submit_job(
                RealTimeJob(
                    job_id=f"audio_{i}",
                    name=f"AudioBlock_{i}",
                    job_type=JobType.AUDIO_BLOCK,
                    work_fn=synthetic_cpu_work,
                    args=(0.2,),
                    period_ms=20.0,
                    relative_deadline_ms=20.0,
                    release_time=now,
                    fixed_priority=1,
                )
            )

        sched.run_all_pending()
        monitor = RealTimeMonitor()
        analysis = monitor.analyze_scheduler_run(sched)

        completed = sched.get_completed_jobs()
        audio_misses = sum(1 for j in completed if j.job_type == JobType.AUDIO_BLOCK and j.deadline_missed)
        heavy_misses = sum(1 for j in completed if j.job_type == JobType.KEYWORD_INFERENCE and j.deadline_missed)

        results[policy.value] = {
            "execution_order": [d.selected_job_id for d in sched.get_decision_history()],
            "total_jobs": len(completed),
            "total_deadline_misses": analysis["total_deadline_misses"],
            "deadline_miss_ratio_pct": analysis["deadline_miss_ratio_pct"],
            "audio_block_deadline_misses": audio_misses,
            "heavy_job_deadline_misses": heavy_misses,
            "audio_block_response_time_ms": analysis["per_job_type_breakdown"]["AUDIO_BLOCK"]["response_time_ms"],
        }

    return results


def run_pipeline_benchmark_for_policy(
    policy: SchedulingPolicy,
    core_tests: List[Tuple[str, pathlib.Path, np.ndarray]],
) -> Dict[str, Any]:
    """
    Runs the full speech recognition pipeline under the specified scheduling policy.
    Uses the exact same 10 core command audio recordings.
    """
    logger.info(f"--- Running Pipeline Benchmark under {policy.value} ---")
    recognizer = ScheduledKeywordRecognizer(
        policy=policy,
        late_job_strategy=LateJobStrategy.EXECUTE_AND_RECORD,
        queue_capacity=100,
    )
    recognizer.start(use_microphone=False)

    confirmed_results: List[Dict[str, Any]] = []

    def on_command(res: RecognitionResult, tel: LivePipelineTelemetry):
        confirmed_results.append({
            "command": res.command,
            "confidence": round(res.confidence, 4),
            "reaction_latency_ms": tel.reaction_latency_ms,
            "feature_extraction_ms": tel.feature_extraction_ms,
            "model_inference_ms": tel.model_inference_ms,
            "postprocessing_ms": tel.postprocessing_ms,
            "queue_delay_ms": tel.queue_delay_ms,
        })

    recognizer.on_command_callback = on_command

    silence_pad = np.zeros(3200, dtype=np.int16)  # 200 ms silence between words
    hop_size = 320

    for cmd, path, audio_f32 in core_tests:
        audio_pcm16 = np.clip(audio_f32 * 32767.0, -32768, 32767).astype(np.int16)
        full_stream = np.concatenate([silence_pad, audio_pcm16, silence_pad])
        num_hops = len(full_stream) // hop_size

        for h in range(num_hops):
            hop = full_stream[h * hop_size : (h + 1) * hop_size]
            recognizer.push_external_hop(hop)
            # Sleep ~1 ms to yield to worker thread, simulating near-real-time streaming
            time.sleep(0.001)

        # Allow worker thread to drain pending jobs
        time.sleep(0.05)

    # Wait for completion of remaining jobs
    time.sleep(0.15)
    recognizer.stop()

    analysis = recognizer.get_analysis_report()
    decisions = recognizer.scheduler.get_decision_history()

    return {
        "policy": policy.value,
        "confirmed_commands_count": len(confirmed_results),
        "confirmed_commands": confirmed_results,
        "timing_analysis": analysis,
        "sample_decisions": [d.to_dict() for d in decisions[:10]],
    }


def run_live_microphone_verification() -> Dict[str, Any]:
    """
    Runs physical live microphone acquisition through ScheduledKeywordRecognizer
    with real-time EDF scheduling.
    """
    logger.info("--- Testing Physical Microphone Hardware with Real-Time Scheduling ---")
    dev = get_default_input_device()
    recognizer = ScheduledKeywordRecognizer(
        policy=SchedulingPolicy.EDF,
        device_id=dev.device_id,
        queue_capacity=100,
    )

    recognizer.start(use_microphone=True)

    # Stream from physical microphone for 2.5 seconds (125 hops @ 20ms)
    time.sleep(2.5)

    recognizer.stop()
    analysis = recognizer.get_analysis_report()
    decisions = recognizer.scheduler.get_decision_history()

    return {
        "device_name": dev.name,
        "device_id": dev.device_id,
        "hops_ingested": recognizer.total_audio_blocks_ingested,
        "hops_dropped": recognizer.total_audio_blocks_dropped,
        "hops_processed": recognizer.total_hops_processed,
        "ping_pong_status": recognizer.ping_pong.get_status(),
        "noise_floor_adapted": round(float(recognizer.vad.noise_floor.noise_floor), 6),
        "total_jobs_executed": analysis["total_jobs_executed"],
        "total_deadline_misses": analysis["total_deadline_misses"],
        "deadline_miss_ratio_pct": analysis["deadline_miss_ratio_pct"],
        "mean_scheduling_overhead_us": analysis["global_distributions"]["scheduling_overhead_us"]["mean"],
        "decisions_recorded": len(decisions),
        "passed": recognizer.total_audio_blocks_ingested > 50 and recognizer.total_audio_blocks_dropped == 0,
    }


def main():
    logger.info("Starting Stage 7 Real-Time Scheduling Evaluation & Timing Analysis...")
    t_start = time.perf_counter()

    root = pathlib.Path("data/raw")
    manifest_path = pathlib.Path("data/manifests/test_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        test_manifest = json.load(f)

    test_samples_by_word = defaultdict(list)
    for r in test_manifest:
        if not r.get("is_silence_slice"):
            test_samples_by_word[r["label"]].append(r["path"])

    core_tests = []
    for cmd in CORE_COMMANDS:
        cmd_paths = test_samples_by_word.get(cmd, [])
        if not cmd_paths:
            continue
        wav_path = root / cmd_paths[0]
        audio_f32 = load_wav_16k_mono(wav_path)
        core_tests.append((cmd, wav_path, audio_f32))

    logger.info(f"Loaded {len(core_tests)} verified core command test samples.")

    # 1. Synthetic Contention Benchmark
    logger.info("Running synthetic contention benchmarks...")
    synthetic_results = run_synthetic_contention_benchmark()

    # 1b. Synthetic Overload Benchmark
    logger.info("Running synthetic overload benchmarks...")
    overload_results = run_overload_contention_benchmark()

    # 2. Pipeline Benchmark across Baseline FIFO, RM, and EDF
    pipeline_results: Dict[str, Any] = {}
    for policy in [SchedulingPolicy.FIFO, SchedulingPolicy.RM, SchedulingPolicy.EDF]:
        res = run_pipeline_benchmark_for_policy(policy, core_tests)
        pipeline_results[policy.value] = res

    # 3. Physical Hardware Microphone Verification
    logger.info("Running live microphone verification...")
    mic_results = run_live_microphone_verification()

    # Compile Final JSON Report
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_s": round(time.perf_counter() - t_start, 2),
        "job_specifications": {
            "AUDIO_BLOCK": {
                "period_ms": 20.0,
                "relative_deadline_ms": 20.0,
                "fixed_priority": 1,
                "wcet_ms": 0.15,
                "justification": "Input audio sampled at 16 kHz in 320-sample hops (=20ms). Stability deadline D=T=20ms prevents buffer accumulation.",
            },
            "FEATURE_EXTRACTION": {
                "period_ms": 40.0,
                "relative_deadline_ms": 40.0,
                "fixed_priority": 2,
                "wcet_ms": 0.50,
                "justification": "Log-Mel spectrogram generation (49x40) from 1.0s context. Leaves ample budget for neural inference within 100ms total.",
            },
            "KEYWORD_INFERENCE": {
                "period_ms": 100.0,
                "relative_deadline_ms": 100.0,
                "fixed_priority": 3,
                "wcet_ms": 3.50,
                "justification": "Residual DS-CNN forward pass and decision logic. 100ms deadline meets human conversational feedback bounds (<150ms).",
            },
            "MONITORING": {
                "period_ms": 200.0,
                "relative_deadline_ms": 200.0,
                "fixed_priority": 4,
                "wcet_ms": 0.30,
                "justification": "System telemetry, CPU tracking, and queue metrics at 5 Hz. Non-critical background task that yields to all audio jobs.",
            },
        },
        "synthetic_contention_benchmark": synthetic_results,
        "synthetic_overload_benchmark": overload_results,
        "pipeline_benchmarks": pipeline_results,
        "live_microphone_verification": mic_results,
    }

    results_dir = pathlib.Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "realtime_scheduling_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"Results successfully saved to {json_path}")
    return final_output


if __name__ == "__main__":
    main()
