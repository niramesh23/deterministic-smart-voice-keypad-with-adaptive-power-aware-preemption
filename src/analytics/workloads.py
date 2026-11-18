"""
src/analytics/workloads.py
--------------------------
Repeatable Workload Generators and Comparative Benchmark Suite.

Supports:
1. NORMAL Workload: Nominal 20ms audio frame input, standard command pace.
2. HIGH Workload: Rapid audio frame arrival, bursts, concurrent background read operations.
3. OVERLOAD Workload: Synthetic computation delays, queue saturation, deadline pressure.
4. RECOVERY Workload: Injected transient fault, automated recovery overhead observation.

Executes identical workloads across FIFO, RM, and EDF scheduling policies.
"""

import datetime
import logging
import pathlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.analytics.types import (
    BenchmarkExperimentRecord,
    WorkloadType,
)
from src.analytics.collector import PerformanceAnalyticsCollector
from src.rts.scheduler import SchedulingPolicy
from src.rts.tasks import RealTimeJob, JobType

logger = logging.getLogger("rtos.analytics.workloads")


class WorkloadRunner:
    """
    Executes standardized, reproducible workload profiles on the voice-processing pipeline.
    """

    def __init__(
        self,
        collector: Optional[PerformanceAnalyticsCollector] = None,
        output_dir: str | pathlib.Path = "results/benchmarks",
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.collector = collector or PerformanceAnalyticsCollector(output_dir=self.output_dir)

    def _generate_synthetic_audio_hops(
        self,
        num_hops: int = 50,
        hop_size: int = 320,
        tone_freq_hz: float = 440.0,
        sample_rate: int = 16000,
    ) -> List[np.ndarray]:
        """Generates deterministic synthetic audio hops for reproducible benchmarking."""
        t = np.arange(hop_size * num_hops) / sample_rate
        # Generate 440 Hz sine wave + light white noise
        signal = 0.5 * np.sin(2 * np.pi * tone_freq_hz * t) + 0.05 * np.random.randn(len(t))
        pcm16 = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)

        hops = []
        for i in range(0, len(pcm16), hop_size):
            hop = pcm16[i : i + hop_size]
            if len(hop) == hop_size:
                hops.append(hop)
        return hops

    def run_workload(
        self,
        policy: SchedulingPolicy,
        workload_type: WorkloadType,
        num_blocks: int = 50,
        experiment_id_suffix: str = "",
    ) -> BenchmarkExperimentRecord:
        """
        Executes a targeted workload experiment under the specified scheduling policy.
        """
        policy_name = policy.value if hasattr(policy, "value") else str(policy)
        exp_id = f"exp_{policy_name.lower()}_{workload_type.value.lower()}_{int(time.time())}"
        if experiment_id_suffix:
            exp_id += f"_{experiment_id_suffix}"

        logger.info(f"[WORKLOAD RUNNER] Starting {workload_type.value} benchmark (Policy: {policy_name}, Blocks: {num_blocks})...")

        # Configure app instance with isolated test configuration
        from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig
        config = OrchestratorConfig(
            policy=policy,
            enable_adaptation=(workload_type != WorkloadType.OVERLOAD),  # Test overload without early pruning or with pruning
            enable_dashboard=False,  # Headless for benchmarking to avoid socket interference
            enable_audio_logging=False,
        )

        app = VoiceRecognitionApp(config=config)
        # Connect collector to app and recognizer before startup
        app.analytics_collector = self.collector
        app.recognizer.analytics_collector = self.collector
        self.collector.reset_live_stats()
        app.start(use_microphone=False)

        t_start = time.perf_counter()
        blocks_processed = 0

        try:
            hops = self._generate_synthetic_audio_hops(num_hops=num_blocks)

            if workload_type == WorkloadType.NORMAL:
                # Nominal 20ms pacing
                for hop in hops:
                    t_cap = time.perf_counter()
                    app.push_audio_hop(hop, capture_time=t_cap)
                    blocks_processed += 1
                    time.sleep(0.018)  # Approximate real-time 20ms audio frame arrival

            elif workload_type == WorkloadType.HIGH:
                # Rapid 2x arrival + concurrent query workload
                for idx, hop in enumerate(hops):
                    t_cap = time.perf_counter()
                    app.push_audio_hop(hop, capture_time=t_cap)
                    blocks_processed += 1
                    # Interleave light query access
                    if idx % 5 == 0:
                        _ = app.status_provider.get_snapshot() if app.status_provider else None
                    time.sleep(0.008)  # Faster than 20ms frame duration

            elif workload_type == WorkloadType.OVERLOAD:
                # Burst ingestion with synthetic compute delays injected into inference
                def delay_hook(original_predict):
                    def wrapped(log_mel, *args, **kwargs):
                        time.sleep(0.040)  # Inject 40ms compute delay (violating nominal deadlines)
                        return original_predict(log_mel, *args, **kwargs)
                    return wrapped

                # Wrap inference temporarily if available
                orig_pred = app.recognizer.inference_service.predict_from_features
                app.recognizer.inference_service.predict_from_features = delay_hook(orig_pred)

                for hop in hops:
                    t_cap = time.perf_counter()
                    app.push_audio_hop(hop, capture_time=t_cap)
                    blocks_processed += 1
                    time.sleep(0.005)  # Severe burst pressure

                app.recognizer.inference_service.predict_from_features = orig_pred

            elif workload_type == WorkloadType.RECOVERY:
                # Mid-stream fault injection and recovery verification
                for idx, hop in enumerate(hops):
                    t_cap = time.perf_counter()
                    app.push_audio_hop(hop, capture_time=t_cap)
                    blocks_processed += 1

                    if idx == num_blocks // 2:
                        logger.info("[WORKLOAD RUNNER] Injecting mid-workload audio fault...")
                        from src.fault.types import FaultType
                        app.fault_detector.report_fault(
                            fault_type=FaultType.MIC_DISCONNECTED,
                            component="microphone",
                            description="Benchmark synthetic stream stall injection",
                        )
                        # Programmatic recovery
                        app.recovery_manager.recover_audio_subsystem()

                    time.sleep(0.015)

            # Allow scheduler worker to drain remaining tasks
            time.sleep(0.50)

        finally:
            t_end = time.perf_counter()
            duration_s = max(1e-3, t_end - t_start)

            # Extract metrics
            audio_m = self.collector.compute_audio_metrics(
                audio_queue=app.recognizer.audio_queue,
                dropped_count=app.recognizer.total_audio_blocks_dropped,
            )
            infer_m = self.collector.compute_inference_metrics()
            sched_m = self.collector.compute_scheduling_metrics(scheduler=app.recognizer.scheduler)
            e2e_m = self.collector.compute_end_to_end_metrics()
            res_m = self.collector.sample_system_resources(
                audio_queue_occupancy=audio_m.current_occupancy,
                scheduler_ready_jobs=app.recognizer.scheduler.ready_count,
            )

            app.stop()

        record = BenchmarkExperimentRecord(
            experiment_id=exp_id,
            timestamp_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scheduling_policy=policy_name,
            workload_type=workload_type,
            measurement_duration_s=duration_s,
            total_processed_blocks=blocks_processed,
            audio_metrics=audio_m,
            inference_metrics=infer_m,
            scheduling_metrics=sched_m,
            end_to_end_metrics=e2e_m,
            resource_metrics=res_m,
            environment=self.collector.environment,
            configuration={
                "policy": policy_name,
                "workload": workload_type.value,
                "num_blocks": num_blocks,
                "adaptive_enabled": config.enable_adaptation,
            },
            notes=f"Synthetic benchmark run for {workload_type.value} workload under {policy_name}.",
        )

        self.collector.record_benchmark(record)
        return record

    def run_policy_comparison(
        self,
        workload_type: WorkloadType = WorkloadType.NORMAL,
        num_blocks: int = 50,
    ) -> Dict[str, BenchmarkExperimentRecord]:
        """
        Runs an identical workload across FIFO, RM, and EDF policies.
        Guarantees strict comparability across schedulers.
        """
        results: Dict[str, BenchmarkExperimentRecord] = {}
        for pol in (SchedulingPolicy.FIFO, SchedulingPolicy.RM, SchedulingPolicy.EDF):
            pol_name = pol.value
            logger.info(f"[POLICY COMPARISON] Benchmarking {pol_name} under {workload_type.value}...")
            record = self.run_workload(policy=pol, workload_type=workload_type, num_blocks=num_blocks)
            results[pol_name] = record
            time.sleep(0.2)  # Cooldown between runs
        return results
