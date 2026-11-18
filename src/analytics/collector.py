"""
src/analytics/collector.py
--------------------------
Central Performance Analytics Collector.

Tracks, computes, and serializes real-time performance metrics across:
- Audio buffering and ingestion
- Keyword inference and feature extraction
- Real-time task scheduling (FIFO, RM, EDF)
- Multi-stage end-to-end latency breakdowns
- Process and system resource utilization (via psutil)
- Structured benchmark experiment export (JSON and CSV)
"""

import csv
import datetime
import json
import logging
import os
import pathlib
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

from src.analytics.types import (
    AudioProcessingMetrics,
    BenchmarkEnvironment,
    BenchmarkExperimentRecord,
    DistributionStats,
    EndToEndApplicationMetrics,
    InferenceMetrics,
    RealTimeSchedulingMetrics,
    SystemResourceMetrics,
    WorkloadType,
)

logger = logging.getLogger("rtos.analytics.collector")


class PerformanceAnalyticsCollector:
    """
    Thread-safe performance monitoring and benchmark data collection engine.
    Collects live telemetry without disrupting soft real-time execution budgets.
    """

    def __init__(
        self,
        output_dir: str | pathlib.Path = "results/benchmarks",
        window_size: int = 1000,
        time_func: Optional[Any] = None,
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self._time_func = time_func or time.perf_counter
        self._lock = threading.RLock()

        # Environment metadata
        self.environment = BenchmarkEnvironment()

        # System resource handle (cached for minimal overhead)
        try:
            self._process = psutil.Process(os.getpid())
            # Prime CPU percent measurement
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
            self._psutil_available = True
        except Exception as e:
            logger.warning(f"[ANALYTICS] psutil initialization fallback: {e}")
            self._process = None
            self._psutil_available = False

        # Live Sliding Windows for Timing Metrics
        self._audio_proc_latencies: List[float] = []
        self._audio_queue_wait_times: List[float] = []
        self._buffer_switch_intervals: List[float] = []
        self._last_buffer_switch_time: Optional[float] = None

        self._inference_latencies: List[float] = []
        self._feature_extraction_latencies: List[float] = []
        self._inference_timestamps: List[float] = []
        self._total_inferences_count: int = 0
        self._collector_start_time = self._time_func()

        # Scheduling Metrics
        self._sched_execution_times: List[float] = []
        self._sched_response_times: List[float] = []
        self._sched_waiting_times: List[float] = []
        self._sched_overheads_us: List[float] = []
        self._sched_active_time_s: float = 0.0

        # End-to-End Metrics (5 distinct measurements)
        self._e2e_audio_to_infer: List[float] = []
        self._e2e_rec_to_conf: List[float] = []
        self._e2e_conf_to_pin: List[float] = []
        self._e2e_conf_to_lock: List[float] = []
        self._e2e_command_to_access: List[float] = []

        # High-performance caching for rapid dashboard polling
        self._cached_live_telemetry: Optional[Dict[str, Any]] = None
        self._last_telemetry_time: float = 0.0
        self._cached_system_resources: Optional[SystemResourceMetrics] = None
        self._last_resource_time: float = 0.0

        # Intermediate timestamp caches for multi-stage tracking
        self._pending_digit1_capture_time: Optional[float] = None
        self._pending_digit4_conf_time: Optional[float] = None

        # Historical benchmark storage
        self._benchmark_records: List[BenchmarkExperimentRecord] = []
        self._csv_filepath = self.output_dir / "benchmarks_summary.csv"
        self._init_csv_file()

    def _init_csv_file(self):
        """Initializes benchmark CSV header if file does not exist."""
        if not self._csv_filepath.exists():
            try:
                # Sample row to extract fieldnames
                dummy_row = BenchmarkExperimentRecord(
                    experiment_id="init",
                    timestamp_iso="init",
                    scheduling_policy="FIFO",
                    workload_type=WorkloadType.NORMAL,
                    measurement_duration_s=0.0,
                    total_processed_blocks=0,
                    audio_metrics=AudioProcessingMetrics(),
                    inference_metrics=InferenceMetrics(),
                    scheduling_metrics=RealTimeSchedulingMetrics(),
                    end_to_end_metrics=EndToEndApplicationMetrics(),
                    resource_metrics=SystemResourceMetrics(),
                    environment=self.environment,
                ).to_flat_csv_row()
                with open(self._csv_filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(dummy_row.keys()))
                    writer.writeheader()
            except Exception as e:
                logger.error(f"[ANALYTICS] Failed to initialize CSV header: {e}")

    # ==========================================================================
    # 1. Telemetry Ingestion Methods
    # ==========================================================================

    def record_audio_block_timing(
        self,
        capture_timestamp: float,
        enqueue_timestamp: Optional[float] = None,
        dequeue_timestamp: Optional[float] = None,
        framer_processed_timestamp: Optional[float] = None,
    ):
        """Records latency and waiting duration for an individual audio block."""
        with self._lock:
            now = self._time_func()
            t_proc_end = framer_processed_timestamp or now

            # Processing latency: capture to framer processed
            proc_lat = max(0.0, (t_proc_end - capture_timestamp) * 1000.0)
            self._audio_proc_latencies.append(proc_lat)
            if len(self._audio_proc_latencies) > self.window_size:
                self._audio_proc_latencies.pop(0)

            # Queue waiting time: enqueue to dequeue
            if enqueue_timestamp is not None and dequeue_timestamp is not None:
                wait_ms = max(0.0, (dequeue_timestamp - enqueue_timestamp) * 1000.0)
                self._audio_queue_wait_times.append(wait_ms)
                if len(self._audio_queue_wait_times) > self.window_size:
                    self._audio_queue_wait_times.pop(0)

    def record_buffer_switch(self, switch_time: Optional[float] = None):
        """Records ping-pong buffer alternation interval."""
        with self._lock:
            t = switch_time or self._time_func()
            if self._last_buffer_switch_time is not None:
                interval_ms = max(0.0, (t - self._last_buffer_switch_time) * 1000.0)
                self._buffer_switch_intervals.append(interval_ms)
                if len(self._buffer_switch_intervals) > self.window_size:
                    self._buffer_switch_intervals.pop(0)
            self._last_buffer_switch_time = t

    def record_feature_extraction(self, dsp_latency_ms: float):
        """Records DSP Log-Mel spectrogram computation latency."""
        with self._lock:
            self._feature_extraction_latencies.append(dsp_latency_ms)
            if len(self._feature_extraction_latencies) > self.window_size:
                self._feature_extraction_latencies.pop(0)

    def record_inference_execution(self, latency_ms: float):
        """Records neural network forward-pass latency."""
        with self._lock:
            now = self._time_func()
            self._inference_latencies.append(latency_ms)
            self._inference_timestamps.append(now)
            self._total_inferences_count += 1

            if len(self._inference_latencies) > self.window_size:
                self._inference_latencies.pop(0)
            if len(self._inference_timestamps) > self.window_size:
                self._inference_timestamps.pop(0)

    def record_scheduler_job_execution(
        self,
        execution_time_ms: float,
        response_time_ms: float,
        waiting_time_ms: float,
        overhead_us: Optional[float] = None,
    ):
        """Records real-time task dispatch timing."""
        with self._lock:
            self._sched_execution_times.append(execution_time_ms)
            self._sched_response_times.append(response_time_ms)
            self._sched_waiting_times.append(waiting_time_ms)
            self._sched_active_time_s += (execution_time_ms / 1000.0)

            if overhead_us is not None:
                self._sched_overheads_us.append(overhead_us)

            if len(self._sched_execution_times) > self.window_size:
                self._sched_execution_times.pop(0)
            if len(self._sched_response_times) > self.window_size:
                self._sched_response_times.pop(0)
            if len(self._sched_waiting_times) > self.window_size:
                self._sched_waiting_times.pop(0)
            if len(self._sched_overheads_us) > self.window_size:
                self._sched_overheads_us.pop(0)

    # ==========================================================================
    # 2. End-to-End Multi-Stage Latency Ingestion
    # ==========================================================================

    def record_e2e_audio_to_inference(self, capture_time: float, infer_done_time: float):
        """Stage 1: Audio capture to neural net inference completion."""
        with self._lock:
            delta_ms = max(0.0, (infer_done_time - capture_time) * 1000.0)
            self._e2e_audio_to_infer.append(delta_ms)
            if len(self._e2e_audio_to_infer) > self.window_size:
                self._e2e_audio_to_infer.pop(0)

    def record_e2e_recognition_to_confirmation(self, scored_time: float, confirmed_time: float):
        """Stage 2: Model prediction scoring to FSM command confirmation."""
        with self._lock:
            delta_ms = max(0.0, (confirmed_time - scored_time) * 1000.0)
            self._e2e_rec_to_conf.append(delta_ms)
            if len(self._e2e_rec_to_conf) > self.window_size:
                self._e2e_rec_to_conf.pop(0)

    def record_digit1_capture(self, capture_time: float):
        """Caches timestamp when the first PIN digit utterance was captured."""
        with self._lock:
            self._pending_digit1_capture_time = capture_time

    def record_digit4_confirmation(self, conf_time: float):
        """Caches timestamp when the 4th PIN digit was confirmed by FSM."""
        with self._lock:
            self._pending_digit4_conf_time = conf_time

    def record_e2e_pin_verification(self, verified_time: float):
        """Stage 3: 4th digit confirmation to cryptographic PIN verification completion."""
        with self._lock:
            if self._pending_digit4_conf_time is not None:
                delta_ms = max(0.0, (verified_time - self._pending_digit4_conf_time) * 1000.0)
                self._e2e_conf_to_pin.append(delta_ms)
                if len(self._e2e_conf_to_pin) > self.window_size:
                    self._e2e_conf_to_pin.pop(0)
                self._pending_digit4_conf_time = None

    def record_e2e_confirmation_to_lock(self, confirmed_time: float, lock_transition_time: float):
        """Stage 4: Confirmed command delivery to lock state decision completion."""
        with self._lock:
            delta_ms = max(0.0, (lock_transition_time - confirmed_time) * 1000.0)
            self._e2e_conf_to_lock.append(delta_ms)
            if len(self._e2e_conf_to_lock) > self.window_size:
                self._e2e_conf_to_lock.pop(0)

    def record_e2e_command_to_access(self, access_decision_time: float):
        """Stage 5: Full pipeline - First digit capture to final access granted/denied."""
        with self._lock:
            if self._pending_digit1_capture_time is not None:
                delta_ms = max(0.0, (access_decision_time - self._pending_digit1_capture_time) * 1000.0)
                self._e2e_command_to_access.append(delta_ms)
                if len(self._e2e_command_to_access) > self.window_size:
                    self._e2e_command_to_access.pop(0)
                self._pending_digit1_capture_time = None

    # ==========================================================================
    # 3. System Resource Sampling (psutil)
    # ==========================================================================

    def sample_system_resources(
        self,
        audio_queue_occupancy: int = 0,
        scheduler_ready_jobs: int = 0,
    ) -> SystemResourceMetrics:
        """
        Samples current CPU, memory, and backlog metrics.
        Clearly separates process-level vs system-level utilization.
        """
        now = self._time_func()
        if self._cached_system_resources is not None and (now - self._last_resource_time < 0.8):
            cached = self._cached_system_resources
            backlog = int(audio_queue_occupancy + scheduler_ready_jobs)
            return SystemResourceMetrics(
                process_cpu_pct=cached.process_cpu_pct,
                system_cpu_pct=cached.system_cpu_pct,
                process_memory_rss_mb=cached.process_memory_rss_mb,
                process_memory_vms_mb=cached.process_memory_vms_mb,
                active_worker_threads=cached.active_worker_threads,
                processing_backlog=backlog,
                measurement_method=cached.measurement_method,
            )

        proc_cpu = 0.0
        sys_cpu = 0.0
        rss_mb = 0.0
        vms_mb = 0.0
        num_threads = threading.active_count()

        if self._psutil_available and self._process is not None:
            try:
                # Process CPU % (relative to 1 core, normalized)
                proc_cpu = self._process.cpu_percent(interval=None)
                # System-wide overall CPU %
                sys_cpu = psutil.cpu_percent(interval=None)
                mem = self._process.memory_info()
                rss_mb = mem.rss / (1024.0 * 1024.0)
                vms_mb = mem.vms / (1024.0 * 1024.0)
                num_threads = self._process.num_threads()
            except Exception as e:
                logger.debug(f"[ANALYTICS] Resource sampling error: {e}")

        backlog = int(audio_queue_occupancy + scheduler_ready_jobs)

        metrics = SystemResourceMetrics(
            process_cpu_pct=proc_cpu,
            system_cpu_pct=sys_cpu,
            process_memory_rss_mb=rss_mb,
            process_memory_vms_mb=vms_mb,
            active_worker_threads=num_threads,
            processing_backlog=backlog,
            measurement_method="psutil",
        )
        self._cached_system_resources = metrics
        self._last_resource_time = now
        return metrics

    # ==========================================================================
    # 4. Snapshot & Aggregation Computation
    # ==========================================================================

    def compute_audio_metrics(
        self,
        audio_queue: Optional[Any] = None,
        dropped_count: int = 0,
    ) -> AudioProcessingMetrics:
        """Aggregates current audio pipeline distribution and queue stats."""
        with self._lock:
            q_metrics = audio_queue.get_metrics() if (audio_queue is not None and hasattr(audio_queue, "get_metrics")) else {}
            curr_occ = q_metrics.get("current_occupancy", 0)
            cap = q_metrics.get("capacity", 200)
            peak = q_metrics.get("peak_occupancy", 0)
            enqueued = q_metrics.get("total_enqueued", 0)
            dequeued = q_metrics.get("total_dequeued", 0)
            overflows = q_metrics.get("overflow_count", 0)
            drops = dropped_count or q_metrics.get("dropped_blocks_count", 0)
            util_pct = (curr_occ / max(1, cap)) * 100.0

            tot_enq = max(enqueued, len(self._audio_proc_latencies))
            tot_deq = max(dequeued, len(self._audio_proc_latencies))

            return AudioProcessingMetrics(
                processing_latency=DistributionStats.from_values(self._audio_proc_latencies),
                queue_waiting_time=DistributionStats.from_values(self._audio_queue_wait_times),
                buffer_switch_interval=DistributionStats.from_values(self._buffer_switch_intervals),
                current_occupancy=curr_occ,
                peak_occupancy=peak,
                queue_capacity=cap,
                queue_utilization_pct=round(util_pct, 2),
                total_enqueued=tot_enq,
                total_dequeued=tot_deq,
                frame_drop_count=drops,
                overflow_count=overflows,
            )

    def compute_inference_metrics(self) -> InferenceMetrics:
        """Aggregates neural network timing and throughput stats."""
        with self._lock:
            now = self._time_func()
            elapsed_s = max(1e-3, now - self._collector_start_time)

            # Recent throughput across window
            if len(self._inference_timestamps) >= 2:
                recent_elapsed = self._inference_timestamps[-1] - self._inference_timestamps[0]
                ips = (len(self._inference_timestamps) - 1) / max(1e-3, recent_elapsed)
            elif self._total_inferences_count > 0:
                ips = self._total_inferences_count / elapsed_s
            else:
                ips = 0.0

            return InferenceMetrics(
                inference_latency=DistributionStats.from_values(self._inference_latencies),
                feature_extraction_latency=DistributionStats.from_values(self._feature_extraction_latencies),
                total_inferences=self._total_inferences_count,
                throughput_ips=ips,
                active_duration_s=elapsed_s,
            )

    def compute_scheduling_metrics(self, scheduler: Optional[Any] = None) -> RealTimeSchedulingMetrics:
        """Aggregates real-time task scheduling metrics and per-task distributions."""
        with self._lock:
            now = self._time_func()
            window_s = max(1e-3, now - self._collector_start_time)
            util_pct = (self._sched_active_time_s / window_s) * 100.0

            policy_str = str(getattr(scheduler, "policy", "EDF"))
            if hasattr(scheduler, "policy") and hasattr(scheduler.policy, "value"):
                policy_str = scheduler.policy.value

            submitted = getattr(scheduler, "total_submitted", 0)
            executed = getattr(scheduler, "total_executed", len(self._sched_execution_times))
            dropped = getattr(scheduler, "total_dropped", 0)
            missed = getattr(scheduler, "total_deadline_misses", 0)
            total = max(1, executed + dropped)
            dmr = (missed / total) * 100.0

            # Job type breakdown if scheduler provides it
            breakdown = {}
            exec_times = list(self._sched_execution_times)
            resp_times = list(self._sched_response_times)
            wait_times = list(self._sched_waiting_times)

            if scheduler and hasattr(scheduler, "get_completed_jobs"):
                completed = scheduler.get_completed_jobs()
                if not exec_times:
                    exec_times = [j.measured_execution_time_ms for j in completed if j.measured_execution_time_ms is not None]
                    resp_times = [j.response_time_ms for j in completed if j.response_time_ms is not None]
                    wait_times = [j.queue_delay_ms for j in completed if j.queue_delay_ms is not None]

                for j in completed:
                    t_name = j.job_type.value if hasattr(j.job_type, "value") else str(j.job_type)
                    if t_name not in breakdown:
                        breakdown[t_name] = {"count": 0, "missed": 0, "durations": []}
                    breakdown[t_name]["count"] += 1
                    if j.deadline_missed:
                        breakdown[t_name]["missed"] += 1
                    if j.measured_execution_time_ms is not None:
                        breakdown[t_name]["durations"].append(j.measured_execution_time_ms)
                for k, v in breakdown.items():
                    durations = v.pop("durations")
                    v["distribution"] = DistributionStats.from_values(durations).to_dict()

            return RealTimeSchedulingMetrics(
                policy=policy_str,
                total_submitted=submitted,
                total_executed=executed,
                total_dropped=dropped,
                deadline_miss_count=missed,
                deadline_miss_ratio_pct=dmr,
                task_utilization_pct=min(100.0, util_pct),
                execution_time=DistributionStats.from_values(exec_times),
                response_time=DistributionStats.from_values(resp_times),
                waiting_time=DistributionStats.from_values(wait_times),
                scheduling_overhead_us=DistributionStats.from_values(self._sched_overheads_us),
                per_job_type=breakdown,
            )

    def compute_end_to_end_metrics(self) -> EndToEndApplicationMetrics:
        """Aggregates the 5 separately measured end-to-end latency stages."""
        with self._lock:
            return EndToEndApplicationMetrics(
                audio_to_inference=DistributionStats.from_values(self._e2e_audio_to_infer),
                recognition_to_confirmation=DistributionStats.from_values(self._e2e_rec_to_conf),
                confirmation_to_pin=DistributionStats.from_values(self._e2e_conf_to_pin),
                confirmation_to_lock=DistributionStats.from_values(self._e2e_conf_to_lock),
                command_to_access_decision=DistributionStats.from_values(self._e2e_command_to_access),
            )

    def get_live_telemetry(
        self,
        audio_queue: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        adaptive_controller: Optional[Any] = None,
        dropped_audio_blocks: int = 0,
    ) -> Dict[str, Any]:
        """
        Unified live snapshot pushed to dashboard and available via GET /api/analytics.
        Clearly differentiates live running metrics from benchmark test runs.
        """
        with self._lock:
            now = self._time_func()
            if self._cached_live_telemetry is not None and (now - self._last_telemetry_time < 0.4):
                return self._cached_live_telemetry

            q_occ = audio_queue.qsize() if audio_queue and hasattr(audio_queue, "qsize") else 0
            sched_ready = scheduler.ready_count if scheduler and hasattr(scheduler, "ready_count") else 0

            audio_m = self.compute_audio_metrics(audio_queue, dropped_count=dropped_audio_blocks)
            infer_m = self.compute_inference_metrics()
            sched_m = self.compute_scheduling_metrics(scheduler)
            e2e_m = self.compute_end_to_end_metrics()
            res_m = self.sample_system_resources(q_occ, sched_ready)

            ctrl_state = "DISABLED"
            if adaptive_controller and hasattr(adaptive_controller, "state"):
                ctrl_state = adaptive_controller.state.value

            telemetry = {
                "mode": "LIVE_MONITORING",
                "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "scheduling_policy": sched_m.policy,
                "adaptive_controller_state": ctrl_state,
                "audio": audio_m.to_dict(),
                "inference": infer_m.to_dict(),
                "scheduling": sched_m.to_dict(),
                "end_to_end": e2e_m.to_dict(),
                "resources": res_m.to_dict(),
                "environment": self.environment.to_dict(),
                "total_benchmarks_recorded": len(self._benchmark_records),
            }
            self._cached_live_telemetry = telemetry
            self._last_telemetry_time = now
            return telemetry

    # ==========================================================================
    # 5. Benchmark Experiment Logging (JSON & CSV Export)
    # ==========================================================================

    def record_benchmark(self, experiment: BenchmarkExperimentRecord):
        """
        Thread-safe persistence of a formal benchmark experiment run.
        Appends to internal records, writes standalone JSON file, and updates summary CSV.
        """
        with self._lock:
            self._benchmark_records.append(experiment)

            # 1. Write dedicated JSON artifact
            json_filename = f"benchmark_{experiment.experiment_id}.json"
            json_path = self.output_dir / json_filename
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(experiment.to_dict(), f, indent=2)
                logger.info(f"[ANALYTICS] Benchmark JSON exported: {json_path}")
            except Exception as e:
                logger.error(f"[ANALYTICS] Failed to write benchmark JSON: {e}")

            # 2. Append row to summary CSV
            try:
                row = experiment.to_flat_csv_row()
                with open(self._csv_filepath, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writerow(row)
                logger.info(f"[ANALYTICS] Benchmark row appended to CSV: {self._csv_filepath}")
            except Exception as e:
                logger.error(f"[ANALYTICS] Failed to append benchmark CSV row: {e}")

    def list_benchmarks(self) -> List[Dict[str, Any]]:
        """Returns list of summaries for all completed benchmarks."""
        with self._lock:
            return [b.to_dict() for b in self._benchmark_records]

    def reset_live_stats(self):
        """Resets sliding distributions (used between benchmark runs to ensure clean isolation)."""
        with self._lock:
            self._audio_proc_latencies.clear()
            self._audio_queue_wait_times.clear()
            self._buffer_switch_intervals.clear()
            self._last_buffer_switch_time = None
            self._inference_latencies.clear()
            self._feature_extraction_latencies.clear()
            self._inference_timestamps.clear()
            self._total_inferences_count = 0
            self._collector_start_time = self._time_func()

            self._sched_execution_times.clear()
            self._sched_response_times.clear()
            self._sched_waiting_times.clear()
            self._sched_overheads_us.clear()
            self._sched_active_time_s = 0.0

            self._e2e_audio_to_infer.clear()
            self._e2e_rec_to_conf.clear()
            self._e2e_conf_to_pin.clear()
            self._e2e_conf_to_lock.clear()
            self._e2e_command_to_access.clear()
            self._pending_digit1_capture_time = None
            self._pending_digit4_conf_time = None