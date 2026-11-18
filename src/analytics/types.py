"""
src/analytics/types.py
----------------------
Core data structures, enums, and metric records for Real-Time Performance Analytics.

All timing metrics specify exact mathematical definitions and timestamps used.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np


class WorkloadType(str, Enum):
    """Categorization of test workloads for performance evaluation."""
    NORMAL = "NORMAL"          # Nominal 20ms hops, standard speech rate, no artificial load
    HIGH = "HIGH"              # Burst arrival, 2x frame rate, concurrent background queries
    OVERLOAD = "OVERLOAD"      # Synthetic compute delays, queue saturation, deadline stress
    RECOVERY = "RECOVERY"      # Fault injection and automated recovery overhead tracking


@dataclass
class DistributionStats:
    """
    Continuous distribution metrics for latency and duration measurements.
    All times in milliseconds unless explicitly noted.
    """
    count: int = 0
    min_val: float = 0.0
    mean_val: float = 0.0
    median_val: float = 0.0    # 50th percentile (P50)
    p95_val: float = 0.0       # 95th percentile
    p99_val: float = 0.0       # 99th percentile
    max_val: float = 0.0
    std_val: float = 0.0

    @classmethod
    def from_values(cls, values: List[float]) -> "DistributionStats":
        if not values:
            return cls()
        arr = np.array(values, dtype=np.float64)
        return cls(
            count=int(len(arr)),
            min_val=round(float(np.min(arr)), 3),
            mean_val=round(float(np.mean(arr)), 3),
            median_val=round(float(np.median(arr)), 3),
            p95_val=round(float(np.percentile(arr, 95)), 3),
            p99_val=round(float(np.percentile(arr, 99)), 3),
            max_val=round(float(np.max(arr)), 3),
            std_val=round(float(np.std(arr)), 3),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "min": self.min_val,
            "mean": self.mean_val,
            "median": self.median_val,
            "p95": self.p95_val,
            "p99": self.p99_val,
            "max": self.max_val,
            "std": self.std_val,
        }


@dataclass
class AudioProcessingMetrics:
    """
    Audio buffering, queueing, and ingestion timing metrics.
    
    Formulas:
    - processing_latency_ms: (t_framer_processed - t_capture) * 1000.0
    - queue_waiting_time_ms: (t_dequeue - t_enqueue) * 1000.0
    - queue_occupancy: Current elements in audio queue
    - queue_utilization_pct: (queue_occupancy / queue_capacity) * 100.0
    - buffer_switch_interval_ms: (t_switch[k] - t_switch[k-1]) * 1000.0
    """
    processing_latency: DistributionStats = field(default_factory=DistributionStats)
    queue_waiting_time: DistributionStats = field(default_factory=DistributionStats)
    buffer_switch_interval: DistributionStats = field(default_factory=DistributionStats)
    current_occupancy: int = 0
    peak_occupancy: int = 0
    queue_capacity: int = 200
    queue_utilization_pct: float = 0.0
    total_enqueued: int = 0
    total_dequeued: int = 0
    frame_drop_count: int = 0
    overflow_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processing_latency_ms": self.processing_latency.to_dict(),
            "queue_waiting_time_ms": self.queue_waiting_time.to_dict(),
            "buffer_switch_interval_ms": self.buffer_switch_interval.to_dict(),
            "current_occupancy": self.current_occupancy,
            "peak_occupancy": self.peak_occupancy,
            "queue_capacity": self.queue_capacity,
            "queue_utilization_pct": self.queue_utilization_pct,
            "total_enqueued": self.total_enqueued,
            "total_dequeued": self.total_dequeued,
            "frame_drop_count": self.frame_drop_count,
            "overflow_count": self.overflow_count,
        }


@dataclass
class InferenceMetrics:
    """
    ML model forward-pass and feature extraction timing metrics.
    
    Formulas:
    - inference_latency_ms: (t_infer_end - t_infer_start) * 1000.0
    - throughput_ips: total_inferences / active_duration_s
    """
    inference_latency: DistributionStats = field(default_factory=DistributionStats)
    feature_extraction_latency: DistributionStats = field(default_factory=DistributionStats)
    total_inferences: int = 0
    throughput_ips: float = 0.0
    active_duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inference_latency_ms": self.inference_latency.to_dict(),
            "feature_extraction_latency_ms": self.feature_extraction_latency.to_dict(),
            "total_inferences": self.total_inferences,
            "throughput_ips": round(self.throughput_ips, 2),
            "active_duration_s": round(self.active_duration_s, 3),
        }


@dataclass
class RealTimeSchedulingMetrics:
    """
    Task-level real-time scheduling performance metrics.
    
    Formulas:
    - execution_time_ms (C_i): (t_completion - t_start) * 1000.0
    - response_time_ms (R_i): (t_completion - t_release) * 1000.0
    - waiting_time_ms (W_i): (t_start - t_release) * 1000.0
    - deadline_miss_count: sum(1 for j where t_completion > d_i) + dropped
    - deadline_miss_ratio_pct (DMR): (deadline_miss_count / total_jobs) * 100.0
    - task_utilization_pct (U): (sum(C_i / 1000.0) / active_window_s) * 100.0
    - scheduling_overhead_us: Sorting and selection overhead in microseconds
    """
    policy: str = "EDF"
    total_submitted: int = 0
    total_executed: int = 0
    total_dropped: int = 0
    deadline_miss_count: int = 0
    deadline_miss_ratio_pct: float = 0.0
    task_utilization_pct: float = 0.0
    execution_time: DistributionStats = field(default_factory=DistributionStats)
    response_time: DistributionStats = field(default_factory=DistributionStats)
    waiting_time: DistributionStats = field(default_factory=DistributionStats)
    scheduling_overhead_us: DistributionStats = field(default_factory=DistributionStats)
    per_job_type: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "total_submitted": self.total_submitted,
            "total_executed": self.total_executed,
            "total_dropped": self.total_dropped,
            "deadline_miss_count": self.deadline_miss_count,
            "deadline_miss_ratio_pct": round(self.deadline_miss_ratio_pct, 2),
            "task_utilization_pct": round(self.task_utilization_pct, 2),
            "execution_time_ms": self.execution_time.to_dict(),
            "response_time_ms": self.response_time.to_dict(),
            "waiting_time_ms": self.waiting_time.to_dict(),
            "scheduling_overhead_us": self.scheduling_overhead_us.to_dict(),
            "per_job_type": self.per_job_type,
        }


@dataclass
class EndToEndApplicationMetrics:
    """
    End-to-end multi-stage pipeline latency metrics.
    All 5 metrics are measured separately and never conflated into an ambiguous number.
    
    Formulas:
    1. audio_to_inference_ms: (t_inference_done - t_capture) * 1000.0
    2. recognition_to_confirmation_ms: (t_fsm_confirmed - t_scored) * 1000.0
    3. confirmation_to_pin_ms: (t_pin_verified - t_digit4_confirmed) * 1000.0
    4. confirmation_to_lock_ms: (t_lock_transition - t_command_confirmed) * 1000.0
    5. command_to_access_decision_ms: (t_access_decision - t_first_digit_capture) * 1000.0
    """
    audio_to_inference: DistributionStats = field(default_factory=DistributionStats)
    recognition_to_confirmation: DistributionStats = field(default_factory=DistributionStats)
    confirmation_to_pin: DistributionStats = field(default_factory=DistributionStats)
    confirmation_to_lock: DistributionStats = field(default_factory=DistributionStats)
    command_to_access_decision: DistributionStats = field(default_factory=DistributionStats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "1_audio_to_inference_ms": self.audio_to_inference.to_dict(),
            "2_recognition_to_confirmation_ms": self.recognition_to_confirmation.to_dict(),
            "3_confirmation_to_pin_ms": self.confirmation_to_pin.to_dict(),
            "4_confirmation_to_lock_ms": self.confirmation_to_lock.to_dict(),
            "5_command_to_access_decision_ms": self.command_to_access_decision.to_dict(),
        }


@dataclass
class SystemResourceMetrics:
    """
    Host and process resource utilization metrics.
    Explicitly distinguishes between process-level and system-level metrics.
    """
    process_cpu_pct: float = 0.0          # Process CPU utilization (normalized to % per core)
    system_cpu_pct: float = 0.0           # Overall system CPU utilization across all cores
    process_memory_rss_mb: float = 0.0    # Resident Set Size in MB
    process_memory_vms_mb: float = 0.0    # Virtual Memory Size in MB
    active_worker_threads: int = 0        # Count of active threads in application process
    processing_backlog: int = 0           # Pending audio frames + ready scheduler jobs
    measurement_method: str = "psutil"    # Library/method used for resource measurement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_cpu_pct": round(self.process_cpu_pct, 2),
            "system_cpu_pct": round(self.system_cpu_pct, 2),
            "process_memory_rss_mb": round(self.process_memory_rss_mb, 2),
            "process_memory_vms_mb": round(self.process_memory_vms_mb, 2),
            "active_worker_threads": self.active_worker_threads,
            "processing_backlog": self.processing_backlog,
            "measurement_method": self.measurement_method,
        }


@dataclass
class BenchmarkEnvironment:
    """Snapshot of execution environment for reproducibility and comparison integrity."""
    os_name: str = field(default_factory=lambda: platform.system())
    os_release: str = field(default_factory=lambda: platform.release())
    os_version: str = field(default_factory=lambda: platform.version())
    machine: str = field(default_factory=lambda: platform.machine())
    processor: str = field(default_factory=lambda: platform.processor())
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    cpu_count_logical: int = field(default_factory=lambda: os.cpu_count() or 1)
    soft_realtime_notice: str = (
        "Soft real-time prototype on Windows/CPython. "
        "No hard real-time guarantees. Measurements subject to OS scheduler jitter."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "os_name": self.os_name,
            "os_release": self.os_release,
            "os_version": self.os_version,
            "machine": self.machine,
            "processor": self.processor,
            "python_version": self.python_version,
            "cpu_count_logical": self.cpu_count_logical,
            "soft_realtime_notice": self.soft_realtime_notice,
        }


@dataclass
class BenchmarkExperimentRecord:
    """
    Complete structured record of a repeatable performance benchmark run.
    Serialized to JSON and CSV.
    """
    experiment_id: str
    timestamp_iso: str
    scheduling_policy: str
    workload_type: WorkloadType
    measurement_duration_s: float
    total_processed_blocks: int
    audio_metrics: AudioProcessingMetrics
    inference_metrics: InferenceMetrics
    scheduling_metrics: RealTimeSchedulingMetrics
    end_to_end_metrics: EndToEndApplicationMetrics
    resource_metrics: SystemResourceMetrics
    environment: BenchmarkEnvironment
    configuration: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp_iso": self.timestamp_iso,
            "scheduling_policy": self.scheduling_policy,
            "workload_type": self.workload_type.value,
            "measurement_duration_s": round(self.measurement_duration_s, 3),
            "total_processed_blocks": self.total_processed_blocks,
            "audio": self.audio_metrics.to_dict(),
            "inference": self.inference_metrics.to_dict(),
            "scheduling": self.scheduling_metrics.to_dict(),
            "end_to_end": self.end_to_end_metrics.to_dict(),
            "resources": self.resource_metrics.to_dict(),
            "environment": self.environment.to_dict(),
            "configuration": self.configuration,
            "notes": self.notes,
        }

    def to_flat_csv_row(self) -> Dict[str, Any]:
        """Flattens primary metrics into a single-row CSV dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp_iso,
            "policy": self.scheduling_policy,
            "workload": self.workload_type.value,
            "duration_s": round(self.measurement_duration_s, 3),
            "blocks_processed": self.total_processed_blocks,
            # Audio
            "audio_proc_mean_ms": self.audio_metrics.processing_latency.mean_val,
            "audio_proc_p95_ms": self.audio_metrics.processing_latency.p95_val,
            "audio_queue_wait_mean_ms": self.audio_metrics.queue_waiting_time.mean_val,
            "audio_drops": self.audio_metrics.frame_drop_count,
            "audio_overflows": self.audio_metrics.overflow_count,
            # Inference
            "infer_mean_ms": self.inference_metrics.inference_latency.mean_val,
            "infer_p50_ms": self.inference_metrics.inference_latency.median_val,
            "infer_p95_ms": self.inference_metrics.inference_latency.p95_val,
            "infer_p99_ms": self.inference_metrics.inference_latency.p99_val,
            "infer_max_ms": self.inference_metrics.inference_latency.max_val,
            "throughput_ips": self.inference_metrics.throughput_ips,
            # Scheduling
            "sched_resp_mean_ms": self.scheduling_metrics.response_time.mean_val,
            "sched_resp_p95_ms": self.scheduling_metrics.response_time.p95_val,
            "sched_resp_p99_ms": self.scheduling_metrics.response_time.p99_val,
            "sched_miss_count": self.scheduling_metrics.deadline_miss_count,
            "sched_miss_ratio_pct": self.scheduling_metrics.deadline_miss_ratio_pct,
            "task_utilization_pct": self.scheduling_metrics.task_utilization_pct,
            "sched_overhead_mean_us": self.scheduling_metrics.scheduling_overhead_us.mean_val,
            # End-to-End
            "e2e_audio_to_infer_p95_ms": self.end_to_end_metrics.audio_to_inference.p95_val,
            "e2e_rec_to_conf_mean_ms": self.end_to_end_metrics.recognition_to_confirmation.mean_val,
            "e2e_conf_to_pin_mean_ms": self.end_to_end_metrics.confirmation_to_pin.mean_val,
            "e2e_conf_to_lock_mean_ms": self.end_to_end_metrics.confirmation_to_lock.mean_val,
            "e2e_command_to_access_mean_ms": self.end_to_end_metrics.command_to_access_decision.mean_val,
            # Resources
            "process_cpu_pct": self.resource_metrics.process_cpu_pct,
            "system_cpu_pct": self.resource_metrics.system_cpu_pct,
            "memory_rss_mb": self.resource_metrics.process_memory_rss_mb,
            "memory_vms_mb": self.resource_metrics.process_memory_vms_mb,
            "active_threads": self.resource_metrics.active_worker_threads,
            "backlog": self.resource_metrics.processing_backlog,
        }
