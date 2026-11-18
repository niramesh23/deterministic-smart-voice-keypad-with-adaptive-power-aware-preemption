"""
src/rts/monitor.py
------------------
Real-time performance monitoring, timing analysis, statistical distributions,
and diagnostic metrics collection.

Metric Definitions:
1. Execution Time (C_i):
   C_i = (t_completion - t_start) * 1000.0  [ms]
   Pure CPU duration spent executing the job's work function from dispatch to return.
2. Response Time (R_i):
   R_i = (t_completion - t_release) * 1000.0  [ms]
   Total turnaround duration from job arrival/release to final completion,
   including all queue delay and execution time.
3. Queue Delay (t_queue):
   t_queue = (t_start - t_release) * 1000.0  [ms]
   Time elapsed while the job waited in the ready queue before being dispatched.
4. Release Jitter (J_i):
   J_i = |t_release - t_expected_release| * 1000.0  [ms]
   Deviation between actual release timestamp and the expected periodic release instant:
   r_expected(k) = r_0 + k * (T_i / 1000.0).
5. Deadline Miss Ratio (DMR):
   DMR = (N_missed / N_total) * 100.0  [%]
   Fraction of executed/dropped jobs that finished after their absolute deadline d_i.
6. Scheduling Overhead (t_overhead):
   Duration consumed by the scheduler sorting and dispatch logic per decision, in microseconds (us).
7. Queue Overflows & Dropped Blocks:
   Count of jobs rejected when ready queue exceeds bounded capacity, and dropped late jobs.
8. CPU Usage (%):
   Process CPU utilization measured via time.process_time() relative to time.perf_counter().
"""

from dataclasses import dataclass, field
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.rts.tasks import RealTimeJob, JobType, JobStatus
from src.rts.scheduler import RealTimeScheduler, SchedulingDecision

logger = logging.getLogger("rtos.rts.monitor")


@dataclass
class DistributionStats:
    """Summary statistics for a continuous timing distribution."""
    count: int = 0
    min_val: float = 0.0
    mean_val: float = 0.0
    median_val: float = 0.0
    p95_val: float = 0.0
    p99_val: float = 0.0
    max_val: float = 0.0
    std_val: float = 0.0

    @classmethod
    def from_values(cls, values: List[float]) -> "DistributionStats":
        if not values:
            return cls()
        arr = np.array(values, dtype=np.float64)
        return cls(
            count=len(arr),
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


class RealTimeMonitor:
    """
    Collects, aggregates, and computes real-time timing statistics for scheduled jobs.
    """

    def __init__(self):
        self._start_perf_time = time.perf_counter()
        self._start_process_time = time.process_time()

    def reset_cpu_timer(self):
        """Resets the process CPU utilization baseline."""
        self._start_perf_time = time.perf_counter()
        self._start_process_time = time.process_time()

    def get_cpu_utilization(self) -> float:
        """
        Calculates process CPU utilization percentage since timer reset.
        Uses Python standard library time.process_time() vs time.perf_counter().
        Normalized to 100% per core.
        """
        elapsed_perf = max(1e-6, time.perf_counter() - self._start_perf_time)
        elapsed_process = max(0.0, time.process_time() - self._start_process_time)
        # Process CPU percentage across all active cores
        cpu_pct = (elapsed_process / elapsed_perf) * 100.0
        return round(cpu_pct, 2)

    def analyze_scheduler_run(
        self,
        scheduler: RealTimeScheduler,
        total_audio_blocks_ingested: int = 0,
        total_audio_blocks_dropped: int = 0,
    ) -> Dict[str, Any]:
        """
        Performs full statistical analysis across all completed and dropped jobs.
        """
        completed = scheduler.get_completed_jobs()
        dropped = scheduler.get_dropped_jobs()
        decisions = scheduler.get_decision_history()
        total_jobs = len(completed) + len(dropped)

        # Single-pass categorization across completed jobs to prevent GIL contention
        by_type_completed: Dict[JobType, List[RealTimeJob]] = {jt: [] for jt in JobType}
        exec_times = []
        resp_times = []
        queue_delays = []
        jitters = []
        missed_count = len(dropped)

        for j in completed:
            if j.deadline_missed:
                missed_count += 1
            if j.measured_execution_time_ms is not None:
                exec_times.append(j.measured_execution_time_ms)
            if j.response_time_ms is not None:
                resp_times.append(j.response_time_ms)
            if j.queue_delay_ms is not None:
                queue_delays.append(j.queue_delay_ms)
            if j.release_jitter_ms is not None and j.expected_release_time is not None:
                jitters.append(j.release_jitter_ms)
            if j.job_type in by_type_completed:
                by_type_completed[j.job_type].append(j)

        by_type_dropped: Dict[JobType, List[RealTimeJob]] = {jt: [] for jt in JobType}
        for j in dropped:
            if j.job_type in by_type_dropped:
                by_type_dropped[j.job_type].append(j)

        overheads = [d.dispatch_overhead_us for d in decisions]
        miss_ratio_pct = (missed_count / max(1, total_jobs)) * 100.0

        # Per-job-type breakdown
        job_type_breakdown: Dict[str, Any] = {}
        for jt in JobType:
            type_completed = by_type_completed[jt]
            type_dropped = by_type_dropped[jt]
            type_total = len(type_completed) + len(type_dropped)
            if type_total == 0:
                continue

            t_exec = [j.measured_execution_time_ms for j in type_completed if j.measured_execution_time_ms is not None]
            t_resp = [j.response_time_ms for j in type_completed if j.response_time_ms is not None]
            t_q = [j.queue_delay_ms for j in type_completed if j.queue_delay_ms is not None]
            t_jit = [j.release_jitter_ms for j in type_completed if j.release_jitter_ms is not None and j.expected_release_time is not None]

            t_missed = sum(1 for j in type_completed if j.deadline_missed) + len(type_dropped)

            job_type_breakdown[jt.value] = {
                "total_jobs": type_total,
                "completed": len(type_completed),
                "dropped": len(type_dropped),
                "deadline_misses": t_missed,
                "miss_ratio_pct": round((t_missed / max(1, type_total)) * 100.0, 2),
                "execution_time_ms": DistributionStats.from_values(t_exec).to_dict(),
                "response_time_ms": DistributionStats.from_values(t_resp).to_dict(),
                "queue_delay_ms": DistributionStats.from_values(t_q).to_dict(),
                "release_jitter_ms": DistributionStats.from_values(t_jit).to_dict() if t_jit else None,
            }

        cpu_usage_pct = self.get_cpu_utilization()

        return {
            "policy": scheduler.policy.value,
            "late_job_strategy": scheduler.late_job_strategy.value,
            "total_jobs_submitted": scheduler.total_submitted,
            "total_jobs_executed": len(completed),
            "total_jobs_dropped": len(dropped),
            "total_deadline_misses": missed_count,
            "deadline_miss_ratio_pct": round(miss_ratio_pct, 2),
            "total_queue_overflows": scheduler.total_queue_overflows,
            "audio_blocks_ingested": total_audio_blocks_ingested,
            "audio_blocks_dropped": total_audio_blocks_dropped,
            "cpu_utilization_pct": cpu_usage_pct,
            "global_distributions": {
                "execution_time_ms": DistributionStats.from_values(exec_times).to_dict(),
                "response_time_ms": DistributionStats.from_values(resp_times).to_dict(),
                "queue_delay_ms": DistributionStats.from_values(queue_delays).to_dict(),
                "release_jitter_ms": DistributionStats.from_values(jitters).to_dict() if jitters else None,
                "scheduling_overhead_us": DistributionStats.from_values(overheads).to_dict(),
            },
            "per_job_type_breakdown": job_type_breakdown,
        }

