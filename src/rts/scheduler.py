"""
src/rts/scheduler.py
--------------------
Application-level Real-Time Scheduler supporting Baseline (FIFO), Rate Monotonic (RM),
and Earliest Deadline First (EDF) scheduling policies.

Key Concepts:
- Job Selection: Algorithmic choice of which pending job gets CPU execution next.
- Actual Execution: Cooperative execution of the selected job in Python user-space.
- Preemption Distinction: Python on standard Windows OS does not provide true kernel-level
  zero-overhead preemption. Schedulers operate at task boundaries (non-preemptive run-to-completion
  per job), scheduling between discrete tasks as they are submitted or complete.

Policies:
1. Baseline (FIFO):
   - Orders jobs strictly by release time (arrival order).
   - No priority or deadline awareness.
2. Rate Monotonic (RM):
   - Fixed priority assignment based on job period (T_i): Shorter period -> Higher priority.
   - For simultaneous releases, priority is evaluated first, tie-breaking on arrival time and job ID.
3. Earliest Deadline First (EDF):
   - Dynamic priority assignment based on absolute deadline (d_i = r_i + D_i).
   - Job with earliest absolute deadline executes first.
   - Tie-breaking on release time, then job ID.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import heapq
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.rts.tasks import RealTimeJob, JobStatus, JobType

logger = logging.getLogger("rtos.rts.scheduler")


class SchedulingPolicy(str, Enum):
    """Supported real-time scheduling policies."""
    FIFO = "FIFO"
    RM = "RM"
    EDF = "EDF"


@dataclass
class SchedulingDecision:
    """Audit record capturing an individual scheduling decision."""
    decision_index: int
    timestamp: float
    policy: SchedulingPolicy
    selected_job_id: str
    selected_job_name: str
    selected_job_type: str
    selected_job_priority: int
    selected_job_deadline: float
    ready_queue_size: int
    contending_job_ids: List[str]
    dispatch_overhead_us: float
    selection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "timestamp": round(self.timestamp, 6),
            "policy": self.policy.value,
            "selected_job_id": self.selected_job_id,
            "selected_job_name": self.selected_job_name,
            "selected_job_type": self.selected_job_type,
            "selected_job_priority": self.selected_job_priority,
            "selected_job_deadline": round(self.selected_job_deadline, 6),
            "ready_queue_size": self.ready_queue_size,
            "contending_job_ids": self.contending_job_ids,
            "dispatch_overhead_us": round(self.dispatch_overhead_us, 2),
            "selection_reason": self.selection_reason,
        }


class LateJobStrategy(str, Enum):
    """Policy for handling jobs whose absolute deadline has already passed upon dispatch."""
    EXECUTE_AND_RECORD = "EXECUTE_AND_RECORD"  # Execute job anyway, record deadline miss
    DROP_IF_EXPIRED = "DROP_IF_EXPIRED"        # Drop expired job, do not execute payload


class RealTimeScheduler:
    """
    Application-level real-time task scheduler.
    Supports FIFO, RM, and EDF job queueing and execution.
    """

    def __init__(
        self,
        policy: SchedulingPolicy = SchedulingPolicy.EDF,
        late_job_strategy: LateJobStrategy = LateJobStrategy.EXECUTE_AND_RECORD,
        max_queue_capacity: int = 200,
        on_job_completed: Optional[Callable[[RealTimeJob], None]] = None,
    ):
        self.policy = policy
        self.late_job_strategy = late_job_strategy
        self.max_queue_capacity = max_queue_capacity
        self.on_job_completed = on_job_completed
        self.on_task_exception: Optional[Callable[[RealTimeJob, Exception], None]] = None

        # Internal state (O(1) bounded queues to prevent telemetry memory leaks and GIL freezes)
        self._ready_jobs: List[RealTimeJob] = []
        self._completed_jobs: deque[RealTimeJob] = deque(maxlen=1000)
        self._dropped_jobs: deque[RealTimeJob] = deque(maxlen=500)
        self._decision_history: deque[SchedulingDecision] = deque(maxlen=1000)

        # Synchronization
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Counters & Telemetry
        self.total_submitted = 0
        self.total_executed = 0
        self.total_deadline_misses = 0
        self.total_dropped = 0
        self.total_queue_overflows = 0
        self.total_dispatch_overhead_us = 0.0

    @property
    def ready_count(self) -> int:
        with self._lock:
            return len(self._ready_jobs)

    def set_policy(self, policy: SchedulingPolicy):
        """Updates the active scheduling policy."""
        with self._lock:
            self.policy = policy
            logger.info(f"Scheduler policy changed to: {self.policy.value}")

    def submit_job(self, job: RealTimeJob) -> bool:
        """
        Submits a job to the ready queue.
        Returns True if accepted, False if dropped due to queue overflow.
        """
        with self._condition:
            if len(self._ready_jobs) >= self.max_queue_capacity:
                self.total_queue_overflows += 1
                job.mark_dropped(reason="Ready queue overflow capacity exceeded")
                self._dropped_jobs.append(job)
                self.total_dropped += 1
                logger.warning(f"Queue overflow! Dropped job {job.job_id} ({job.name})")
                return False

            job.status = JobStatus.READY
            self._ready_jobs.append(job)
            self.total_submitted += 1
            self._condition.notify()
            return True

    def _get_sort_key(self, job: RealTimeJob) -> Tuple:
        """
        Computes the priority comparison tuple based on the active scheduling policy.
        Lower values in the tuple indicate higher dispatch priority.
        """
        if self.policy == SchedulingPolicy.FIFO:
            # Baseline: Arrival/release time, then job ID
            return (job.release_time, job.job_id)

        elif self.policy == SchedulingPolicy.RM:
            # Rate Monotonic: Fixed priority (1=highest), then release time, then job ID
            return (job.fixed_priority, job.release_time, job.job_id)

        elif self.policy == SchedulingPolicy.EDF:
            # Earliest Deadline First: Absolute deadline, then release time, then job ID
            return (job.absolute_deadline, job.release_time, job.job_id)

        raise ValueError(f"Unknown scheduling policy: {self.policy}")

    def select_next_job(self) -> Optional[RealTimeJob]:
        """
        Inspects ready queue and selects the highest-priority job according to active policy.
        Records scheduling decision and measures selection overhead.
        """
        with self._lock:
            if not self._ready_jobs:
                return None

            t_start = time.perf_counter()

            # Sort ready jobs by policy sort key
            self._ready_jobs.sort(key=self._get_sort_key)
            selected_job = self._ready_jobs.pop(0)

            t_end = time.perf_counter()
            overhead_us = (t_end - t_start) * 1e6
            self.total_dispatch_overhead_us += overhead_us

            # Record reason
            if self.policy == SchedulingPolicy.FIFO:
                reason = f"FIFO arrival time: {selected_job.release_time:.6f}"
            elif self.policy == SchedulingPolicy.RM:
                reason = f"RM fixed priority: {selected_job.fixed_priority} (T={selected_job.period_ms}ms)"
            elif self.policy == SchedulingPolicy.EDF:
                reason = f"EDF absolute deadline: {selected_job.absolute_deadline:.6f} (D={selected_job.relative_deadline_ms}ms)"

            decision = SchedulingDecision(
                decision_index=len(self._decision_history) + 1,
                timestamp=t_start,
                policy=self.policy,
                selected_job_id=selected_job.job_id,
                selected_job_name=selected_job.name,
                selected_job_type=selected_job.job_type.value,
                selected_job_priority=selected_job.fixed_priority,
                selected_job_deadline=selected_job.absolute_deadline,
                ready_queue_size=len(self._ready_jobs) + 1,
                contending_job_ids=[j.job_id for j in self._ready_jobs[:5]],
                dispatch_overhead_us=overhead_us,
                selection_reason=reason,
            )
            self._decision_history.append(decision)

            return selected_job

    def execute_next_job(self, current_time: Optional[float] = None) -> Optional[RealTimeJob]:
        """
        Selects and executes the single highest-priority eligible job.
        Handles late/expired job strategies.
        """
        job = self.select_next_job()
        if job is None:
            return None

        now = time.perf_counter() if current_time is None else current_time

        # Check deadline expiration before execution
        if job.is_expired(now):
            if self.late_job_strategy == LateJobStrategy.DROP_IF_EXPIRED:
                job.mark_dropped(reason=f"Expired prior to dispatch (deadline {job.absolute_deadline:.4f} < now {now:.4f})")
                job.args = ()
                job.kwargs = {}
                with self._lock:
                    self._dropped_jobs.append(job)
                    self.total_dropped += 1
                    self.total_deadline_misses += 1
                logger.info(f"Late job dropped: {job.job_id} ({job.name})")
                if self.on_job_completed is not None:
                    try:
                        self.on_job_completed(job)
                    except Exception as e:
                        logger.error(f"Error in on_job_completed callback: {e}")
                return job

        # Execute payload with exception isolation
        try:
            job.execute()
        except Exception as exc:
            logger.error(f"[RTS SCHEDULER] Task execution exception in job {job.job_id} ({job.name}): {exc}")
            if self.on_task_exception is not None:
                try:
                    self.on_task_exception(job, exc)
                except Exception as cb_err:
                    logger.error(f"Error in on_task_exception callback: {cb_err}")
        finally:
            # Release large audio arrays from job payload to eliminate memory leak
            job.args = ()
            job.kwargs = {}

        with self._lock:
            self._completed_jobs.append(job)
            self.total_executed += 1
            if job.deadline_missed:
                self.total_deadline_misses += 1

        if self.on_job_completed is not None:
            try:
                self.on_job_completed(job)
            except Exception as e:
                logger.error(f"Error in on_job_completed callback: {e}")

        return job

    def run_all_pending(self) -> int:
        """Executes all currently queued jobs until ready queue is empty. Returns count executed."""
        executed = 0
        while self.ready_count > 0:
            job = self.execute_next_job()
            if job is not None:
                executed += 1
        return executed

    def start_async_worker(self):
        """Starts asynchronous background worker thread for live pipeline processing."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="RTSSchedulerWorker",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info(f"RTS Scheduler worker started (Policy: {self.policy.value})")

    def stop_async_worker(self, timeout: float = 1.0):
        """Stops asynchronous background worker cleanly."""
        with self._condition:
            if not self._is_running:
                return
            self._is_running = False
            self._condition.notify_all()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None
        logger.info("RTS Scheduler worker stopped.")

    def _worker_loop(self):
        """Continuous background dispatch loop."""
        while True:
            with self._condition:
                while self._is_running and len(self._ready_jobs) == 0:
                    self._condition.wait(timeout=0.05)
                if not self._is_running and len(self._ready_jobs) == 0:
                    break

            # Execute highest priority job with worker loop resilience
            try:
                self.execute_next_job()
            except Exception as loop_err:
                logger.error(f"[RTS SCHEDULER] Unhandled error in worker loop: {loop_err}")

    def cancel_job(self, job_id: str) -> bool:
        """Cancels a pending job by ID if still in ready queue."""
        with self._lock:
            for idx, job in enumerate(self._ready_jobs):
                if job.job_id == job_id:
                    self._ready_jobs.pop(idx)
                    job.mark_cancelled("Cancelled by scheduler request")
                    job.args = ()
                    job.kwargs = {}
                    self._dropped_jobs.append(job)
                    return True
        return False

    def clear(self):
        """Clears all internal queues and histories."""
        with self._lock:
            self._ready_jobs.clear()
            self._completed_jobs.clear()
            self._dropped_jobs.clear()
            self._decision_history.clear()
            self.total_submitted = 0
            self.total_executed = 0
            self.total_deadline_misses = 0
            self.total_dropped = 0
            self.total_queue_overflows = 0
            self.total_dispatch_overhead_us = 0.0

    def get_completed_jobs(self) -> List[RealTimeJob]:
        with self._lock:
            return list(self._completed_jobs)

    def get_dropped_jobs(self) -> List[RealTimeJob]:
        with self._lock:
            return list(self._dropped_jobs)

    def get_decision_history(self) -> List[SchedulingDecision]:
        with self._lock:
            return list(self._decision_history)

    def get_status_summary(self) -> Dict[str, Any]:
        """Returns overview of scheduler operational state."""
        with self._lock:
            mean_overhead = (self.total_dispatch_overhead_us / max(1, len(self._decision_history)))
            return {
                "policy": self.policy.value,
                "late_job_strategy": self.late_job_strategy.value,
                "ready_jobs_count": len(self._ready_jobs),
                "total_submitted": self.total_submitted,
                "total_executed": self.total_executed,
                "total_deadline_misses": self.total_deadline_misses,
                "total_dropped": self.total_dropped,
                "total_queue_overflows": self.total_queue_overflows,
                "decisions_recorded": len(self._decision_history),
                "mean_dispatch_overhead_us": round(mean_overhead, 2),
            }

