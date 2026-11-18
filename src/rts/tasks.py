"""
src/rts/tasks.py
----------------
Real-time job abstractions, timing attributes, and lifecycle state management.

Job Model:
Each computational unit is modeled as an independent schedulable job J_i = (r_i, C_i, D_i, d_i, T_i, P_i):
- Release Time (r_i): The exact instant (monotonic timestamp) the job becomes ready for execution.
- Execution Time (C_i): Pure CPU time spent running the job payload from dispatch to completion.
- Relative Deadline (D_i): Maximum allowable time from release to completion (in milliseconds).
- Absolute Deadline (d_i): The hard instant d_i = r_i + (D_i / 1000.0) at which the job must be finished.
- Period (T_i): Nominal recurrence interval for periodic jobs (in milliseconds).
- Fixed Priority (P_i): Static priority for Rate Monotonic (RM) scheduling (1 = highest priority).
- Response Time (R_i): Total elapsed duration from release to completion: R_i = completion_time - release_time.
- Queue Delay (t_queue): Duration spent waiting in the ready queue: t_queue = start_time - release_time.
- Release Jitter (J_i): Deviation of actual release time from the expected strictly periodic release:
  J_i = |release_time - expected_release_time|.

Job Categories & Real-World Timing Justifications:
1. AUDIO_BLOCK (Period: 20 ms, Deadline: 20 ms, WCET: ~0.15 ms):
   - Justification: Input audio is captured in chunks of 320 samples at 16,000 Hz (= 20.0 ms).
     If an audio block is not processed (ping-pong transfer, framing, RMS energy, and VAD state update)
     within 20.0 ms, incoming samples will accumulate in buffers, inducing buffer overflow and audio loss.
     Hence, D = T = 20.0 ms is the strict operational stability deadline.
2. FEATURE_EXTRACTION (Period: Sporadic/40 ms, Deadline: 40 ms, WCET: ~0.5 ms):
   - Justification: Converts 1.0s context window into 40-bin Log-Mel spectrogram. Must complete
     with low latency to allow sufficient computational margin for subsequent neural network inference
     within the total end-to-end 100 ms speech recognition budget.
3. KEYWORD_INFERENCE (Period: Sporadic/100 ms, Deadline: 100 ms, WCET: ~3.5 ms):
   - Justification: Evaluates Residual DS-CNN forward pass, Softmax, EMA probability smoothing,
     and decision margin logic. The 100 ms deadline satisfies human conversational interaction bounds
     (where lag becomes perceptible at >150 ms) while providing ample headroom for CPU compute.
4. MONITORING (Period: 200 ms, Deadline: 200 ms, WCET: ~0.2 ms):
   - Justification: System telemetry, CPU tracking, queue health checks, and noise floor logging.
     Non-critical background job that should run regularly (5 Hz) but yield to all audio/ML tasks.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Dict, Optional, Tuple


class JobType(str, Enum):
    """Categorization of real-time pipeline tasks."""
    AUDIO_BLOCK = "AUDIO_BLOCK"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
    KEYWORD_INFERENCE = "KEYWORD_INFERENCE"
    MONITORING = "MONITORING"
    SYNTHETIC = "SYNTHETIC"


class JobStatus(str, Enum):
    """Discrete lifecycle states for a real-time job."""
    PENDING = "PENDING"          # Created but not yet submitted to the scheduler
    READY = "READY"              # Placed in the scheduler ready queue, waiting for CPU
    RUNNING = "RUNNING"          # Dispatched to a worker and currently executing
    COMPLETED = "COMPLETED"      # Finished successfully within deadline
    MISSED_DEADLINE = "MISSED_DEADLINE"  # Finished execution after absolute deadline expired
    DROPPED = "DROPPED"          # Dropped prior to execution due to deadline expiration/overload
    CANCELLED = "CANCELLED"      # Cancelled prior to execution by application request


@dataclass
class RealTimeJob:
    """
    Core schedulable unit in the application-level real-time system.
    """
    job_id: str
    name: str
    job_type: JobType
    work_fn: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # Real-Time Parameters
    period_ms: Optional[float] = None
    relative_deadline_ms: float = 20.0
    release_time: float = field(default_factory=time.perf_counter)
    expected_release_time: Optional[float] = None
    absolute_deadline: float = 0.0
    fixed_priority: int = 10  # Lower number = higher priority (for RM: 1=highest)

    # Measured Timing Attributes (Populated during lifecycle)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    measured_execution_time_ms: Optional[float] = None  # C_i
    response_time_ms: Optional[float] = None            # R_i
    queue_delay_ms: Optional[float] = None              # t_queue
    release_jitter_ms: Optional[float] = None           # J_i
    deadline_missed: bool = False

    # Status & Output
    status: JobStatus = JobStatus.READY
    result: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Calculate absolute deadline if not explicitly set
        if self.absolute_deadline <= 0.0:
            self.absolute_deadline = self.release_time + (self.relative_deadline_ms / 1000.0)

        # Calculate release jitter if expected release time was recorded
        if self.expected_release_time is not None:
            self.release_jitter_ms = abs(self.release_time - self.expected_release_time) * 1000.0
        else:
            self.release_jitter_ms = 0.0

        # Automatic RM fixed priority assignment if period is defined
        # Shorter period -> higher priority (lower integer)
        if self.period_ms is not None and self.fixed_priority == 10:
            if self.period_ms <= 20.0:
                self.fixed_priority = 1
            elif self.period_ms <= 50.0:
                self.fixed_priority = 2
            elif self.period_ms <= 100.0:
                self.fixed_priority = 3
            else:
                self.fixed_priority = 4

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Checks if current time has passed the job's absolute deadline."""
        now = time.perf_counter() if current_time is None else current_time
        return now > self.absolute_deadline

    def mark_started(self, start_time: Optional[float] = None):
        """Transitions job to RUNNING state and records dispatch timing."""
        now = time.perf_counter() if start_time is None else start_time
        self.start_time = now
        self.status = JobStatus.RUNNING
        self.queue_delay_ms = max(0.0, (now - self.release_time) * 1000.0)

    def mark_completed(self, result: Any = None, completion_time: Optional[float] = None):
        """Transitions job to COMPLETED (or MISSED_DEADLINE) and computes execution/response times."""
        now = time.perf_counter() if completion_time is None else completion_time
        self.completion_time = now
        self.result = result

        if self.start_time is not None:
            self.measured_execution_time_ms = max(0.0, (now - self.start_time) * 1000.0)
        else:
            self.measured_execution_time_ms = 0.0

        exec_ms = self.measured_execution_time_ms or 0.0
        self.response_time_ms = max(exec_ms, (now - self.release_time) * 1000.0)
        self.deadline_missed = now > self.absolute_deadline

        if self.deadline_missed:
            self.status = JobStatus.MISSED_DEADLINE
        else:
            self.status = JobStatus.COMPLETED

    def mark_dropped(self, reason: str = "Deadline expired before execution"):
        """Marks the job as dropped due to deadline overrun or queue overflow."""
        self.status = JobStatus.DROPPED
        self.deadline_missed = True
        self.metadata["drop_reason"] = reason

    def mark_cancelled(self, reason: str = "Job cancelled by application"):
        """Marks the job as cancelled prior to execution."""
        self.status = JobStatus.CANCELLED
        self.metadata["cancel_reason"] = reason

    def execute(self) -> Any:
        """Executes the job's work function with full error handling and timing recording."""
        self.mark_started()
        try:
            res = self.work_fn(*self.args, **self.kwargs)
            self.mark_completed(result=res)
            return res
        except Exception as exc:
            now = time.perf_counter()
            self.completion_time = now
            if self.start_time is not None:
                self.measured_execution_time_ms = (now - self.start_time) * 1000.0
            self.response_time_ms = (now - self.release_time) * 1000.0
            self.error = exc
            self.status = JobStatus.MISSED_DEADLINE if (now > self.absolute_deadline) else JobStatus.COMPLETED
            raise exc

    def to_dict(self) -> Dict[str, Any]:
        """Serializes job timing attributes to dictionary for JSON reports."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "period_ms": self.period_ms,
            "relative_deadline_ms": self.relative_deadline_ms,
            "release_time": round(self.release_time, 6),
            "expected_release_time": round(self.expected_release_time, 6) if self.expected_release_time else None,
            "absolute_deadline": round(self.absolute_deadline, 6),
            "fixed_priority": self.fixed_priority,
            "start_time": round(self.start_time, 6) if self.start_time else None,
            "completion_time": round(self.completion_time, 6) if self.completion_time else None,
            "measured_execution_time_ms": round(self.measured_execution_time_ms, 3) if self.measured_execution_time_ms is not None else None,
            "response_time_ms": round(self.response_time_ms, 3) if self.response_time_ms is not None else None,
            "queue_delay_ms": round(self.queue_delay_ms, 3) if self.queue_delay_ms is not None else None,
            "release_jitter_ms": round(self.release_jitter_ms, 3) if self.release_jitter_ms is not None else None,
            "deadline_missed": self.deadline_missed,
            "metadata": self.metadata,
        }


# ==============================================================================
# Job Factory Functions
# ==============================================================================

def create_audio_block_job(
    job_id: str,
    work_fn: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    release_time: Optional[float] = None,
    expected_release_time: Optional[float] = None,
) -> RealTimeJob:
    """Creates a high-frequency, periodic audio block processing job (T=20ms, D=20ms, P=1)."""
    now = time.perf_counter() if release_time is None else release_time
    return RealTimeJob(
        job_id=job_id,
        name=f"AudioBlock_{job_id}",
        job_type=JobType.AUDIO_BLOCK,
        work_fn=work_fn,
        args=args,
        kwargs=kwargs or {},
        period_ms=20.0,
        relative_deadline_ms=20.0,
        release_time=now,
        expected_release_time=expected_release_time,
        fixed_priority=1,
    )


def create_feature_extraction_job(
    job_id: str,
    work_fn: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    release_time: Optional[float] = None,
) -> RealTimeJob:
    """Creates a DSP feature extraction job (Sporadic/40ms, D=40ms, P=2)."""
    now = time.perf_counter() if release_time is None else release_time
    return RealTimeJob(
        job_id=job_id,
        name=f"FeatureExtraction_{job_id}",
        job_type=JobType.FEATURE_EXTRACTION,
        work_fn=work_fn,
        args=args,
        kwargs=kwargs or {},
        period_ms=40.0,
        relative_deadline_ms=40.0,
        release_time=now,
        fixed_priority=2,
    )


def create_keyword_inference_job(
    job_id: str,
    work_fn: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    release_time: Optional[float] = None,
) -> RealTimeJob:
    """Creates a deep learning keyword inference job (Sporadic/100ms, D=100ms, P=3)."""
    now = time.perf_counter() if release_time is None else release_time
    return RealTimeJob(
        job_id=job_id,
        name=f"KeywordInference_{job_id}",
        job_type=JobType.KEYWORD_INFERENCE,
        work_fn=work_fn,
        args=args,
        kwargs=kwargs or {},
        period_ms=100.0,
        relative_deadline_ms=100.0,
        release_time=now,
        fixed_priority=3,
    )


def create_monitoring_job(
    job_id: str,
    work_fn: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    release_time: Optional[float] = None,
    expected_release_time: Optional[float] = None,
) -> RealTimeJob:
    """Creates a background telemetry and monitoring job (T=200ms, D=200ms, P=4)."""
    now = time.perf_counter() if release_time is None else release_time
    return RealTimeJob(
        job_id=job_id,
        name=f"Monitoring_{job_id}",
        job_type=JobType.MONITORING,
        work_fn=work_fn,
        args=args,
        kwargs=kwargs or {},
        period_ms=200.0,
        relative_deadline_ms=200.0,
        release_time=now,
        expected_release_time=expected_release_time,
        fixed_priority=4,
    )

