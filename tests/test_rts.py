"""
tests/test_rts.py
-----------------
Unit and integration tests for Real-Time Scheduling (RTS) subsystem:
- Job abstraction, timestamps, deadlines, and state transitions
- Baseline FIFO ordering under contention
- Rate Monotonic (RM) fixed priority ordering and simultaneous release tie-breaking
- Earliest Deadline First (EDF) dynamic deadline ordering
- Late job handling strategies (EXECUTE_AND_RECORD vs DROP_IF_EXPIRED)
- Job cancellation and bounded queue overflow handling
- Scheduler decision logging and overhead instrumentation
- End-to-end integration of ScheduledKeywordRecognizer with real DSP & ML jobs
"""

import time
import numpy as np
import pytest

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
from src.application.scheduled_recognizer import ScheduledKeywordRecognizer


def dummy_work(duration_ms: float = 1.0, return_val: str = "ok"):
    """Simulates CPU work for a specified duration."""
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1000.0 < duration_ms:
        pass
    return return_val


def test_job_attributes_and_lifecycle():
    """Verifies RealTimeJob attributes, deadlines, and lifecycle state transitions."""
    t_rel = time.perf_counter()
    job = RealTimeJob(
        job_id="test_01",
        name="TestJob",
        job_type=JobType.SYNTHETIC,
        work_fn=dummy_work,
        args=(2.0, "done"),
        period_ms=20.0,
        relative_deadline_ms=20.0,
        release_time=t_rel,
    )

    assert job.status == JobStatus.READY
    assert job.period_ms == 20.0
    assert job.relative_deadline_ms == 20.0
    assert job.fixed_priority == 1  # 20ms period maps to highest priority 1
    assert pytest.approx(job.absolute_deadline, rel=1e-4) == t_rel + 0.020

    # Execute job
    res = job.execute()
    assert res == "done"
    assert job.status == JobStatus.COMPLETED
    assert job.measured_execution_time_ms is not None
    assert job.measured_execution_time_ms >= 1.5
    assert job.response_time_ms is not None
    assert job.deadline_missed is False


def test_baseline_fifo_ordering():
    """
    Verifies Baseline FIFO ordering:
    Jobs must execute strictly in order of arrival (release time),
    regardless of relative deadline or priority.
    """
    sched = RealTimeScheduler(policy=SchedulingPolicy.FIFO)

    t0 = time.perf_counter()
    # Submit Job 1: Long deadline, low priority, but arrived first (t0 + 0ms)
    j1 = RealTimeJob(
        job_id="J1",
        name="LongDeadline_ArrivedFirst",
        job_type=JobType.SYNTHETIC,
        work_fn=dummy_work,
        args=(0.5,),
        period_ms=200.0,
        relative_deadline_ms=200.0,
        release_time=t0,
        fixed_priority=4,
    )

    # Submit Job 2: Urgent deadline, high priority, but arrived second (t0 + 5ms)
    j2 = RealTimeJob(
        job_id="J2",
        name="UrgentDeadline_ArrivedSecond",
        job_type=JobType.SYNTHETIC,
        work_fn=dummy_work,
        args=(0.5,),
        period_ms=20.0,
        relative_deadline_ms=20.0,
        release_time=t0 + 0.005,
        fixed_priority=1,
    )

    sched.submit_job(j1)
    sched.submit_job(j2)

    # In FIFO, J1 must be selected first because it arrived earlier
    first = sched.select_next_job()
    second = sched.select_next_job()

    assert first.job_id == "J1"
    assert second.job_id == "J2"


def test_rate_monotonic_priority_ordering():
    """
    Verifies Rate Monotonic (RM) scheduling:
    Jobs with shorter periods (higher priority) must be selected before
    jobs with longer periods, regardless of arrival order.
    """
    sched = RealTimeScheduler(policy=SchedulingPolicy.RM)

    t0 = time.perf_counter()
    # J_monitor: T=200ms -> Priority 4 (Lowest), arrived earlier
    j_mon = RealTimeJob(
        job_id="J_mon",
        name="Monitor",
        job_type=JobType.MONITORING,
        work_fn=dummy_work,
        args=(0.2,),
        period_ms=200.0,
        relative_deadline_ms=200.0,
        release_time=t0,
        fixed_priority=4,
    )

    # J_infer: T=100ms -> Priority 3, arrived next
    j_inf = RealTimeJob(
        job_id="J_inf",
        name="Inference",
        job_type=JobType.KEYWORD_INFERENCE,
        work_fn=dummy_work,
        args=(0.2,),
        period_ms=100.0,
        relative_deadline_ms=100.0,
        release_time=t0 + 0.001,
        fixed_priority=3,
    )

    # J_audio: T=20ms -> Priority 1 (Highest), arrived last
    j_aud = RealTimeJob(
        job_id="J_aud",
        name="AudioBlock",
        job_type=JobType.AUDIO_BLOCK,
        work_fn=dummy_work,
        args=(0.2,),
        period_ms=20.0,
        relative_deadline_ms=20.0,
        release_time=t0 + 0.002,
        fixed_priority=1,
    )

    sched.submit_job(j_mon)
    sched.submit_job(j_inf)
    sched.submit_job(j_aud)

    # In RM, dispatch order must be: J_aud (P=1), then J_inf (P=3), then J_mon (P=4)
    sel1 = sched.select_next_job()
    sel2 = sched.select_next_job()
    sel3 = sched.select_next_job()

    assert sel1.job_id == "J_aud"
    assert sel2.job_id == "J_inf"
    assert sel3.job_id == "J_mon"


def test_rate_monotonic_simultaneous_release_tiebreaking():
    """
    Verifies RM behavior when jobs are released simultaneously:
    Priority takes precedence; if priorities match, arrival/job_id breaks ties.
    """
    sched = RealTimeScheduler(policy=SchedulingPolicy.RM)
    t_shared = time.perf_counter()

    j_aud1 = RealTimeJob(
        job_id="J_aud1",
        name="AudioBlock1",
        job_type=JobType.AUDIO_BLOCK,
        work_fn=dummy_work,
        period_ms=20.0,
        relative_deadline_ms=20.0,
        release_time=t_shared,
        fixed_priority=1,
    )
    j_aud2 = RealTimeJob(
        job_id="J_aud2",
        name="AudioBlock2",
        job_type=JobType.AUDIO_BLOCK,
        work_fn=dummy_work,
        period_ms=20.0,
        relative_deadline_ms=20.0,
        release_time=t_shared,
        fixed_priority=1,
    )
    j_mon = RealTimeJob(
        job_id="J_mon",
        name="Monitor",
        job_type=JobType.MONITORING,
        work_fn=dummy_work,
        period_ms=200.0,
        relative_deadline_ms=200.0,
        release_time=t_shared,
        fixed_priority=4,
    )

    sched.submit_job(j_mon)
    sched.submit_job(j_aud2)
    sched.submit_job(j_aud1)

    # Simultaneous: P=1 jobs selected before P=4
    # Between J_aud1 and J_aud2 (same priority and release), job_id breaks tie
    sel1 = sched.select_next_job()
    sel2 = sched.select_next_job()
    sel3 = sched.select_next_job()

    assert sel1.job_id == "J_aud1"
    assert sel2.job_id == "J_aud2"
    assert sel3.job_id == "J_mon"


def test_earliest_deadline_first_ordering():
    """
    Verifies Earliest Deadline First (EDF) scheduling:
    Jobs must be selected strictly by earliest absolute deadline d_i = r_i + D_i.
    """
    sched = RealTimeScheduler(policy=SchedulingPolicy.EDF)
    now = time.perf_counter()

    # J1: Released at now, D = 100ms -> d1 = now + 0.100s
    j1 = RealTimeJob(
        job_id="J1",
        name="Inference",
        job_type=JobType.KEYWORD_INFERENCE,
        work_fn=dummy_work,
        relative_deadline_ms=100.0,
        release_time=now,
    )

    # J2: Released at now, D = 20ms -> d2 = now + 0.020s
    j2 = RealTimeJob(
        job_id="J2",
        name="AudioBlock",
        job_type=JobType.AUDIO_BLOCK,
        work_fn=dummy_work,
        relative_deadline_ms=20.0,
        release_time=now,
    )

    # J3: Released at now, D = 40ms -> d3 = now + 0.040s
    j3 = RealTimeJob(
        job_id="J3",
        name="FeatureExtract",
        job_type=JobType.FEATURE_EXTRACTION,
        work_fn=dummy_work,
        relative_deadline_ms=40.0,
        release_time=now,
    )

    sched.submit_job(j1)
    sched.submit_job(j2)
    sched.submit_job(j3)

    # In EDF, order must be J2 (20ms) -> J3 (40ms) -> J1 (100ms)
    sel1 = sched.select_next_job()
    sel2 = sched.select_next_job()
    sel3 = sched.select_next_job()

    assert sel1.job_id == "J2"
    assert sel2.job_id == "J3"
    assert sel3.job_id == "J1"


def test_edf_dynamic_priority_ordering():
    """
    Demonstrates dynamic deadline ordering under EDF:
    An older job with originally long relative deadline can have an absolute deadline
    earlier than a newly arrived job with shorter relative deadline.
    """
    sched = RealTimeScheduler(policy=SchedulingPolicy.EDF)
    t0 = time.perf_counter()

    # J_old: released 80 ms ago with D=100ms -> absolute deadline is t0 + 20ms
    j_old = RealTimeJob(
        job_id="J_old",
        name="OlderInferenceJob",
        job_type=JobType.KEYWORD_INFERENCE,
        work_fn=dummy_work,
        relative_deadline_ms=100.0,
        release_time=t0 - 0.080,
    )

    # J_new: released right now with D=40ms -> absolute deadline is t0 + 40ms
    j_new = RealTimeJob(
        job_id="J_new",
        name="NewFeatureExtraction",
        job_type=JobType.FEATURE_EXTRACTION,
        work_fn=dummy_work,
        relative_deadline_ms=40.0,
        release_time=t0,
    )

    sched.submit_job(j_new)
    sched.submit_job(j_old)

    # Under EDF, J_old has absolute deadline (t0 + 0.020) which is earlier than J_new (t0 + 0.040)
    # So J_old must execute first despite being an inference job!
    first = sched.select_next_job()
    second = sched.select_next_job()

    assert first.job_id == "J_old"
    assert second.job_id == "J_new"


def test_late_job_strategies():
    """
    Verifies handling of late/expired jobs:
    1. EXECUTE_AND_RECORD: job runs and records MISSED_DEADLINE.
    2. DROP_IF_EXPIRED: job payload is dropped and marked DROPPED.
    """
    # Case 1: EXECUTE_AND_RECORD
    sched_exec = RealTimeScheduler(
        policy=SchedulingPolicy.EDF,
        late_job_strategy=LateJobStrategy.EXECUTE_AND_RECORD,
    )
    t_past = time.perf_counter() - 0.100  # released 100ms ago with 20ms deadline (expired 80ms ago)
    j_late1 = RealTimeJob(
        job_id="Late_1",
        name="LateAudio",
        job_type=JobType.AUDIO_BLOCK,
        work_fn=dummy_work,
        args=(0.1, "ran"),
        relative_deadline_ms=20.0,
        release_time=t_past,
    )
    sched_exec.submit_job(j_late1)
    res_job1 = sched_exec.execute_next_job()

    assert res_job1.status == JobStatus.MISSED_DEADLINE
    assert res_job1.deadline_missed is True
    assert res_job1.result == "ran"
    assert sched_exec.total_deadline_misses == 1

    # Case 2: DROP_IF_EXPIRED
    sched_drop = RealTimeScheduler(
        policy=SchedulingPolicy.EDF,
        late_job_strategy=LateJobStrategy.DROP_IF_EXPIRED,
    )
    j_late2 = RealTimeJob(
        job_id="Late_2",
        name="LateAudioDropped",
        job_type=JobType.AUDIO_BLOCK,
        work_fn=dummy_work,
        args=(0.1, "should_not_run"),
        relative_deadline_ms=20.0,
        release_time=t_past,
    )
    sched_drop.submit_job(j_late2)
    res_job2 = sched_drop.execute_next_job()

    assert res_job2.status == JobStatus.DROPPED
    assert res_job2.result is None  # Payload never executed
    assert sched_drop.total_dropped == 1
    assert sched_drop.total_deadline_misses == 1


def test_job_cancellation():
    """Verifies cancelling a job prior to execution."""
    sched = RealTimeScheduler(policy=SchedulingPolicy.EDF)
    j1 = RealTimeJob(
        job_id="CancelMe",
        name="ToCancel",
        job_type=JobType.SYNTHETIC,
        work_fn=dummy_work,
    )
    sched.submit_job(j1)
    assert sched.ready_count == 1

    cancelled = sched.cancel_job("CancelMe")
    assert cancelled is True
    assert sched.ready_count == 0
    assert j1.status == JobStatus.CANCELLED


def test_queue_overflow_drop():
    """Verifies bounded capacity prevents memory leaks and drops overflowing jobs cleanly."""
    sched = RealTimeScheduler(max_queue_capacity=5)
    for i in range(10):
        j = RealTimeJob(
            job_id=f"j_{i}",
            name=f"Job_{i}",
            job_type=JobType.SYNTHETIC,
            work_fn=dummy_work,
        )
        sched.submit_job(j)

    assert sched.ready_count == 5
    assert sched.total_dropped == 5
    assert sched.total_queue_overflows == 5


def test_decision_history_and_dispatch_overhead():
    """Verifies scheduler records decision history and measures dispatch overhead."""
    sched = RealTimeScheduler(policy=SchedulingPolicy.EDF)
    sched.submit_job(RealTimeJob(job_id="j1", name="J1", job_type=JobType.SYNTHETIC, work_fn=dummy_work))
    sched.submit_job(RealTimeJob(job_id="j2", name="J2", job_type=JobType.SYNTHETIC, work_fn=dummy_work))

    sched.execute_next_job()
    sched.execute_next_job()

    history = sched.get_decision_history()
    assert len(history) == 2
    for decision in history:
        assert isinstance(decision, SchedulingDecision)
        assert decision.dispatch_overhead_us >= 0.0
        assert decision.policy == SchedulingPolicy.EDF
        assert len(decision.selection_reason) > 0


def test_scheduled_recognizer_pipeline_with_audio():
    """
    Integration test: Runs audio stream through ScheduledKeywordRecognizer
    under EDF policy, verifying real AudioBlock, FeatureExtraction, KeywordInference,
    and Monitoring jobs are dispatched and tracked by the real-time scheduler.
    """
    recognizer = ScheduledKeywordRecognizer(
        policy=SchedulingPolicy.EDF,
        queue_capacity=50,
    )

    # Ingest 25 hops of synthetic audio (500 ms @ 16 kHz)
    hop_size = 320
    t_start = time.perf_counter()
    for seq in range(25):
        # Generate synthetic sine burst during hops 5..15 to trigger VAD
        if 5 <= seq <= 15:
            t = np.linspace(0, 0.02, hop_size, endpoint=False)
            sine = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
            block_data = sine
        else:
            block_data = np.zeros(hop_size, dtype=np.int16)

        recognizer.push_external_hop(block_data, capture_time=t_start + (seq * 0.020))

    # Run pending jobs synchronously in scheduler
    executed = recognizer.scheduler.run_all_pending()
    assert executed >= 25

    report = recognizer.get_analysis_report()
    assert report["policy"] == "EDF"
    assert report["total_jobs_executed"] >= 25
    assert "AUDIO_BLOCK" in report["per_job_type_breakdown"]
    assert report["global_distributions"]["execution_time_ms"]["count"] >= 25
    assert report["global_distributions"]["scheduling_overhead_us"]["count"] >= 25

