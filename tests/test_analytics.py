"""
tests/test_analytics.py
-----------------------
Comprehensive verification suite for Stage 6: Real-Time Performance Analytics.

Tests:
1. Latency calculations (Audio processing latency, queue wait, buffer-switch timing).
2. Percentile calculations (Median/P50, P95, P99, Min, Mean, Max, Std on deterministic arrays).
3. Deadline miss calculations (Miss counts, deadline miss ratio percentage).
4. Queue utilization & buffer-switch timing.
5. Throughput calculations (Inferences per second over active time).
6. CPU and memory collection (psutil process-level vs system-level metrics).
7. Scheduler-specific measurements (FIFO vs RM vs EDF comparative execution).
8. Exported benchmark results (Structured JSON and summary CSV schema validation).
9. Repeatability of experiments (Multiple runs with identical workload).
10. Dashboard Analytics REST APIs (GET /api/analytics, authenticated POST /api/analytics/benchmark).
11. End-to-end multi-stage latency separation (5 distinct un-conflated metrics).
"""

import csv
import json
import os
import pathlib
import time
import urllib.request
import urllib.error
import numpy as np
import pytest

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
from src.analytics.collector import PerformanceAnalyticsCollector
from src.analytics.workloads import WorkloadRunner
from src.audio.queue import BoundedAudioQueue, QueueOverflowPolicy
from src.rts.scheduler import RealTimeScheduler, SchedulingPolicy
from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig
from src.dashboard.server import DashboardServer, DashboardStatusProvider
from src.security.lock_controller import LockController, LockState, AccessResult
from src.security.user_manager import UserManager, UserRole
from src.security.audit_logger import AuditLogger


# ==============================================================================
# 1. Latency Calculations & Timing Formulas
# ==============================================================================

def test_01_latency_and_timing_calculations(tmp_path: pathlib.Path):
    """
    Validates exact mathematical latency formulas:
    - Audio proc latency: (t_framer_processed - t_capture) * 1000.0
    - Audio wait time: (t_dequeue - t_enqueue) * 1000.0
    - Buffer switch interval: (t_switch[k] - t_switch[k-1]) * 1000.0
    """
    collector = PerformanceAnalyticsCollector(output_dir=tmp_path)

    # 1. Audio timing
    t_cap = 100.000
    t_enq = 100.002
    t_deq = 100.005  # wait = 3.0 ms
    t_proc = 100.010 # proc = 10.0 ms

    collector.record_audio_block_timing(
        capture_timestamp=t_cap,
        enqueue_timestamp=t_enq,
        dequeue_timestamp=t_deq,
        framer_processed_timestamp=t_proc,
    )

    audio_m = collector.compute_audio_metrics()
    assert audio_m.processing_latency.count == 1
    assert audio_m.processing_latency.mean_val == pytest.approx(10.0, abs=0.01)
    assert audio_m.queue_waiting_time.count == 1
    assert audio_m.queue_waiting_time.mean_val == pytest.approx(3.0, abs=0.01)

    # 2. Buffer switch interval
    collector.record_buffer_switch(switch_time=10.000)
    collector.record_buffer_switch(switch_time=10.020) # 20.0 ms interval
    collector.record_buffer_switch(switch_time=10.040) # 20.0 ms interval

    audio_m2 = collector.compute_audio_metrics()
    assert audio_m2.buffer_switch_interval.count == 2
    assert audio_m2.buffer_switch_interval.mean_val == pytest.approx(20.0, abs=0.01)


# ==============================================================================
# 2. Percentile Calculations on Deterministic Arrays
# ==============================================================================

def test_02_distribution_percentile_calculations():
    """
    Validates DistributionStats statistical percentile computations (P50, P95, P99, Min, Mean, Max, Std)
    using a deterministic synthetic distribution 1..100.
    """
    # Array from 1.0 to 100.0
    values = [float(i) for i in range(1, 101)]
    stats = DistributionStats.from_values(values)

    assert stats.count == 100
    assert stats.min_val == 1.0
    assert stats.max_val == 100.0
    assert stats.mean_val == pytest.approx(50.5, abs=0.1)
    assert stats.median_val == pytest.approx(50.5, abs=0.1)
    assert stats.p95_val == pytest.approx(95.05, abs=0.2)
    assert stats.p99_val == pytest.approx(99.01, abs=0.2)
    assert stats.std_val > 0.0

    # Empty array handling
    empty_stats = DistributionStats.from_values([])
    assert empty_stats.count == 0
    assert empty_stats.mean_val == 0.0
    assert empty_stats.p95_val == 0.0


# ==============================================================================
# 3. Deadline Miss Calculations
# ==============================================================================

def test_03_deadline_miss_calculations(tmp_path: pathlib.Path):
    """
    Validates deadline miss counting and Deadline Miss Ratio (DMR) formula:
    DMR = (N_missed / N_total) * 100.0
    """
    collector = PerformanceAnalyticsCollector(output_dir=tmp_path)

    # Mock scheduler
    class MockScheduler:
        def __init__(self):
            self.policy = SchedulingPolicy.EDF
            self.total_submitted = 100
            self.total_executed = 95
            self.total_dropped = 5
            self.total_deadline_misses = 20
            self.ready_count = 0

    mock_sched = MockScheduler()
    sched_m = collector.compute_scheduling_metrics(scheduler=mock_sched)

    assert sched_m.total_executed == 95
    assert sched_m.total_dropped == 5
    assert sched_m.deadline_miss_count == 20
    # Total evaluated jobs = executed (95) + dropped (5) = 100 -> DMR = 20.0%
    assert sched_m.deadline_miss_ratio_pct == pytest.approx(20.0, abs=0.01)


# ==============================================================================
# 4. Queue Utilization
# ==============================================================================

def test_04_queue_utilization_and_overflow(tmp_path: pathlib.Path):
    """
    Validates queue occupancy, peak occupancy, capacity, and utilization percentage.
    """
    collector = PerformanceAnalyticsCollector(output_dir=tmp_path)
    q = BoundedAudioQueue(capacity=10, overflow_policy=QueueOverflowPolicy.DROP_OLDEST)

    # Push 6 items
    class DummyBlock:
        def __init__(self, seq):
            self.sequence_number = seq
            self.capture_timestamp = time.perf_counter()
            self.enqueue_timestamp = time.perf_counter()

    for i in range(6):
        q.push(DummyBlock(i))

    audio_m = collector.compute_audio_metrics(audio_queue=q)
    assert audio_m.current_occupancy == 6
    assert audio_m.queue_capacity == 10
    assert audio_m.queue_utilization_pct == pytest.approx(60.0, abs=0.1)

    # Evict 2 items
    _ = q.pop(timeout=0)
    _ = q.pop(timeout=0)

    audio_m2 = collector.compute_audio_metrics(audio_queue=q)
    assert audio_m2.current_occupancy == 4
    assert audio_m2.queue_utilization_pct == pytest.approx(40.0, abs=0.1)


# ==============================================================================
# 5. Throughput Calculations
# ==============================================================================

def test_05_throughput_calculations(tmp_path: pathlib.Path):
    """
    Validates Inference Throughput formula:
    IPS = N_inferences / active_duration_s
    """
    # Deterministic clock
    clock_time = 1000.0
    def mock_time():
        return clock_time

    collector = PerformanceAnalyticsCollector(output_dir=tmp_path, time_func=mock_time)

    # Record 10 inferences spaced by 0.1s
    for i in range(10):
        clock_time += 0.10
        collector.record_inference_execution(latency_ms=12.5)

    inf_m = collector.compute_inference_metrics()
    assert inf_m.total_inferences == 10
    # 9 intervals over 0.9s = 10 inferences / 1.0s total -> ~10 IPS
    assert inf_m.throughput_ips == pytest.approx(10.0, abs=0.5)
    assert inf_m.inference_latency.count == 10
    assert inf_m.inference_latency.mean_val == pytest.approx(12.5, abs=0.01)


# ==============================================================================
# 6. System Resource Metrics via psutil
# ==============================================================================

def test_06_system_resource_metrics(tmp_path: pathlib.Path):
    """
    Validates cross-platform resource sampling via psutil:
    - Process CPU % and System CPU % are measured separately.
    - Process Memory RSS and VMS are positive and reported in MB.
    - Active worker thread count is reported.
    """
    collector = PerformanceAnalyticsCollector(output_dir=tmp_path)
    res = collector.sample_system_resources(audio_queue_occupancy=5, scheduler_ready_jobs=3)

    assert isinstance(res, SystemResourceMetrics)
    assert res.process_cpu_pct >= 0.0
    assert res.system_cpu_pct >= 0.0
    assert res.process_memory_rss_mb > 0.0  # Python process uses memory
    assert res.process_memory_vms_mb > 0.0
    assert res.active_worker_threads >= 1
    assert res.processing_backlog == 8      # 5 + 3
    assert res.measurement_method == "psutil"


# ==============================================================================
# 7. Scheduler Comparison (FIFO vs RM vs EDF)
# ==============================================================================

def test_07_scheduler_comparison_under_identical_workload(tmp_path: pathlib.Path):
    """
    Runs an identical workload across FIFO, RM, and EDF policies.
    Verifies that all three policies produce valid, comparable metrics without crashing.
    """
    runner = WorkloadRunner(output_dir=tmp_path)
    comparison = runner.run_policy_comparison(workload_type=WorkloadType.NORMAL, num_blocks=15)

    assert "FIFO" in comparison
    assert "RM" in comparison
    assert "EDF" in comparison

    for pol in ("FIFO", "RM", "EDF"):
        rec = comparison[pol]
        assert rec.scheduling_policy == pol
        assert rec.total_processed_blocks == 15
        assert rec.measurement_duration_s > 0.0
        assert rec.scheduling_metrics.execution_time.count > 0
        assert rec.audio_metrics.total_enqueued > 0


# ==============================================================================
# 8. Exported Benchmark Results (JSON and CSV validation)
# ==============================================================================

def test_08_benchmark_export_json_and_csv(tmp_path: pathlib.Path):
    """
    Validates export of formal benchmark results to standalone JSON files and summary CSV.
    """
    collector = PerformanceAnalyticsCollector(output_dir=tmp_path)

    # Create dummy benchmark record
    rec = BenchmarkExperimentRecord(
        experiment_id="test_exp_01",
        timestamp_iso="2026-11-05T00:00:00Z",
        scheduling_policy="EDF",
        workload_type=WorkloadType.NORMAL,
        measurement_duration_s=2.5,
        total_processed_blocks=50,
        audio_metrics=collector.compute_audio_metrics(),
        inference_metrics=collector.compute_inference_metrics(),
        scheduling_metrics=collector.compute_scheduling_metrics(),
        end_to_end_metrics=collector.compute_end_to_end_metrics(),
        resource_metrics=collector.sample_system_resources(),
        environment=collector.environment,
        notes="Automated schema test record",
    )

    collector.record_benchmark(rec)

    # 1. Verify JSON file
    json_path = tmp_path / "benchmark_test_exp_01.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["experiment_id"] == "test_exp_01"
    assert data["scheduling_policy"] == "EDF"
    assert data["workload_type"] == "NORMAL"
    assert "environment" in data
    assert "soft_realtime_notice" in data["environment"]

    # 2. Verify CSV file
    csv_path = tmp_path / "benchmarks_summary.csv"
    assert csv_path.exists()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "test_exp_01"
    assert rows[0]["policy"] == "EDF"
    assert rows[0]["workload"] == "NORMAL"


# ==============================================================================
# 9. Repeatability of Experiments
# ==============================================================================

def test_09_repeatability_of_experiments(tmp_path: pathlib.Path):
    """
    Executes two successive runs of the same workload under identical policy.
    Verifies metric structure repeatability and isolation between runs.
    """
    runner = WorkloadRunner(output_dir=tmp_path)

    run1 = runner.run_workload(policy=SchedulingPolicy.EDF, workload_type=WorkloadType.NORMAL, num_blocks=10, experiment_id_suffix="run1")
    run2 = runner.run_workload(policy=SchedulingPolicy.EDF, workload_type=WorkloadType.NORMAL, num_blocks=10, experiment_id_suffix="run2")

    assert run1.experiment_id != run2.experiment_id
    assert run1.total_processed_blocks == run2.total_processed_blocks == 10
    assert run1.scheduling_policy == run2.scheduling_policy == "EDF"
    assert run1.measurement_duration_s > 0.0
    assert run2.measurement_duration_s > 0.0


# ==============================================================================
# 10. Dashboard Analytics REST APIs
# ==============================================================================

def test_10_dashboard_analytics_api_endpoints(tmp_path: pathlib.Path):
    """
    Verifies Dashboard REST APIs for Real-Time Analytics:
    - GET /api/analytics returns live telemetry snapshot and benchmark list.
    - POST /api/analytics/benchmark without auth returns 401/403.
    - POST /api/analytics/benchmark with valid admin credentials triggers workload.
    """
    audit_db = tmp_path / "audit.db"
    users_db = tmp_path / "users.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    user_mgr = UserManager(db_path=str(users_db), audit_logger=audit_logger, allow_demo_bootstrap=False)
    if len(user_mgr.get_all_users()) == 0:
        user_mgr.bootstrap_initial_admin(admin_id="admin_01", username="Administrator", pin="1234")

    collector = PerformanceAnalyticsCollector(output_dir=tmp_path)
    runner = WorkloadRunner(collector=collector, output_dir=tmp_path)

    provider = DashboardStatusProvider(
        audit_logger=audit_logger,
        user_manager=user_mgr,
        analytics_collector=collector,
        workload_runner=runner,
    )
    server = DashboardServer(host="127.0.0.1", port=0, provider=provider)
    server.start()
    base_url = server.url

    try:
        # 1. GET /api/analytics
        resp = urllib.request.urlopen(f"{base_url}/api/analytics")
        assert resp.code == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "live" in data
        assert "benchmarks" in data
        assert data["live"]["mode"] == "LIVE_MONITORING"

        # 2. Login as admin
        login_body = json.dumps({"user_id": "admin_01", "pin": "1234"}).encode("utf-8")
        req_login = urllib.request.Request(
            f"{base_url}/api/admin/login",
            data=login_body,
            headers={"Content-Type": "application/json"},
        )
        resp_login = urllib.request.urlopen(req_login)
        token = json.loads(resp_login.read().decode("utf-8"))["token"]

        # 3. POST /api/analytics/benchmark without auth -> 401
        bench_body = json.dumps({"policy": "EDF", "workload": "NORMAL", "num_blocks": 5}).encode("utf-8")
        req_unauth = urllib.request.Request(
            f"{base_url}/api/analytics/benchmark",
            data=bench_body,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req_unauth)
        assert err.value.code in (401, 403)

        # 4. POST /api/analytics/benchmark with admin token -> 200 OK
        req_auth = urllib.request.Request(
            f"{base_url}/api/analytics/benchmark",
            data=bench_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        resp_bench = urllib.request.urlopen(req_auth)
        assert resp_bench.code == 200
        res_data = json.loads(resp_bench.read().decode("utf-8"))
        assert res_data["success"] is True
        assert "benchmark" in res_data
        assert res_data["benchmark"]["scheduling_policy"] == "EDF"

    finally:
        server.stop()
        audit_logger.stop()


# ==============================================================================
# 11. End-to-End Multi-Stage Latency Separation
# ==============================================================================

def test_11_end_to_end_latency_metric_separation(tmp_path: pathlib.Path):
    """
    Verifies that all 5 end-to-end stages are tracked separately and not conflated:
    1. Audio acquisition to inference completion
    2. Command recognition to command confirmation
    3. Command confirmation to PIN verification completion
    4. Command confirmation to lock-state decision
    5. Full command-to-access-decision latency
    """
    collector = PerformanceAnalyticsCollector(output_dir=tmp_path)

    # Simulate Stage 1
    collector.record_e2e_audio_to_inference(capture_time=10.000, infer_done_time=10.085) # 85ms

    # Simulate Stage 2
    collector.record_e2e_recognition_to_confirmation(scored_time=10.085, confirmed_time=10.090) # 5ms

    # Simulate Stage 3
    collector.record_digit4_confirmation(conf_time=10.090)
    collector.record_e2e_pin_verification(verified_time=10.093) # 3ms

    # Simulate Stage 4
    collector.record_e2e_confirmation_to_lock(confirmed_time=10.090, lock_transition_time=10.094) # 4ms

    # Simulate Stage 5
    collector.record_digit1_capture(capture_time=9.000)
    collector.record_e2e_command_to_access(access_decision_time=10.095) # 1095ms

    e2e = collector.compute_end_to_end_metrics()
    assert e2e.audio_to_inference.mean_val == pytest.approx(85.0, abs=0.1)
    assert e2e.recognition_to_confirmation.mean_val == pytest.approx(5.0, abs=0.1)
    assert e2e.confirmation_to_pin.mean_val == pytest.approx(3.0, abs=0.1)
    assert e2e.confirmation_to_lock.mean_val == pytest.approx(4.0, abs=0.1)
    assert e2e.command_to_access_decision.mean_val == pytest.approx(1095.0, abs=0.1)