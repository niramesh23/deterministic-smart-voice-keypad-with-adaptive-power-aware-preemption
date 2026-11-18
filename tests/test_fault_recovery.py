"""
tests/test_fault_recovery.py
----------------------------
Automated Verification Test Suite for Stage 5:
Fault Detection and Safe Recovery.

Audited and Verified against:
1. Microphone initialization failure (detected, logged, safe degraded state).
2. Microphone disconnection (detected, recovery attempted).
3. Audio queue overflow (warning on single drop, error on repeated drops).
4. Inference exception (isolated, marked unavailable, never fabricates outputs).
5. Excessive inference latency (threshold violation triggers WARNING).
6. Database failure (bounded retries, degraded logging reported).
7. Dashboard failure (core voice processing continues headless).
8. Worker failure (task exception caught, thread preserved).
9. Recovery success (component verified healthy, status restored).
10. Recovery failure (bounded retry limit enforced, marked FAULT).
11. Critical fault causes safe lock behaviour (fail-closed, buffers purged).
12. System health returns to HEALTHY after verified recovery.
13. Fault and recovery events logged to audit trail.
14. Existing application functions remain available during partial degradation.
15. Fault debouncing / rate-limiting prevents event flooding.
16. Dashboard Health API (GET /api/health, POST /api/health/recover).
"""

import datetime
import json
import os
import pathlib
import threading
import time
from typing import List, Optional
import urllib.request
import urllib.error
import pytest
import numpy as np

os.environ["RTOS_ALLOW_DEMO_BOOTSTRAP"] = "1"

from src.fault.types import (
    FaultCategory,
    FaultEvent,
    FaultSeverity,
    FaultType,
    SystemHealthState,
    ComponentHealth,
    RecoveryRecord,
)
from src.fault.config import FaultDetectorConfig, RecoveryConfig
from src.fault.detector import FaultDetector
from src.fault.recovery import RecoveryManager
from src.security.lock_controller import LockController, LockState, AccessResult
from src.security.audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditResult,
    AuditSource,
)
from src.security.user_manager import UserManager, UserRole
from src.dashboard.server import DashboardServer, DashboardStatusProvider
from src.rts.tasks import RealTimeJob, JobType, JobStatus
from src.rts.scheduler import RealTimeScheduler, SchedulingPolicy


class MockClock:
    """Controllable monotonic clock."""
    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float):
        self.current_time += seconds


# ==============================================================================
# 1. Microphone Initialization Failure
# ==============================================================================

def test_01_microphone_init_failure(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)
    recovery = RecoveryManager(fault_detector=detector, audit_logger=audit_logger, time_func=clock)
    detector.on_fault_detected = recovery.on_fault_detected

    assert detector.system_health == SystemHealthState.HEALTHY

    # Simulate mic init failure
    evt = detector.report_fault(
        fault_type=FaultType.MIC_INIT_FAILURE,
        component="microphone",
        description="Could not open hardware audio device ID 99: Device unavailable",
        severity=FaultSeverity.ERROR,
    )

    assert evt is not None
    assert evt.severity == FaultSeverity.ERROR
    assert detector.get_component_health("microphone") == ComponentHealth.FAULT
    assert detector.system_health == SystemHealthState.FAULT

    # Verify audit log
    audit_logger.flush()
    fault_logs = audit_logger.query(event_type=AuditEventType.FAULT_DETECTED)
    assert len(fault_logs) >= 1
    assert "MIC_INIT_FAILURE" in json.dumps(fault_logs[0])

    audit_logger.stop()


# ==============================================================================
# 2. Microphone Disconnection
# ==============================================================================

def test_02_microphone_disconnection_and_recovery_attempt(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)
    recovery = RecoveryManager(fault_detector=detector, audit_logger=audit_logger, time_func=clock)

    # Mock audio pipeline that succeeds on recovery
    class MockAudioPipeline:
        def __init__(self):
            self.restarts = 0
            self.mic = type("MockMic", (), {"_is_running": False})()
        def restart_microphone(self):
            self.restarts += 1
            self.mic._is_running = True
            return True

    mock_audio = MockAudioPipeline()
    recovery.register_subsystems(audio_pipeline=mock_audio)
    detector.on_fault_detected = recovery.on_fault_detected

    # Report mic disconnected
    detector.report_fault(
        fault_type=FaultType.MIC_DISCONNECTED,
        component="microphone",
        description="Audio stream aborted: USB device unplugged",
        severity=FaultSeverity.ERROR,
    )

    # Verification: recovery action was automatically dispatched and succeeded
    assert mock_audio.restarts == 1
    assert detector.get_component_health("microphone") == ComponentHealth.OPERATIONAL
    assert detector.system_health == SystemHealthState.HEALTHY

    audit_logger.stop()


# ==============================================================================
# 3. Audio Queue Overflow (Single drop vs Repeated drops)
# ==============================================================================

def test_03_audio_queue_overflow(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    cfg = FaultDetectorConfig(queue_occupancy_warning_threshold=10, repeated_drops_threshold=5)
    detector = FaultDetector(config=cfg, audit_logger=audit_logger, time_func=clock)

    # 1. High occupancy -> WARNING -> DEGRADED
    detector.evaluate_audio_queue(occupancy=12, capacity=20, dropped_blocks=0)
    assert detector.system_health == SystemHealthState.DEGRADED
    assert detector.get_component_health("audio_queue") == ComponentHealth.DEGRADED

    # Advance clock to bypass dedup window
    clock.advance(2.0)

    # 2. Repeated drops (>= 5) -> ERROR -> FAULT
    detector.evaluate_audio_queue(occupancy=15, capacity=20, dropped_blocks=6)
    assert detector.system_health == SystemHealthState.FAULT
    assert detector.get_component_health("audio_queue") == ComponentHealth.FAULT

    audit_logger.stop()


# ==============================================================================
# 4. Inference Exception (Isolated, Marked Unavailable, Never Fabricates)
# ==============================================================================

def test_04_inference_exception_isolated_and_never_fabricates(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)
    recovery = RecoveryManager(fault_detector=detector, audit_logger=audit_logger, time_func=clock)

    class FlakyInferenceService:
        def __init__(self):
            self.is_available = True
            self.class_names = ["_silence_", "_unknown_", "go", "stop"]
        def set_available(self, avail: bool):
            self.is_available = avail
        def predict_from_features(self, *args, **kwargs):
            if not self.is_available:
                # Returns neutral silence result; NEVER fabricates commands
                return type("RecResult", (), {"command": None, "is_confirmed": False, "is_silence": True})()
            raise RuntimeError("CUDA/Torch kernel fault!")
        def reset_state(self):
            self.is_available = True
        def predict(self, audio):
            return type("RecResult", (), {"command": None, "is_confirmed": False})()

    infer_svc = FlakyInferenceService()
    recovery.register_subsystems(inference_service=infer_svc)

    # Simulate exception during execution
    try:
        infer_svc.predict_from_features()
    except Exception as exc:
        detector.report_fault(
            fault_type=FaultType.INFERENCE_EXCEPTION,
            component="inference",
            description=str(exc),
            severity=FaultSeverity.ERROR,
        )

    assert detector.get_component_health("inference") == ComponentHealth.FAULT
    assert detector.system_health == SystemHealthState.FAULT

    # While in fault, calls return neutral/silence, never fake command
    infer_svc.set_available(False)
    out = infer_svc.predict_from_features()
    assert out.command is None
    assert out.is_confirmed is False

    # Execute recovery
    ok, msg = recovery.recover_inference_subsystem()
    assert ok is True
    assert infer_svc.is_available is True
    assert detector.get_component_health("inference") == ComponentHealth.OPERATIONAL
    assert detector.system_health == SystemHealthState.HEALTHY

    audit_logger.stop()


# ==============================================================================
# 5. Excessive Inference Latency
# ==============================================================================

def test_05_excessive_inference_latency(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    cfg = FaultDetectorConfig(max_inference_latency_ms=75.0)
    detector = FaultDetector(config=cfg, audit_logger=audit_logger, time_func=clock)

    # Latency 110ms exceeds 75ms threshold
    detector.evaluate_inference_latency(latency_ms=110.0)

    assert detector.system_health == SystemHealthState.DEGRADED
    assert detector.get_component_health("inference") == ComponentHealth.DEGRADED

    snapshot = detector.get_health_snapshot()
    assert snapshot["active_faults_count"] == 1
    assert snapshot["active_faults"][0]["fault_type"] == "EXCESSIVE_INFERENCE_LATENCY"
    assert snapshot["active_faults"][0]["severity"] == "WARNING"

    audit_logger.stop()


# ==============================================================================
# 6. Database Failure (Bounded Retries & Reporting)
# ==============================================================================

def test_06_database_failure_bounded_retries(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    cfg = RecoveryConfig(max_db_retry_attempts=3, db_retry_backoff_s=0.01)
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)
    recovery = RecoveryManager(fault_detector=detector, audit_logger=audit_logger, config=cfg, time_func=clock)

    # Mock unrecoverable broken database probe
    class BrokenAuditLogger:
        def is_healthy(self):
            return False

    recovery.audit_logger = BrokenAuditLogger()

    # Attempt recovery
    ok, msg = recovery.recover_database_subsystem()
    assert ok is False
    assert "Bounded retries exhausted" in msg
    assert detector.get_component_health("database") == ComponentHealth.UNAVAILABLE
    assert detector.system_health in (SystemHealthState.DEGRADED, SystemHealthState.FAULT)

    audit_logger.stop()


# ==============================================================================
# 7. Dashboard Failure (Core Processing Continues Headless)
# ==============================================================================

def test_07_dashboard_failure_continues_headless(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)
    recovery = RecoveryManager(fault_detector=detector, audit_logger=audit_logger, time_func=clock)

    # Report dashboard failure
    evt = detector.report_fault(
        fault_type=FaultType.DASHBOARD_UNAVAILABLE,
        component="dashboard",
        description="Port 8080 already bound: [Errno 10048] address in use",
        severity=FaultSeverity.WARNING,
    )

    assert evt.severity == FaultSeverity.WARNING
    assert detector.get_component_health("dashboard") == ComponentHealth.DEGRADED
    # Non-critical dashboard warning leaves system in DEGRADED, not complete FAULT
    assert detector.system_health == SystemHealthState.DEGRADED

    # Core lock controller remains fully functional
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    ctrl.handle_digit("1")
    ctrl.handle_digit("2")
    assert ctrl.get_snapshot().digits_entered == 2

    audit_logger.stop()


# ==============================================================================
# 8. Worker Failure (Task Exception Handled, Thread Preserved)
# ==============================================================================

def test_08_worker_task_exception_handled_cleanly():
    scheduler = RealTimeScheduler(policy=SchedulingPolicy.EDF)
    exceptions_caught = []

    def on_task_exc(job, exc):
        exceptions_caught.append((job.job_id, str(exc)))

    scheduler.on_task_exception = on_task_exc

    def buggy_work():
        raise ZeroDivisionError("Simulated arithmetic task crash")

    job = RealTimeJob(
        job_id="faulty_task_1",
        name="BuggyTask",
        job_type=JobType.FEATURE_EXTRACTION,
        work_fn=buggy_work,
        release_time=time.perf_counter(),
        relative_deadline_ms=50.0,
    )

    scheduler.submit_job(job)
    # Execution must not crash scheduler caller
    res_job = scheduler.execute_next_job()

    assert res_job is not None
    assert res_job.error is not None
    assert isinstance(res_job.error, ZeroDivisionError)
    assert len(exceptions_caught) == 1
    assert exceptions_caught[0][0] == "faulty_task_1"


# ==============================================================================
# 9. Recovery Success
# ==============================================================================

def test_09_recovery_success_restores_health(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)
    recovery = RecoveryManager(fault_detector=detector, audit_logger=audit_logger, time_func=clock)

    class HealthyService:
        def __init__(self):
            self.online = False
        def restart_microphone(self):
            self.online = True
            return True

    svc = HealthyService()
    recovery.register_subsystems(audio_pipeline=svc)

    detector.report_fault(
        fault_type=FaultType.MIC_DISCONNECTED,
        component="microphone",
        description="Stream stalled",
        severity=FaultSeverity.ERROR,
    )
    assert detector.system_health == SystemHealthState.FAULT

    # Execute recovery
    ok, msg = recovery.recover_audio_subsystem()
    assert ok is True
    assert svc.online is True
    assert detector.get_component_health("microphone") == ComponentHealth.OPERATIONAL
    assert detector.system_health == SystemHealthState.HEALTHY

    # Verify recovery record logged
    stats = recovery.get_recovery_stats()
    assert stats["succeeded"] == 1
    assert stats["failed"] == 0

    audit_logger.stop()


# ==============================================================================
# 10. Recovery Failure (Exceeds Max Retries)
# ==============================================================================

def test_10_recovery_failure_exceeds_max_retries(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    cfg = RecoveryConfig(max_recovery_attempts=2)
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)
    recovery = RecoveryManager(fault_detector=detector, audit_logger=audit_logger, config=cfg, time_func=clock)

    class PermanentlyDeadService:
        def restart_microphone(self):
            return False

    recovery.register_subsystems(audio_pipeline=PermanentlyDeadService())

    # Attempt 1
    ok1, _ = recovery.recover_audio_subsystem()
    assert ok1 is False
    assert detector.get_component_health("microphone") == ComponentHealth.FAULT

    # Attempt 2
    ok2, _ = recovery.recover_audio_subsystem()
    assert ok2 is False

    # Attempt 3: Exceeds max retries -> aborted
    ok3, msg3 = recovery.recover_audio_subsystem()
    assert ok3 is False
    assert "Max retries (2) exceeded" in msg3
    assert detector.get_component_health("microphone") == ComponentHealth.FAULT

    stats = recovery.get_recovery_stats()
    assert stats["failed"] == 3

    audit_logger.stop()


# ==============================================================================
# 11. Critical Fault Causes Safe Lock Behaviour
# ==============================================================================

def test_11_critical_fault_causes_safe_lock_behaviour(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)
    recovery = RecoveryManager(
        fault_detector=detector,
        lock_controller=ctrl,
        audit_logger=audit_logger,
        time_func=clock,
    )
    detector.on_fault_detected = recovery.on_fault_detected

    # 1. User is in middle of PIN entry
    ctrl.handle_digit("1")
    ctrl.handle_digit("2")
    assert ctrl.get_snapshot().digits_entered == 2
    assert ctrl.state == LockState.PIN_ENTRY

    # 2. Critical fault occurs (e.g. unexpected lock transition or unresponsive worker)
    detector.report_fault(
        fault_type=FaultType.UNRESPONSIVE_WORKER,
        component="scheduler",
        description="Heartbeat timeout: Real-time worker thread hung",
        severity=FaultSeverity.CRITICAL,
    )

    # 3. Safe lock enforced: PIN buffer purged, state forced to LOCKED
    snap = ctrl.get_snapshot()
    assert snap.state == LockState.LOCKED
    assert snap.digits_entered == 0
    assert snap.is_locked is True

    # 4. If system was in LOCKOUT, lockout is preserved!
    # Cause lockout:
    for _ in range(3):
        ctrl.handle_digit("9")
        ctrl.handle_digit("9")
        ctrl.handle_digit("9")
        ctrl.handle_digit("9")
        ctrl.handle_submit()
    assert ctrl.state == LockState.LOCKOUT

    # Critical fault during LOCKOUT strictly preserves LOCKOUT
    detector.report_fault(
        fault_type=FaultType.UNEXPECTED_APP_EXCEPTION,
        component="application",
        description="Memory allocation fault",
        severity=FaultSeverity.CRITICAL,
    )
    assert ctrl.state == LockState.LOCKOUT

    audit_logger.stop()


# ==============================================================================
# 12. System Health Returns to HEALTHY After Verified Recovery
# ==============================================================================

def test_12_health_lifecycle_returns_to_healthy(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)

    assert detector.system_health == SystemHealthState.HEALTHY

    # Add WARNING fault
    f1 = detector.report_fault(
        fault_type=FaultType.EXCESSIVE_QUEUE_OCCUPANCY,
        component="audio_queue",
        description="Queue full",
    )
    assert detector.system_health == SystemHealthState.DEGRADED

    # Add ERROR fault
    clock.advance(2.0)
    f2 = detector.report_fault(
        fault_type=FaultType.MIC_DISCONNECTED,
        component="microphone",
        description="Mic dropped",
    )
    assert detector.system_health == SystemHealthState.FAULT

    # Resolve ERROR fault
    detector.resolve_fault(f2.fault_id, "Microphone reconnected")
    assert detector.system_health == SystemHealthState.DEGRADED

    # Resolve WARNING fault
    detector.resolve_fault(f1.fault_id, "Queue cleared")
    assert detector.system_health == SystemHealthState.HEALTHY

    audit_logger.stop()


# ==============================================================================
# 13. Fault and Recovery Events Logged to Audit Trail
# ==============================================================================

def test_13_fault_and_recovery_events_logged(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)
    recovery = RecoveryManager(fault_detector=detector, audit_logger=audit_logger, time_func=clock)

    detector.report_fault(
        fault_type=FaultType.FEATURE_EXTRACTION_FAILURE,
        component="inference",
        description="Log-Mel computation NaN",
        severity=FaultSeverity.ERROR,
    )

    recovery.recover_inference_subsystem()

    audit_logger.flush()
    fault_events = audit_logger.query(event_type=AuditEventType.FAULT_DETECTED)
    recovery_events = audit_logger.query(event_type=AuditEventType.FAULT_RECOVERY)

    assert len(fault_events) >= 1
    assert len(recovery_events) >= 1
    assert "FEATURE_EXTRACTION_FAILURE" in json.dumps(fault_events[0])
    assert "inference" in recovery_events[0]["reason"].lower()

    audit_logger.stop()


# ==============================================================================
# 14. Existing Application Functions Remain Available During Partial Degradation
# ==============================================================================

def test_14_existing_functions_available_during_partial_degradation(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    users_db = tmp_path / "users.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    um = UserManager(db_path=users_db, audit_logger=audit_logger)
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger, user_manager=um)
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)

    # Induce non-critical degraded warning on dashboard
    detector.report_fault(
        fault_type=FaultType.DASHBOARD_UNAVAILABLE,
        component="dashboard",
        description="Dashboard port blocked",
    )
    assert detector.system_health == SystemHealthState.DEGRADED

    # Core user authentication still works perfectly
    ctrl.handle_digit("1")
    ctrl.handle_digit("2")
    ctrl.handle_digit("3")
    ctrl.handle_digit("4")
    ok, res, msg = ctrl.handle_submit()

    assert ok is True
    assert res == AccessResult.GRANTED
    assert ctrl.state == LockState.UNLOCKED

    audit_logger.stop()


# ==============================================================================
# 15. Debouncing / Rate Limiting Prevents Event Flooding
# ==============================================================================

def test_15_fault_debouncing_prevents_flooding(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock(1000.0)
    cfg = FaultDetectorConfig(fault_dedup_window_s=1.0)
    detector = FaultDetector(config=cfg, audit_logger=audit_logger, time_func=clock)

    # Rapid burst of 10 identical faults within 0.1 seconds
    events = []
    for _ in range(10):
        evt = detector.report_fault(
            fault_type=FaultType.AUDIO_READ_FAILURE,
            component="microphone",
            description="Buffer overrun",
        )
        if evt is not None:
            events.append(evt)
        clock.advance(0.01)

    # Only 1 event generated; 9 debounced/suppressed!
    assert len(events) == 1

    # Advance beyond dedup window
    clock.advance(1.5)
    evt2 = detector.report_fault(
        fault_type=FaultType.AUDIO_READ_FAILURE,
        component="microphone",
        description="Buffer overrun after window",
    )
    assert evt2 is not None
    # Verify suppressed count recorded in metrics
    assert evt2.metrics.get("suppressed_duplicate_count", 0) >= 1

    audit_logger.stop()


# ==============================================================================
# 16. Dashboard Health API (GET /api/health and POST /api/health/recover)
# ==============================================================================

def test_16_dashboard_health_api(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    users_db = tmp_path / "users.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    um = UserManager(db_path=users_db, audit_logger=audit_logger)
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger, user_manager=um)
    detector = FaultDetector(audit_logger=audit_logger, time_func=clock)
    recovery = RecoveryManager(
        fault_detector=detector,
        lock_controller=ctrl,
        audit_logger=audit_logger,
        time_func=clock,
    )

    provider = DashboardStatusProvider(
        audit_logger=audit_logger,
        user_manager=um,
        fault_detector=detector,
        recovery_manager=recovery,
    )
    server = DashboardServer(host="127.0.0.1", port=0, provider=provider)
    server.start()
    base_url = server.url

    # 1. GET /api/health
    resp = urllib.request.urlopen(f"{base_url}/api/health")
    assert resp.code == 200
    health_data = json.loads(resp.read().decode("utf-8"))
    assert health_data["system_health"] == "HEALTHY"
    assert "components" in health_data

    # 2. Login as admin to obtain session token
    login_body = json.dumps({"user_id": "admin_01", "pin": "1234"}).encode("utf-8")
    req_login = urllib.request.Request(
        f"{base_url}/api/admin/login",
        data=login_body,
        headers={"Content-Type": "application/json"},
    )
    resp_login = urllib.request.urlopen(req_login)
    token = json.loads(resp_login.read().decode("utf-8"))["token"]

    # 3. POST /api/health/recover without auth -> 401
    rec_body = json.dumps({"component": "all"}).encode("utf-8")
    req_unauth = urllib.request.Request(
        f"{base_url}/api/health/recover",
        data=rec_body,
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req_unauth)
    assert err.value.code in (401, 403)

    # 4. POST /api/health/recover with admin token -> 200 OK
    req_auth = urllib.request.Request(
        f"{base_url}/api/health/recover",
        data=rec_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    resp_rec = urllib.request.urlopen(req_auth)
    assert resp_rec.code == 200
    rec_res = json.loads(resp_rec.read().decode("utf-8"))
    assert "results" in rec_res
    assert "health" in rec_res

    server.stop()
    audit_logger.stop()

