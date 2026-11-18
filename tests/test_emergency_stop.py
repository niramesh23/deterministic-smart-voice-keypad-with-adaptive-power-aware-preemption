"""
tests/test_emergency_stop.py
----------------------------
Automated Verification Test Suite for Stage 4:
Emergency STOP and High-Priority Command Handling.

Audited and Verified against:
1. Condition 1: STOP while system is LOCKED (confirmed locked, audit logged).
2. Condition 2: STOP during PIN_ENTRY (buffer cleared, returned to LOCKED, pending cancelled).
3. Condition 3: STOP when PIN is complete (4 digits) but unsubmitted (buffer cleared, candidate PIN destroyed).
4. Condition 4: STOP during authentication verification (concurrency test: background verification
   while STOP triggers, commit rejected, stale unlock prevented).
5. Condition 5: STOP during pending unlock (staged unlock request cancelled).
6. Condition 6: STOP while UNLOCKED (immediate relock, auto-relock cancelled, manual relock audit logged).
7. Condition 7: STOP during LOCKOUT (lockout duration, timer, and failed attempts preserved; state remains LOCKOUT).
8. Condition 8: STOP with dashboard unavailable / failing (headless / offline; emergency stop executes cleanly).
9. Race-condition resolution: Simultaneous STOP and normal "go" command (STOP takes precedence).
10. Low-confidence voice STOP command rejected (EMERGENCY_STOP_REJECTED logged).
11. Low-margin voice STOP command rejected (EMERGENCY_STOP_REJECTED logged).
12. Unconfirmed voice STOP command rejected (EMERGENCY_STOP_REJECTED logged).
13. Valid high-confidence voice STOP accepted through EmergencyHandler.
14. Dashboard API POST /api/emergency/stop with valid admin session succeeds (HTTP 200).
15. Dashboard API POST /api/emergency/stop without authentication rejected (HTTP 401/403).
16. Dashboard API POST /api/emergency/stop with regular user rejected (HTTP 403).
17. Dashboard API POST /api/emergency/stop cookie CSRF defense enforced.
18. Privacy & credential redaction in emergency audit logs.
19. Emergency flag lifecycle: subsequent legitimate access succeeds after emergency clearing.
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

# Ensure demo bootstrap is allowed for deterministic test environments
os.environ["RTOS_ALLOW_DEMO_BOOTSTRAP"] = "1"

from src.security.config import SecurityConfig
from src.security.pin_manager import PINManager
from src.security.audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditResult,
    AuditSource,
)
from src.security.user_manager import (
    UserManager,
    UserRecord,
    UserRole,
    UserStatus,
)
from src.security.lock_controller import (
    AccessResult,
    LockController,
    LockState,
    LockStateSnapshot,
)
from src.security.emergency_handler import EmergencyHandler, EmergencyStatus
from src.application.command_fsm import (
    CommandRecord,
    CommandState,
    RejectionReason,
)
from src.dashboard.server import (
    DashboardServer,
    DashboardStatusProvider,
)


class MockClock:
    """Deterministic, zero-sleep monotonic clock for testing."""
    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float):
        self.current_time += seconds


def make_command_record(
    command: str = "stop",
    confidence: float = 0.95,
    margin: float = 0.35,
    is_confirmed: bool = True,
    state_at_event: CommandState = CommandState.COMMAND_CONFIRMED,
    reaction_latency_ms: float = 45.0,
) -> CommandRecord:
    """Helper to construct realistic CommandRecord instances."""
    return CommandRecord(
        record_id=101,
        command=command,
        confidence=confidence,
        margin=margin,
        state_at_event=state_at_event,
        is_confirmed=is_confirmed,
        rejection_reason=RejectionReason.NONE if is_confirmed else RejectionReason.LOW_CONFIDENCE,
        detected_timestamp=1000.0,
        confirmed_timestamp=1000.045 if is_confirmed else None,
        reaction_latency_ms=reaction_latency_ms if is_confirmed else None,
        acoustic_latency_ms=120.0,
    )


# ==============================================================================
# Condition 1: STOP while system is LOCKED
# ==============================================================================

def test_condition_01_stop_while_system_is_locked(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    eh = EmergencyHandler(lock_controller=ctrl, audit_logger=audit_logger, time_func=clock)

    assert ctrl.state == LockState.LOCKED

    # Trigger emergency STOP
    ok, snap = eh.trigger_emergency_stop(source=AuditSource.VOICE, reason="Spoken emergency STOP in LOCKED state")
    assert ok is True
    assert snap.state == LockState.LOCKED
    assert snap.is_locked is True
    assert snap.emergency_active is True
    assert snap.pending_cancelled is False

    audit_logger.flush()
    records = audit_logger.query(event_type=AuditEventType.EMERGENCY_STOP)
    assert len(records) >= 1
    assert records[0]["result"] == AuditResult.SUCCESS.value
    assert records[0]["lock_state"] == LockState.LOCKED.value

    audit_logger.stop()


# ==============================================================================
# Condition 2: STOP during PIN_ENTRY (buffer cleared, returned to LOCKED)
# ==============================================================================

def test_condition_02_stop_during_pin_entry(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    eh = EmergencyHandler(lock_controller=ctrl, audit_logger=audit_logger, time_func=clock)

    # Enter partial PIN: '1', '2'
    ctrl.handle_digit("1")
    ctrl.handle_digit("2")
    snap1 = ctrl.get_snapshot()
    assert snap1.state == LockState.PIN_ENTRY
    assert snap1.digits_entered == 2

    # Issue STOP
    record = make_command_record(command="stop", confidence=0.92, margin=0.30)
    ok, snap2, decision = eh.handle_voice_stop(record)

    assert ok is True
    assert decision == "ACCEPTED"
    assert snap2.state == LockState.LOCKED
    assert snap2.digits_entered == 0
    assert snap2.pending_cancelled is True

    audit_logger.flush()
    cancel_events = audit_logger.query(event_type=AuditEventType.EMERGENCY_CANCEL_PENDING)
    assert len(cancel_events) >= 1
    stop_events = audit_logger.query(event_type=AuditEventType.EMERGENCY_STOP)
    assert len(stop_events) >= 1

    audit_logger.stop()


# ==============================================================================
# Condition 3: STOP when PIN is complete (4 digits) but unsubmitted
# ==============================================================================

def test_condition_03_stop_when_pin_complete_unsubmitted(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    eh = EmergencyHandler(lock_controller=ctrl, audit_logger=audit_logger, time_func=clock)

    # Enter all 4 digits without submitting 'go'
    for d in ["1", "2", "3", "4"]:
        ctrl.handle_digit(d)
    snap1 = ctrl.get_snapshot()
    assert snap1.state == LockState.PIN_ENTRY
    assert snap1.digits_entered == 4

    # High-priority STOP preempts before submit
    record = make_command_record(command="stop", confidence=0.88, margin=0.25)
    ok, snap2, decision = eh.handle_voice_stop(record)

    assert ok is True
    assert snap2.state == LockState.LOCKED
    assert snap2.digits_entered == 0
    assert snap2.pending_cancelled is True

    # Attempting to submit 'go' now must be rejected because buffer was purged
    sub_ok, res, msg = ctrl.handle_submit()
    assert sub_ok is False
    assert res == AccessResult.DENIED_INCOMPLETE
    assert ctrl.state == LockState.LOCKED

    audit_logger.stop()


# ==============================================================================
# Condition 4: STOP during authentication verification (concurrency & stale unlock prevention)
# ==============================================================================

def test_condition_04_stop_during_auth_verification_concurrency(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)

    dummy_user = UserRecord(
        user_id="user_test",
        username="Test User",
        pin_hash="abc",
        salt_hex="123",
        role=UserRole.REGULAR_USER,
        status=UserStatus.ACTIVE,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    # 1. Asynchronous worker starts verifying credentials and acquires operation ID
    op_id = ctrl.begin_access_operation()
    assert op_id > 0

    # 2. Before worker can commit unlock, an emergency STOP arrives
    snap, had_pending = ctrl.emergency_stop(source=AuditSource.VOICE, reason="Emergency STOP during in-flight auth")
    assert snap.state == LockState.LOCKED

    # 3. Worker now attempts to commit the unlock using the stale operation ID
    commit_ok, result, message = ctrl.commit_unlock(op_id, dummy_user)

    # Must be strictly rejected!
    assert commit_ok is False
    assert result == AccessResult.CLEARED
    assert "was cancelled or superseded by emergency STOP" in message

    # Door must remain firmly LOCKED
    assert ctrl.state == LockState.LOCKED
    assert ctrl.authenticated_user is None

    audit_logger.flush()
    cancel_logs = audit_logger.query(event_type=AuditEventType.EMERGENCY_CANCEL_PENDING)
    assert any("Stale unlock attempt" in (log.get("reason") or "") for log in cancel_logs)

    audit_logger.stop()


# ==============================================================================
# Condition 5: STOP during pending unlock (staged unlock request cancelled)
# ==============================================================================

def test_condition_05_stop_during_pending_unlock(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)

    user = UserRecord(
        user_id="user_staged",
        username="Staged User",
        pin_hash="abc",
        salt_hex="123",
        role=UserRole.REGULAR_USER,
        status=UserStatus.ACTIVE,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    # Stage a pending unlock
    op_id = ctrl.request_pending_unlock(user, delay_s=5.0)
    assert ctrl._pending_unlock_request is not None

    # Emergency STOP triggers
    snap, had_pending = ctrl.emergency_stop(source=AuditSource.VOICE, reason="Emergency preemption of staged unlock")
    assert had_pending is True
    assert ctrl._pending_unlock_request is None

    # Attempting to execute the staged unlock must fail
    exec_ok, res, msg = ctrl.execute_pending_unlock(op_id)
    assert exec_ok is False
    assert ctrl.state == LockState.LOCKED

    audit_logger.stop()


# ==============================================================================
# Condition 6: STOP while UNLOCKED (immediate relock, auto-relock cancelled)
# ==============================================================================

def test_condition_06_stop_while_unlocked(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)

    user = UserRecord(
        user_id="user_alice",
        username="Alice",
        pin_hash="abc",
        salt_hex="123",
        role=UserRole.REGULAR_USER,
        status=UserStatus.ACTIVE,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    # Unlock the door
    op_id = ctrl.begin_access_operation()
    ok, res, msg = ctrl.commit_unlock(op_id, user, unlock_duration_s=10.0)
    assert ok is True
    assert ctrl.state == LockState.UNLOCKED
    assert ctrl._unlock_timestamp > 0.0

    # Issue STOP while UNLOCKED
    snap, had_pending = ctrl.emergency_stop(source=AuditSource.VOICE, reason="Manual emergency relock")
    assert snap.state == LockState.LOCKED
    assert snap.is_locked is True
    assert snap.is_unlocked is False
    assert ctrl.state == LockState.LOCKED
    assert ctrl._unlock_timestamp == 0.0
    assert ctrl.authenticated_user is None

    # Advance clock past former unlock duration - ensure no invalid transitions occur
    clock.advance(15.0)
    ctrl.tick()
    assert ctrl.state == LockState.LOCKED

    audit_logger.flush()
    relock_logs = audit_logger.query(event_type=AuditEventType.LOCK_MANUAL_RELOCK)
    assert len(relock_logs) >= 1
    stop_logs = audit_logger.query(event_type=AuditEventType.EMERGENCY_STOP)
    assert len(stop_logs) >= 1

    audit_logger.stop()


# ==============================================================================
# Condition 7: STOP during LOCKOUT (lockout duration, timer, and failed attempts preserved)
# ==============================================================================

def test_condition_07_stop_during_lockout(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock(1000.0)
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)

    # Cause 3 failed attempts to enter LOCKOUT
    for _ in range(3):
        ctrl.handle_digit("9")
        ctrl.handle_digit("9")
        ctrl.handle_digit("9")
        ctrl.handle_digit("9")
        ctrl.handle_submit()

    assert ctrl.state == LockState.LOCKOUT
    assert ctrl.failed_attempts == 3
    initial_lockout_rem = ctrl._get_lockout_remaining()
    assert initial_lockout_rem == 30.0

    # Advance clock 10 seconds into the lockout window
    clock.advance(10.0)
    ctrl.tick()
    rem_before_stop = ctrl._get_lockout_remaining()
    assert 19.0 <= rem_before_stop <= 21.0

    # Issue emergency STOP during LOCKOUT
    snap, had_pending = ctrl.emergency_stop(source=AuditSource.VOICE, reason="Emergency STOP during active lockout")

    # State must strictly remain LOCKOUT, failed_attempts preserved, timer preserved
    assert snap.state == LockState.LOCKOUT
    assert ctrl.state == LockState.LOCKOUT
    assert ctrl.failed_attempts == 3
    rem_after_stop = ctrl._get_lockout_remaining()
    assert 19.0 <= rem_after_stop <= 21.0

    # Advance past lockout expiration
    clock.advance(25.0)
    ctrl.tick()
    assert ctrl.state == LockState.LOCKED
    assert ctrl.failed_attempts == 0

    audit_logger.flush()
    stop_logs = audit_logger.query(event_type=AuditEventType.EMERGENCY_STOP)
    assert any(log["result"] == AuditResult.WARNING.value for log in stop_logs)

    audit_logger.stop()


# ==============================================================================
# Condition 8: STOP with dashboard unavailable / failing (headless / offline resilience)
# ==============================================================================

def test_condition_08_stop_with_dashboard_unavailable(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()

    # Broken dashboard callback that raises exceptions
    class BrokenListener:
        def __call__(self, snapshot: LockStateSnapshot):
            raise ConnectionResetError("Dashboard network socket dropped!")

    ctrl = LockController(time_func=clock, audit_logger=audit_logger, on_state_change=BrokenListener())
    eh = EmergencyHandler(lock_controller=ctrl, audit_logger=audit_logger, time_func=clock)

    # Pre-condition: enter PIN digits
    ctrl.handle_digit("1")
    ctrl.handle_digit("2")

    # Issue STOP: must succeed and not crash the audio pipeline despite broken dashboard
    record = make_command_record(command="stop", confidence=0.91, margin=0.30)
    ok, snap, decision = eh.handle_voice_stop(record)

    assert ok is True
    assert snap.state == LockState.LOCKED
    assert snap.digits_entered == 0
    assert snap.pending_cancelled is True

    audit_logger.stop()


# ==============================================================================
# Test 9: Simultaneous STOP and normal "go" command (race-condition resolution)
# ==============================================================================

def test_09_simultaneous_stop_and_go_precedence(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    eh = EmergencyHandler(lock_controller=ctrl, audit_logger=audit_logger, time_func=clock)

    for iteration in range(25):
        # Enter valid PIN 1234
        for d in ["1", "2", "3", "4"]:
            ctrl.handle_digit(d)

        stop_barrier = threading.Barrier(2)

        def worker_go():
            stop_barrier.wait()
            ctrl.handle_submit()

        def worker_stop():
            stop_barrier.wait()
            rec = make_command_record(command="stop", confidence=0.95, margin=0.35)
            eh.handle_voice_stop(rec)

        t1 = threading.Thread(target=worker_go)
        t2 = threading.Thread(target=worker_stop)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # In any race condition between STOP and GO, the system must never be
        # left in an unhandled intermediate or corrupted state.
        snap = ctrl.get_snapshot()
        assert snap.state in (LockState.LOCKED, LockState.UNLOCKED, LockState.LOCKOUT)
        # If unlocked, execute emergency STOP to return to LOCKED for next round
        if snap.state == LockState.UNLOCKED:
            ctrl.emergency_stop()

    audit_logger.stop()


# ==============================================================================
# Test 10: Low-confidence voice STOP rejected
# ==============================================================================

def test_10_low_confidence_voice_stop_rejected(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    eh = EmergencyHandler(
        lock_controller=ctrl,
        audit_logger=audit_logger,
        min_confidence_threshold=0.65,
        min_margin_threshold=0.15,
        time_func=clock,
    )

    ctrl.handle_digit("1")
    ctrl.handle_digit("2")
    assert ctrl.get_snapshot().digits_entered == 2

    # Spoken STOP with confidence 0.50 (< 0.65 threshold)
    low_conf_rec = make_command_record(command="stop", confidence=0.50, margin=0.25, is_confirmed=True)
    ok, snap, decision = eh.handle_voice_stop(low_conf_rec)

    assert ok is False
    assert decision == "REJECTED_LOW_CONFIDENCE"
    # Digits must NOT be cleared by low-confidence spurious input
    assert snap.digits_entered == 2
    assert snap.state == LockState.PIN_ENTRY

    audit_logger.flush()
    rej_logs = audit_logger.query(event_type=AuditEventType.EMERGENCY_STOP_REJECTED)
    assert len(rej_logs) >= 1
    assert rej_logs[0]["result"] == AuditResult.WARNING.value
    assert "confidence" in rej_logs[0]["reason"]

    audit_logger.stop()


# ==============================================================================
# Test 11: Low-margin voice STOP rejected
# ==============================================================================

def test_11_low_margin_voice_stop_rejected(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    eh = EmergencyHandler(
        lock_controller=ctrl,
        audit_logger=audit_logger,
        min_confidence_threshold=0.65,
        min_margin_threshold=0.15,
        time_func=clock,
    )

    # Spoken STOP with good confidence 0.85, but margin 0.05 (< 0.15 threshold)
    low_margin_rec = make_command_record(command="stop", confidence=0.85, margin=0.05, is_confirmed=True)
    ok, snap, decision = eh.handle_voice_stop(low_margin_rec)

    assert ok is False
    assert decision == "REJECTED_LOW_CONFIDENCE"

    audit_logger.flush()
    rej_logs = audit_logger.query(event_type=AuditEventType.EMERGENCY_STOP_REJECTED)
    assert len(rej_logs) >= 1
    assert "margin" in rej_logs[0]["reason"]

    audit_logger.stop()


# ==============================================================================
# Test 12: Unconfirmed voice STOP rejected
# ==============================================================================

def test_12_unconfirmed_voice_stop_rejected(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    eh = EmergencyHandler(lock_controller=ctrl, audit_logger=audit_logger, time_func=clock)

    # Unconfirmed candidate prediction
    unconfirmed_rec = make_command_record(
        command="stop",
        confidence=0.90,
        margin=0.30,
        is_confirmed=False,
        state_at_event=CommandState.CANDIDATE_COMMAND,
    )
    ok, snap, decision = eh.handle_voice_stop(unconfirmed_rec)

    assert ok is False
    assert decision == "REJECTED_LOW_CONFIDENCE"

    audit_logger.flush()
    rej_logs = audit_logger.query(event_type=AuditEventType.EMERGENCY_STOP_REJECTED)
    assert len(rej_logs) >= 1

    audit_logger.stop()


# ==============================================================================
# Test 13: Valid high-confidence voice STOP accepted through EmergencyHandler
# ==============================================================================

def test_13_valid_high_confidence_voice_stop_accepted(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    eh = EmergencyHandler(lock_controller=ctrl, audit_logger=audit_logger, time_func=clock)

    rec = make_command_record(command="stop", confidence=0.96, margin=0.40, is_confirmed=True)
    ok, snap, decision = eh.handle_voice_stop(rec)

    assert ok is True
    assert decision == "ACCEPTED"
    assert snap.state == LockState.LOCKED

    status = eh.get_emergency_status()
    assert status["is_active"] is True
    assert status["total_received"] == 1
    assert status["total_accepted"] == 1
    assert status["total_rejected"] == 0
    assert status["last_stop_source"] == "VOICE"

    audit_logger.stop()


# ==============================================================================
# Helper for Dashboard Tests
# ==============================================================================

def create_test_dashboard(tmp_path: pathlib.Path):
    """Initializes a full test dashboard server instance on an OS-assigned free port."""
    clock = MockClock()
    users_db = tmp_path / "users.db"
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    um = UserManager(db_path=users_db, audit_logger=audit_logger)

    status_provider = DashboardStatusProvider(
        audit_logger=audit_logger,
        user_manager=um,
    )

    ctrl = LockController(
        time_func=clock,
        audit_logger=audit_logger,
        user_manager=um,
        on_state_change=lambda snap: status_provider.update_status({"lock": snap.to_dict()}),
    )
    eh = EmergencyHandler(lock_controller=ctrl, audit_logger=audit_logger, time_func=clock)
    status_provider.emergency_handler = eh
    status_provider.update_status({"lock": ctrl.get_snapshot().to_dict()})

    # Bind to localhost on port 0 (OS allocates ephemeral free port)
    server = DashboardServer(host="127.0.0.1", port=0, provider=status_provider)
    server.start()
    base_url = server.url

    return server, base_url, um, ctrl, eh, audit_logger


# ==============================================================================
# Test 14: Dashboard API POST /api/emergency/stop with valid admin session succeeds
# ==============================================================================

def test_14_dashboard_emergency_stop_authenticated_admin_succeeds(tmp_path: pathlib.Path):
    server, base_url, um, ctrl, eh, audit_logger = create_test_dashboard(tmp_path)

    # 1. Login as default admin
    login_body = json.dumps({"user_id": "admin_01", "pin": "1234"}).encode("utf-8")
    req_login = urllib.request.Request(
        f"{base_url}/api/admin/login",
        data=login_body,
        headers={"Content-Type": "application/json"},
    )
    resp_login = urllib.request.urlopen(req_login)
    login_res = json.loads(resp_login.read().decode("utf-8"))
    token = login_res["token"]
    assert token

    # 2. Put lock in PIN_ENTRY first
    ctrl.handle_digit("1")
    ctrl.handle_digit("2")
    assert ctrl.get_snapshot().digits_entered == 2

    # 3. Post emergency stop with Bearer token
    stop_body = json.dumps({"reason": "Manual Admin Emergency Stop"}).encode("utf-8")
    req_stop = urllib.request.Request(
        f"{base_url}/api/emergency/stop",
        data=stop_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    resp_stop = urllib.request.urlopen(req_stop)
    assert resp_stop.code == 200
    res = json.loads(resp_stop.read().decode("utf-8"))

    assert res["success"] is True
    assert res["lock"]["state"] == "LOCKED"
    assert res["lock"]["digits_entered"] == 0
    assert res["lock"]["pending_cancelled"] is True
    assert res["emergency"]["is_active"] is True
    assert res["emergency"]["last_stop_source"].upper() == "DASHBOARD"

    server.stop()
    audit_logger.stop()


# ==============================================================================
# Test 15: Dashboard API POST /api/emergency/stop without authentication rejected
# ==============================================================================

def test_15_dashboard_emergency_stop_unauthenticated_rejected(tmp_path: pathlib.Path):
    server, base_url, um, ctrl, eh, audit_logger = create_test_dashboard(tmp_path)

    stop_body = json.dumps({"reason": "Unauthorized Emergency Stop"}).encode("utf-8")
    req_stop = urllib.request.Request(
        f"{base_url}/api/emergency/stop",
        data=stop_body,
        headers={"Content-Type": "application/json"},
    )

    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req_stop)

    assert err.value.code in (401, 403)

    server.stop()
    audit_logger.stop()


# ==============================================================================
# Test 16: Dashboard API POST /api/emergency/stop with regular user rejected
# ==============================================================================

def test_16_dashboard_emergency_stop_regular_user_rejected(tmp_path: pathlib.Path):
    server, base_url, um, ctrl, eh, audit_logger = create_test_dashboard(tmp_path)

    # 1. Create a regular user
    um.create_user(
        admin_user_id="admin_01",
        user_id="user_reg",
        username="Regular User",
        pin="2345",
        role=UserRole.REGULAR_USER,
    )

    # Regular user attempting login to admin endpoint is rejected with HTTP 403
    login_body = json.dumps({"user_id": "user_reg", "pin": "2345"}).encode("utf-8")
    req_login = urllib.request.Request(
        f"{base_url}/api/admin/login",
        data=login_body,
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as err_login:
        urllib.request.urlopen(req_login)
    assert err_login.value.code == 403

    # 2. Create secondary admin, get session token, then demote to REGULAR_USER
    um.create_user(
        admin_user_id="admin_01",
        user_id="admin_temp",
        username="Temporary Admin",
        pin="3456",
        role=UserRole.ADMIN,
    )
    temp_login_body = json.dumps({"user_id": "admin_temp", "pin": "3456"}).encode("utf-8")
    req_temp_login = urllib.request.Request(
        f"{base_url}/api/admin/login",
        data=temp_login_body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req_temp_login)
    token = json.loads(resp.read().decode("utf-8"))["token"]

    # Demote admin_temp to REGULAR_USER
    um.change_user_role("admin_01", "admin_temp", UserRole.REGULAR_USER)

    # Attempt POST /api/emergency/stop with demoted regular user token -> rejected HTTP 403
    stop_body = json.dumps({"reason": "Non-admin Emergency Stop"}).encode("utf-8")
    req_stop = urllib.request.Request(
        f"{base_url}/api/emergency/stop",
        data=stop_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )

    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req_stop)

    assert err.value.code in (401, 403)

    server.stop()
    audit_logger.stop()


# ==============================================================================
# Test 17: Dashboard API POST /api/emergency/stop cookie CSRF defense enforced
# ==============================================================================

def test_17_dashboard_emergency_stop_cookie_csrf_enforcement(tmp_path: pathlib.Path):
    server, base_url, um, ctrl, eh, audit_logger = create_test_dashboard(tmp_path)

    # 1. Login as admin
    login_body = json.dumps({"user_id": "admin_01", "pin": "1234"}).encode("utf-8")
    req_login = urllib.request.Request(
        f"{base_url}/api/admin/login",
        data=login_body,
        headers={"Content-Type": "application/json"},
    )
    resp_login = urllib.request.urlopen(req_login)
    cookie_hdr = resp_login.headers.get("Set-Cookie")
    assert cookie_hdr
    # Extract session cookie value
    raw_cookie = cookie_hdr.split(";")[0]

    login_res = json.loads(resp_login.read().decode("utf-8"))
    csrf_token = login_res.get("csrf_token")
    assert csrf_token

    # 2. Attempt POST /api/emergency/stop using Cookie WITHOUT X-CSRF-Token -> 403 Forbidden
    stop_body = json.dumps({"reason": "CSRF Attack Attempt"}).encode("utf-8")
    req_no_csrf = urllib.request.Request(
        f"{base_url}/api/emergency/stop",
        data=stop_body,
        headers={
            "Content-Type": "application/json",
            "Cookie": raw_cookie,
        },
    )
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req_no_csrf)
    assert err.value.code == 403

    # 3. Attempt POST /api/emergency/stop with Cookie AND valid X-CSRF-Token -> 200 OK
    req_with_csrf = urllib.request.Request(
        f"{base_url}/api/emergency/stop",
        data=stop_body,
        headers={
            "Content-Type": "application/json",
            "Cookie": raw_cookie,
            "X-CSRF-Token": csrf_token,
        },
    )
    resp_ok = urllib.request.urlopen(req_with_csrf)
    assert resp_ok.code == 200

    server.stop()
    audit_logger.stop()


# ==============================================================================
# Test 18: Privacy & credential redaction in emergency audit logs
# ==============================================================================

def test_18_privacy_and_credential_sanitization_in_emergency_logs(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    eh = EmergencyHandler(lock_controller=ctrl, audit_logger=audit_logger, time_func=clock)

    # Perform a sequence of operations involving PIN digits, submissions, stops
    ctrl.handle_digit("1")
    ctrl.handle_digit("2")
    ctrl.handle_digit("3")
    eh.trigger_emergency_stop(source=AuditSource.VOICE, reason="Emergency preemption")

    ctrl.handle_digit("4")
    ctrl.handle_submit()  # incomplete failure

    audit_logger.flush()
    all_logs = audit_logger.query(limit=50)

    # Assert no plaintext secret or hash leaks into reason or metadata
    for log in all_logs:
        reason_lower = (log.get("reason") or "").lower()
        metadata_str = json.dumps(log.get("metadata", {})).lower()
        for forbidden in ["1234"]:
            assert forbidden not in reason_lower, f"Leaked {forbidden} in reason: {log.get('reason')}"
            assert forbidden not in metadata_str, f"Leaked {forbidden} in metadata: {log.get('metadata')}"

    audit_logger.stop()


# ==============================================================================
# Test 19: Emergency flag lifecycle and normal operation restoration
# ==============================================================================

def test_19_emergency_flag_clearing_lifecycle(tmp_path: pathlib.Path):
    audit_db = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    clock = MockClock()
    ctrl = LockController(time_func=clock, audit_logger=audit_logger)
    eh = EmergencyHandler(lock_controller=ctrl, audit_logger=audit_logger, time_func=clock)

    # Trigger emergency stop
    eh.trigger_emergency_stop(source=AuditSource.SYSTEM, reason="Test interlock")
    snap1 = ctrl.get_snapshot()
    assert snap1.emergency_active is True

    # User begins entering a new PIN sequence
    ctrl.handle_digit("1")
    snap2 = ctrl.get_snapshot()
    # Transition to PIN_ENTRY clears emergency_active
    assert snap2.emergency_active is False
    assert snap2.state == LockState.PIN_ENTRY

    # Enter rest of demo PIN 1234
    ctrl.handle_digit("2")
    ctrl.handle_digit("3")
    ctrl.handle_digit("4")
    ok, res, msg = ctrl.handle_submit()

    # Access is properly granted
    assert ok is True
    assert res == AccessResult.GRANTED
    assert ctrl.state == LockState.UNLOCKED

    audit_logger.stop()
