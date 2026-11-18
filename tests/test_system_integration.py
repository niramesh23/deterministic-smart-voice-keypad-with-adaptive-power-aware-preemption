"""
tests/test_system_integration.py
--------------------------------
Stage 9: Complete System Integration Test Suite.

Validates the full voice-recognition, smart-lock, security, monitoring,
analytics, and dashboard integration across all required scenarios:
- Scenario A: Successful Access (digits 1, 2, 3, 4 -> 'go' -> authenticated -> UNLOCKED -> audit -> auto-relock)
- Scenario B: Failed Access (incorrect PIN -> failure increment -> LOCKED -> audit)
- Scenario C: Lockout & Expiration (3 failed attempts -> LOCKOUT -> reject -> timer elapsed -> ready)
- Scenario D: Emergency STOP Precedence (STOP clears buffer, cancels pending, locks barrier)
- Scenario E: Multiple Users & RBAC (Admin, Regular, Guest expiration, account disabling)
- Scenario F: Fault Detection & Safe Recovery (Inference/pipeline failure -> safe recovery)
- Scenario G: Analytics & Reporting (Access reports and performance telemetry aggregation)
- Thread Safety & State Consistency (Simultaneous STOP/unlock, duplicate suppression, timeout race conditions)
- Real-Time Behaviour & Coherent Dashboard Snapshots
"""

import json
import os
import pathlib
import threading
import time
from typing import Any, Dict, List
import numpy as np
import pytest

from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig
from src.application.command_fsm import CommandRecord, CommandState
from src.security.config import SecurityConfig
from src.security.lock_controller import LockController, LockState, AccessResult
from src.security.emergency_handler import EmergencyHandler
from src.security.audit_logger import AuditLogger, AuditEventType, AuditResult, AuditSource
from src.security.user_manager import UserManager, UserRole, UserStatus
from src.fault.types import FaultType, FaultSeverity, SystemHealthState
from src.accessibility import AccessibilitySettings, AccessibilitySettingsManager, VoiceFeedback
from src.dashboard.server import DashboardServer, DashboardStatusProvider


# ==============================================================================
# Helper to build isolated test orchestrator
# ==============================================================================
def create_test_app(tmp_path: pathlib.Path, **config_overrides) -> VoiceRecognitionApp:
    audit_db = str(tmp_path / f"audit_{time.time_ns()}.db")
    users_db = str(tmp_path / f"users_{time.time_ns()}.db")
    a11y_cfg = str(tmp_path / f"a11y_{time.time_ns()}.json")

    sec_cfg = SecurityConfig(
        audit_db_path=audit_db,
        users_db_path=users_db,
        unlock_duration_s=config_overrides.get("unlock_duration_s", 0.5),
        lockout_duration_s=config_overrides.get("lockout_duration_s", 1.0),
        pin_timeout_s=config_overrides.get("pin_timeout_s", 10.0),
        enable_audit_logging=True,
    )
    orch_cfg = OrchestratorConfig(
        enable_dashboard=False,  # Headless by default for fast integration tests
        security_config=sec_cfg,
    )
    app = VoiceRecognitionApp(config=orch_cfg)

    # Initialize admin with PIN 1234
    if len(app.lock_controller.user_manager.get_all_users()) == 0:
        app.lock_controller.user_manager.bootstrap_initial_admin(
            admin_id="admin_01",
            username="Administrator",
            pin="1234",
        )

    return app


# ==============================================================================
# Scenario A: Successful Access
# ==============================================================================
def test_scenario_a_successful_access(tmp_path):
    """
    Scenario A:
    - Application starts.
    - User speaks digits 1, 2, 3, 4.
    - User speaks 'go'.
    - System identifies user.
    - Authentication succeeds.
    - Lock becomes UNLOCKED.
    - Audit event is recorded.
    - Dashboard updates.
    - Lock automatically returns to LOCKED.
    """
    app = create_test_app(tmp_path, unlock_duration_s=0.3)
    try:
        app.start(use_microphone=False)

        assert app.lock_controller.state == LockState.LOCKED

        # User speaks digits 1, 2, 3, 4
        app.handle_voice_command("one")
        assert app.lock_controller.state == LockState.PIN_ENTRY
        assert app.lock_controller.pin_manager.current_length == 1

        app.handle_voice_command("two")
        app.handle_voice_command("three")
        app.handle_voice_command("four")
        assert app.lock_controller.pin_manager.current_length == 4

        # User speaks 'go'
        app.handle_voice_command("go")

        # System identifies user and authenticates
        snap = app.lock_controller.get_snapshot()
        assert snap.state == LockState.UNLOCKED
        assert snap.last_access_result == AccessResult.GRANTED
        assert snap.authenticated_username == "Administrator"
        assert snap.authenticated_role == "ADMIN"

        # Check telemetry update in status provider
        app._update_status_provider()
        dash_snap = app.status_provider.get_snapshot()
        assert dash_snap["lock"]["state"] == "UNLOCKED"
        assert dash_snap["lock"]["authenticated_username"] == "Administrator"

        # Wait for auto-relock duration (0.3s)
        time.sleep(0.35)
        app.lock_controller.tick()

        # Lock automatically returns to LOCKED
        assert app.lock_controller.state == LockState.LOCKED
        assert app.lock_controller.authenticated_user is None

        # Check audit trail
        app.lock_controller.audit_logger.flush(0.2)
        events = app.lock_controller.audit_logger.query(limit=20)
        types = [e["event_type"] for e in events]
        assert "AUTH_SUCCESS" in types
        assert "LOCK_RELEASED" in types
        assert "LOCK_AUTO_RELOCK" in types
    finally:
        app.stop()


# ==============================================================================
# Scenario B: Failed Access
# ==============================================================================
def test_scenario_b_failed_access(tmp_path):
    """
    Scenario B:
    - User enters an incorrect PIN.
    - Authentication fails.
    - Lock remains LOCKED.
    - Failure count increases.
    - Audit event is recorded.
    """
    app = create_test_app(tmp_path)
    try:
        app.start(use_microphone=False)

        # Enter wrong PIN 9-9-9-9
        app.handle_voice_command("nine")
        app.handle_voice_command("nine")
        app.handle_voice_command("nine")
        app.handle_voice_command("nine")
        app.handle_voice_command("go")

        snap = app.lock_controller.get_snapshot()
        assert snap.state == LockState.LOCKED
        assert snap.last_access_result == AccessResult.DENIED_INCORRECT
        assert snap.failed_attempts == 1

        # Check audit trail
        app.lock_controller.audit_logger.flush(0.2)
        events = app.lock_controller.audit_logger.query(event_type="AUTH_FAILURE")
        assert len(events) >= 1
        assert events[0]["result"] == "FAILURE"
    finally:
        app.stop()


# ==============================================================================
# Scenario C: Lockout and Expiration
# ==============================================================================
def test_scenario_c_lockout_and_expiration(tmp_path):
    """
    Scenario C:
    - Three consecutive failed attempts occur.
    - System enters LOCKOUT.
    - Further authentication is rejected.
    - Lockout expires.
    - Authentication becomes available again.
    """
    app = create_test_app(tmp_path, lockout_duration_s=0.5)
    try:
        app.start(use_microphone=False)

        # 3 consecutive failed attempts
        for _ in range(3):
            app.handle_voice_command("five")
            app.handle_voice_command("five")
            app.handle_voice_command("five")
            app.handle_voice_command("five")
            app.handle_voice_command("go")

        snap = app.lock_controller.get_snapshot()
        assert snap.state == LockState.LOCKOUT
        assert snap.failed_attempts == 3

        # Further authentication is rejected during active lockout
        app.handle_voice_command("one")
        app.handle_voice_command("two")
        app.handle_voice_command("three")
        app.handle_voice_command("four")
        app.handle_voice_command("go")
        assert app.lock_controller.state == LockState.LOCKOUT

        # Wait for lockout expiration (0.5s)
        time.sleep(0.55)
        app.lock_controller.tick()

        # System returns to LOCKED and ready
        assert app.lock_controller.state == LockState.LOCKED
        assert app.lock_controller.failed_attempts == 0

        # Authentication becomes available again
        app.handle_voice_command("one")
        app.handle_voice_command("two")
        app.handle_voice_command("three")
        app.handle_voice_command("four")
        app.handle_voice_command("go")
        assert app.lock_controller.state == LockState.UNLOCKED
    finally:
        app.stop()


# ==============================================================================
# Scenario D: Emergency STOP Priority
# ==============================================================================
def test_scenario_d_emergency_stop_precedence(tmp_path):
    """
    Scenario D:
    - System enters PIN_ENTRY or UNLOCKED.
    - STOP is recognized.
    - Pending access operation is cancelled.
    - Lock returns to LOCKED.
    - Audit event is recorded.
    """
    app = create_test_app(tmp_path, unlock_duration_s=10.0)
    try:
        app.start(use_microphone=False)

        # 1. STOP during PIN_ENTRY
        app.handle_voice_command("one")
        app.handle_voice_command("two")
        assert app.lock_controller.state == LockState.PIN_ENTRY

        app.handle_voice_command("stop")
        assert app.lock_controller.state == LockState.LOCKED
        assert app.lock_controller.pin_manager.current_length == 0
        assert app.emergency_handler.is_emergency_active is True

        # Clear emergency flag to allow subsequent testing
        app.emergency_handler.clear_emergency_flag()

        # 2. STOP during UNLOCKED
        app.handle_voice_command("one")
        app.handle_voice_command("two")
        app.handle_voice_command("three")
        app.handle_voice_command("four")
        app.handle_voice_command("go")
        assert app.lock_controller.state == LockState.UNLOCKED

        app.handle_voice_command("stop")
        assert app.lock_controller.state == LockState.LOCKED
        assert app.lock_controller.authenticated_user is None

        # Verify audit records
        app.lock_controller.audit_logger.flush(0.2)
        stops = app.lock_controller.audit_logger.query(event_type="EMERGENCY_STOP")
        assert len(stops) >= 2
    finally:
        app.stop()


# ==============================================================================
# Scenario E: Multiple Users & RBAC
# ==============================================================================
def test_scenario_e_multiple_users_and_guest_rules(tmp_path):
    """
    Scenario E:
    - Two different users authenticate.
    - Their access events are associated with the correct user.
    - Guest expiration and disabled-account rules are enforced.
    """
    app = create_test_app(tmp_path, unlock_duration_s=0.2)
    try:
        app.start(use_microphone=False)

        um = app.lock_controller.user_manager
        # Create regular user: "user_bob" PIN "4321"
        um.create_user("admin_01", "user_bob", "Bob Jones", "4321", UserRole.REGULAR_USER)

        # Create expired guest: "guest_eve" PIN "5678"
        um.create_user("admin_01", "guest_eve", "Eve Guest", "5678", UserRole.GUEST, expires_at="2020-01-01T00:00:00+00:00")

        # Create disabled user: "user_dan" PIN "8765"
        um.create_user("admin_01", "user_dan", "Dan Brown", "8765", UserRole.REGULAR_USER)
        um.disable_user("admin_01", "user_dan")

        # 1. Bob Jones (Regular User) authenticates
        app.handle_voice_command("four")
        app.handle_voice_command("three")
        app.handle_voice_command("two")
        app.handle_voice_command("one")
        app.handle_voice_command("go")

        assert app.lock_controller.state == LockState.UNLOCKED
        assert app.lock_controller.authenticated_user.user_id == "user_bob"
        assert app.lock_controller.authenticated_user.role == UserRole.REGULAR_USER

        # Wait for relock
        time.sleep(0.25)
        app.lock_controller.tick()
        assert app.lock_controller.state == LockState.LOCKED

        # 2. Administrator authenticates
        app.handle_voice_command("one")
        app.handle_voice_command("two")
        app.handle_voice_command("three")
        app.handle_voice_command("four")
        app.handle_voice_command("go")

        assert app.lock_controller.state == LockState.UNLOCKED
        assert app.lock_controller.authenticated_user.user_id == "admin_01"
        assert app.lock_controller.authenticated_user.role == UserRole.ADMIN

        time.sleep(0.25)
        app.lock_controller.tick()

        # 3. Expired Guest is rejected
        app.handle_voice_command("five")
        app.handle_voice_command("six")
        app.handle_voice_command("seven")
        app.handle_voice_command("eight")
        app.handle_voice_command("go")
        assert app.lock_controller.state == LockState.LOCKED
        assert app.lock_controller.last_access_result == AccessResult.DENIED_INCORRECT

        # 4. Disabled Account is rejected
        app.handle_voice_command("eight")
        app.handle_voice_command("seven")
        app.handle_voice_command("six")
        app.handle_voice_command("five")
        app.handle_voice_command("go")
        assert app.lock_controller.state == LockState.LOCKED

        # Verify audit records mapped to correct user IDs
        app.lock_controller.audit_logger.flush(0.2)
        bob_events = app.lock_controller.audit_logger.query(user_id="user_bob")
        admin_events = app.lock_controller.audit_logger.query(user_id="admin_01")
        assert len(bob_events) >= 1
        assert len(admin_events) >= 1
    finally:
        app.stop()


# ==============================================================================
# Scenario F: Fault Recovery & Safe Subsystems
# ==============================================================================
def test_scenario_f_fault_detection_and_safe_recovery(tmp_path):
    """
    Scenario F:
    - Simulate an inference or dashboard failure.
    - FaultDetector identifies it.
    - RecoveryManager performs configured recovery.
    - Core system behaviour remains safe.
    """
    app = create_test_app(tmp_path)
    try:
        app.start(use_microphone=False)

        # Trigger simulated inference exception fault
        event = app.fault_detector.report_fault(
            fault_type=FaultType.INFERENCE_EXCEPTION,
            component="inference",
            description="Simulated torch forward-pass matrix mismatch",
            severity=FaultSeverity.ERROR,
            force_immediate=True,
        )

        assert event is not None
        assert event.component == "inference"

        # RecoveryManager should have executed restart action
        r_stats = app.recovery_manager.get_recovery_stats()
        assert r_stats["total_recoveries"] >= 1
        assert r_stats["succeeded"] >= 1

        # Check system health restored to HEALTHY following successful recovery
        assert app.fault_detector.system_health == SystemHealthState.HEALTHY

        # Lock state remains safe (LOCKED)
        assert app.lock_controller.state == LockState.LOCKED

        # Verify core access authentication still operates cleanly after recovery
        app.handle_voice_command("one")
        app.handle_voice_command("two")
        app.handle_voice_command("three")
        app.handle_voice_command("four")
        app.handle_voice_command("go")
        assert app.lock_controller.state == LockState.UNLOCKED
    finally:
        app.stop()


# ==============================================================================
# Scenario G: Analytics & Reporting
# ==============================================================================
def test_scenario_g_analytics_and_reporting(tmp_path):
    """
    Scenario G:
    - Run authentication events.
    - Confirm access reports reflect correct records.
    - Confirm performance metrics are collected.
    """
    app = create_test_app(tmp_path, unlock_duration_s=0.1)
    try:
        app.start(use_microphone=False)

        # 1 valid auth + 1 failed auth
        app.handle_voice_command("one")
        app.handle_voice_command("two")
        app.handle_voice_command("three")
        app.handle_voice_command("four")
        app.handle_voice_command("go")

        time.sleep(0.15)
        app.lock_controller.tick()

        app.handle_voice_command("nine")
        app.handle_voice_command("nine")
        app.handle_voice_command("nine")
        app.handle_voice_command("nine")
        app.handle_voice_command("go")

        app.lock_controller.audit_logger.flush(0.3)

        # Verify Access Analytics
        summary = app.access_analytics.get_summary()
        assert summary.overall.successful_attempts >= 1
        assert summary.overall.failed_attempts >= 1
        assert summary.overall.total_attempts == summary.overall.successful_attempts + summary.overall.failed_attempts

        # Generate JSON Report
        json_report = app.access_analytics.generate_report(fmt="json")
        data = json.loads(json_report.to_json())
        assert "overall_metrics" in data
        assert "records" in data
        assert len(data["records"]) >= 2

        # Generate CSV Report
        csv_report = app.access_analytics.generate_report(fmt="csv")
        csv_text = csv_report.to_csv()
        assert "# RTOS Access Control Security Audit Report" in csv_text
        for col in ["event_type", "result", "lock_state", "source", "user_id", "username", "reason"]:
            assert col in csv_text

        # Verify Real-Time Performance Analytics
        telemetry = app.analytics_collector.get_live_telemetry()
        assert "inference" in telemetry
        assert "scheduling" in telemetry
        assert "resources" in telemetry
    finally:
        app.stop()


# ==============================================================================
# Thread Safety & Concurrency Scenarios
# ==============================================================================
def test_thread_safety_simultaneous_stop_and_unlock(tmp_path):
    """
    Tests race condition where emergency STOP arrives concurrently while an unlock
    operation is staging or committing. Ensures STOP always wins and unlock is aborted.
    """
    app = create_test_app(tmp_path, unlock_duration_s=5.0)
    try:
        app.start(use_microphone=False)

        lock = app.lock_controller
        user = lock.user_manager.get_user("admin_01")

        # Stage a pending unlock request (op_id 1)
        op_id = lock.request_pending_unlock(user, delay_s=0.0)

        # Concurrently issue emergency STOP and attempt to execute pending unlock
        t_stop = threading.Thread(target=lock.emergency_stop)
        t_unlock = threading.Thread(target=lock.execute_pending_unlock, args=(op_id,))

        t_stop.start()
        t_unlock.start()
        t_stop.join()
        t_unlock.join()

        # Authority invariant: System must be in LOCKED state
        assert lock.state == LockState.LOCKED
        assert lock.get_snapshot().is_locked is True
        assert lock.authenticated_user is None
    finally:
        app.stop()


def test_thread_safety_duplicate_suppression_and_timeout(tmp_path):
    """
    Tests that rapid duplicate command IDs are cleanly suppressed and timeout
    during PIN entry clears state atomically.
    """
    app = create_test_app(tmp_path, pin_timeout_s=0.3)
    try:
        app.start(use_microphone=False)
        lock = app.lock_controller

        # Duplicate command ID suppression
        snap1 = lock.handle_command("one", command_id="cmd_unique_101")
        assert lock.pin_manager.current_length == 1

        snap2 = lock.handle_command("one", command_id="cmd_unique_101")
        # Length remains 1 because duplicate was suppressed!
        assert lock.pin_manager.current_length == 1

        # Timeout race condition: wait 0.35s and submit
        time.sleep(0.35)
        ok, res, msg = lock.handle_submit()
        assert ok is False
        assert lock.state == LockState.LOCKED
    finally:
        app.stop()


def test_thread_safety_user_removal_and_lockout_during_pending(tmp_path):
    """
    Tests that if a user is deleted/disabled or lockout is triggered during an in-flight
    operation, commit_unlock strictly rejects the attempt.
    """
    app = create_test_app(tmp_path)
    try:
        app.start(use_microphone=False)
        lock = app.lock_controller
        um = lock.user_manager

        um.create_user("admin_01", "temp_user", "Temporary", "2222", UserRole.REGULAR_USER)
        u_rec = um.get_user("temp_user")

        # Begin operation
        op_id = lock.begin_access_operation()

        # Remove user before commit
        um.remove_user("admin_01", "temp_user")

        # Now trigger lockout
        lock._activate_lockout()
        assert lock.state == LockState.LOCKOUT

        # Attempt to commit stale unlock
        ok, res, msg = lock.commit_unlock(op_id, u_rec)
        assert ok is False
        assert res == AccessResult.DENIED_LOCKOUT
        assert lock.state == LockState.LOCKOUT
    finally:
        app.stop()


# ==============================================================================
# Dashboard Coherence & Single Authoritative Controller
# ==============================================================================
def test_full_dashboard_snapshot_coherence(tmp_path):
    """
    Verifies that DashboardStatusProvider presents a coherent, synchronized snapshot
    reflecting all 10 required application sections without direct state mutation.
    """
    app = create_test_app(tmp_path)
    try:
        app.start(use_microphone=False)
        app._update_status_provider()
        snap = app.status_provider.get_snapshot()

        # 1. Live system status
        assert "microphone" in snap
        assert "scheduler" in snap
        assert "adaptive_controller" in snap
        assert "resources" in snap

        # 2. Lock status
        assert "lock" in snap
        assert snap["lock"]["state"] == "LOCKED"

        # 3. PIN authentication status
        assert snap["lock"]["pin_progress"] == "0/4 digits"
        assert snap["lock"]["failed_attempts"] == 0

        # 4. User management
        users = app.status_provider.get_users()
        assert len(users) >= 1
        assert users[0]["user_id"] == "admin_01"

        # 5. Security audit logs
        events = app.status_provider.query_audit(limit=10)
        assert isinstance(events, list)

        # 6. Emergency STOP status
        assert "emergency" in snap
        assert snap["emergency"]["is_active"] is False

        # 7. Fault monitoring
        assert "health" in snap
        assert snap["health"]["system_health"] == "HEALTHY"

        # 8. Real-time performance analytics
        assert "analytics" in snap

        # 9. Access analytics
        summary = app.status_provider.get_access_analytics_summary()
        assert "overall" in summary

        # 10. Accessibility settings
        a11y = app.status_provider.get_accessibility_settings()
        assert "large_text" in a11y
        assert "high_contrast" in a11y
        assert "spoken_feedback" in a11y

        # Invariant: Neither dashboard provider nor analytics mutated lock state directly
        assert app.lock_controller.state == LockState.LOCKED
    finally:
        app.stop()
