"""
tests/test_audit_logger.py
--------------------------
Automated test suite for Stage 2: Security Audit Logging.

Verifies all 12 prompt-mandated validation scenarios:
1. Successful authentication logged
2. Failed authentication logged
3. Lockout activation logged
4. Automatic relocking logged
5. STOP events logged
6. Records persist after restart
7. Timestamp ordering correct
8. Invalid metadata handled safely
9. Database failures handled safely
10. Concurrent events do not corrupt SQLite database
11. Queue overflow is measurable
12. Shutdown flushes pending records

Plus additional verification:
13. Privacy assurance: Zero PIN, password, or hash exposure in SQLite records or API
14. Dashboard /api/audit endpoint integration and filtering
"""

import json
import os
import pathlib
import sqlite3
import threading
import time
import urllib.request
import pytest

from src.security.audit_logger import (
    AuditLogger,
    AuditRecord,
    AuditEventType,
    AuditResult,
    AuditSource,
)
from src.security.config import SecurityConfig
from src.security.pin_manager import PINManager
from src.security.lock_controller import LockController, LockState, AccessResult
from src.dashboard.server import DashboardServer, DashboardStatusProvider


# ==============================================================================
# 1. Successful Authentication Logged
# ==============================================================================

def test_scenario_01_successful_authentication_logged(tmp_path):
    """
    Scenario 1: Verifies that entry of correct PIN followed by 'go' logs:
    - AUTH_SUBMISSION upon digit entry
    - AUTH_SUCCESS upon correct PIN evaluation
    - LOCK_RELEASED upon unlocking
    """
    db_file = tmp_path / "audit.db"
    cred_file = tmp_path / "credentials.json"
    cfg = SecurityConfig(
        credentials_path=str(cred_file),
        audit_db_path=str(db_file),
        initial_demo_pin="1234",
    )
    controller = LockController(config=cfg)

    # Speak digits '1', '2', '3', '4', then 'go'
    controller.handle_command("one")
    controller.handle_command("two")
    controller.handle_command("three")
    controller.handle_command("four")
    snap = controller.handle_command("go")

    assert snap.state == LockState.UNLOCKED
    assert snap.last_access_result == AccessResult.GRANTED

    # Flush audit logger to SQLite
    controller.audit_logger.flush(timeout=2.0)

    events = controller.audit_logger.query(limit=50)
    event_types = [e["event_type"] for e in events]

    assert AuditEventType.AUTH_SUBMISSION.value in event_types
    assert AuditEventType.AUTH_SUCCESS.value in event_types
    assert AuditEventType.LOCK_RELEASED.value in event_types

    # Check AUTH_SUCCESS record specifics
    auth_success_evt = next(e for e in events if e["event_type"] == AuditEventType.AUTH_SUCCESS.value)
    assert auth_success_evt["result"] == "SUCCESS"
    assert auth_success_evt["command_type"] == "go"
    assert "Voice PIN authentication succeeded" in auth_success_evt["reason"]

    controller.stop()


# ==============================================================================
# 2. Failed Authentication Logged
# ==============================================================================

def test_scenario_02_failed_authentication_logged(tmp_path):
    """
    Scenario 2: Verifies that incorrect PIN submission logs AUTH_FAILURE with failure details.
    """
    db_file = tmp_path / "audit.db"
    cred_file = tmp_path / "credentials.json"
    cfg = SecurityConfig(
        credentials_path=str(cred_file),
        audit_db_path=str(db_file),
        initial_demo_pin="1234",
    )
    controller = LockController(config=cfg)

    # Enter wrong PIN '9999'
    controller.handle_command("nine")
    controller.handle_command("nine")
    controller.handle_command("nine")
    controller.handle_command("nine")
    snap = controller.handle_command("go")

    assert snap.state == LockState.LOCKED
    assert snap.last_access_result == AccessResult.DENIED_INCORRECT

    controller.audit_logger.flush(timeout=2.0)
    events = controller.audit_logger.query(category="AUTH")
    event_types = [e["event_type"] for e in events]

    assert AuditEventType.AUTH_FAILURE.value in event_types
    fail_evt = next(e for e in events if e["event_type"] == AuditEventType.AUTH_FAILURE.value)
    assert fail_evt["result"] == "FAILURE"
    assert fail_evt["lock_state"] in ("PIN_ENTRY", "LOCKED")
    assert "Incorrect PIN" in fail_evt["reason"]

    controller.stop()


# ==============================================================================
# 3. Lockout Activation Logged
# ==============================================================================

def test_scenario_03_lockout_activation_logged(tmp_path):
    """
    Scenario 3: Verifies that reaching maximum failed attempts triggers LOCKOUT_ACTIVATED event.
    """
    db_file = tmp_path / "audit.db"
    cred_file = tmp_path / "credentials.json"
    cfg = SecurityConfig(
        credentials_path=str(cred_file),
        audit_db_path=str(db_file),
        initial_demo_pin="1234",
        max_failed_attempts=3,
    )
    controller = LockController(config=cfg)

    # 3 consecutive incorrect PIN attempts
    for _ in range(3):
        controller.handle_command("five")
        controller.handle_command("five")
        controller.handle_command("five")
        controller.handle_command("five")
        controller.handle_command("go")

    snap = controller.get_snapshot()
    assert snap.state == LockState.LOCKOUT

    controller.audit_logger.flush(timeout=2.0)
    events = controller.audit_logger.query(event_type=AuditEventType.LOCKOUT_ACTIVATED.value)

    assert len(events) >= 1
    lockout_evt = events[0]
    assert lockout_evt["result"] == "WARNING"
    assert "lockout activated" in lockout_evt["reason"].lower()

    controller.stop()


# ==============================================================================
# 4. Automatic Relocking Logged
# ==============================================================================

def test_scenario_04_automatic_relocking_logged(tmp_path):
    """
    Scenario 4: Verifies that expiration of unlock duration triggers LOCK_AUTO_RELOCK event.
    """
    db_file = tmp_path / "audit.db"
    cred_file = tmp_path / "credentials.json"

    # Virtual clock to advance time deterministically
    current_time = 1000.0

    def mock_time():
        return current_time

    cfg = SecurityConfig(
        credentials_path=str(cred_file),
        audit_db_path=str(db_file),
        initial_demo_pin="1234",
        unlock_duration_s=5.0,
    )
    controller = LockController(config=cfg, time_func=mock_time)

    # Unlock system
    controller.handle_command("one")
    controller.handle_command("two")
    controller.handle_command("three")
    controller.handle_command("four")
    controller.handle_command("go")
    assert controller.state == LockState.UNLOCKED

    # Advance time beyond unlock_duration_s
    current_time += 5.5
    controller.tick()

    assert controller.state == LockState.LOCKED
    assert controller.last_access_result == AccessResult.RELOCKED_AUTO

    controller.audit_logger.flush(timeout=2.0)
    events = controller.audit_logger.query(event_type=AuditEventType.LOCK_AUTO_RELOCK.value)

    assert len(events) == 1
    relock_evt = events[0]
    assert relock_evt["result"] == "INFO"
    assert relock_evt["lock_state"] == "LOCKED"
    assert "automatically relocked" in relock_evt["reason"].lower()

    controller.stop()


# ==============================================================================
# 5. STOP Events Logged
# ==============================================================================

def test_scenario_05_stop_events_logged(tmp_path):
    """
    Scenario 5: Verifies that speaking 'stop' logs EMERGENCY_STOP and LOCK_MANUAL_RELOCK when unlocked.
    """
    db_file = tmp_path / "audit.db"
    cred_file = tmp_path / "credentials.json"
    cfg = SecurityConfig(
        credentials_path=str(cred_file),
        audit_db_path=str(db_file),
        initial_demo_pin="1234",
    )
    controller = LockController(config=cfg)

    # 1. STOP during PIN entry
    controller.handle_command("one")
    controller.handle_command("two")
    controller.handle_command("stop")
    assert controller.state == LockState.LOCKED

    # 2. Unlock then STOP
    controller.handle_command("one")
    controller.handle_command("two")
    controller.handle_command("three")
    controller.handle_command("four")
    controller.handle_command("go")
    assert controller.state == LockState.UNLOCKED

    controller.handle_command("stop")
    assert controller.state == LockState.LOCKED

    controller.audit_logger.flush(timeout=2.0)
    events = controller.audit_logger.query(limit=50)

    stop_events = [e for e in events if e["event_type"] == AuditEventType.EMERGENCY_STOP.value]
    assert len(stop_events) >= 2

    manual_relock_events = [e for e in events if e["event_type"] == AuditEventType.LOCK_MANUAL_RELOCK.value]
    assert len(manual_relock_events) >= 1

    controller.stop()


# ==============================================================================
# 6. Records Persist After Restart
# ==============================================================================

def test_scenario_06_records_persist_after_restart(tmp_path):
    """
    Scenario 6: Verifies that SQLite audit events survive complete application restart.
    """
    db_file = tmp_path / "audit_persistent.db"

    # Instance 1: Log several events and stop
    logger1 = AuditLogger(db_path=db_file)
    logger1.log(
        event_type=AuditEventType.AUTH_SUCCESS,
        result=AuditResult.SUCCESS,
        reason="First session authentication",
        lock_state="UNLOCKED",
    )
    logger1.log(
        event_type=AuditEventType.SYSTEM_SHUTDOWN,
        result=AuditResult.INFO,
        reason="Clean shutdown session 1",
        lock_state="LOCKED",
    )
    logger1.stop()

    assert db_file.exists()

    # Instance 2: Start new logger pointing to same database
    logger2 = AuditLogger(db_path=db_file)
    records = logger2.query(limit=10)

    event_types = [r["event_type"] for r in records]
    assert AuditEventType.AUTH_SUCCESS.value in event_types
    assert AuditEventType.SYSTEM_SHUTDOWN.value in event_types
    assert len(records) >= 2

    logger2.stop()


# ==============================================================================
# 7. Timestamp Ordering Correct
# ==============================================================================

def test_scenario_07_timestamp_ordering_correct(tmp_path):
    """
    Scenario 7: Verifies that audit query returns records sorted newest to oldest,
    and timestamps follow valid ISO-8601 UTC format.
    """
    db_file = tmp_path / "audit_timestamps.db"
    logger_inst = AuditLogger(db_path=db_file)

    for i in range(5):
        logger_inst.log(
            event_type=AuditEventType.AUTH_SUBMISSION,
            result=AuditResult.INFO,
            reason=f"Step {i}",
            lock_state="LOCKED",
            metadata={"step": i},
        )
        time.sleep(0.01)

    logger_inst.flush()
    records = logger_inst.query(limit=10)

    assert len(records) == 5
    # Query returns newest to oldest
    for i in range(len(records) - 1):
        assert records[i]["event_id"] > records[i + 1]["event_id"]
        assert records[i]["timestamp"] >= records[i + 1]["timestamp"]

    logger_inst.stop()


# ==============================================================================
# 8. Invalid Metadata Handled Safely
# ==============================================================================

def test_scenario_08_invalid_metadata_handled_safely(tmp_path):
    """
    Scenario 8: Verifies that arbitrary non-serializable objects and nested values in metadata
    are safely sanitized without raising exceptions, and sensitive keys are stripped.
    """
    db_file = tmp_path / "audit_meta.db"
    logger_inst = AuditLogger(db_path=db_file)

    class CustomObject:
        def __str__(self):
            return "<CustomObjectInstance>"

    problematic_metadata = {
        "complex_obj": CustomObject(),
        "lambda_func": lambda x: x * 2,
        "pin": "1234",          # MUST BE STRIPPED
        "password": "secret",   # MUST BE STRIPPED
        "hash": "abcdef1234",   # MUST BE STRIPPED
        "valid_key": "safe_value",
        "nested_dict": {
            "pin": "9999",             # MUST BE STRIPPED (nested)
            "secret": "confidential",  # MUST BE STRIPPED (nested)
            "safe_nested": "ok",
        },
        "nested_list": [{"pin": "1111", "item": "safe"}],
    }

    # Should not raise exception
    ok = logger_inst.log(
        event_type=AuditEventType.ADMIN_ACTION,
        result=AuditResult.INFO,
        reason="Testing metadata sanitization",
        lock_state="LOCKED",
        metadata=problematic_metadata,
    )
    assert ok is True

    logger_inst.flush()
    records = logger_inst.query(limit=5)
    assert len(records) == 1
    meta = records[0]["metadata"]

    assert meta["valid_key"] == "safe_value"
    # Prohibited keys must be stripped at root and nested levels
    assert "pin" not in meta
    assert "password" not in meta
    assert "hash" not in meta
    assert "complex_obj" in meta
    assert "CustomObject" in meta["complex_obj"]

    # Nested sanitization verification
    assert "pin" not in meta["nested_dict"]
    assert "secret" not in meta["nested_dict"]
    assert meta["nested_dict"]["safe_nested"] == "ok"
    assert "pin" not in meta["nested_list"][0]
    assert meta["nested_list"][0]["item"] == "safe"

    logger_inst.stop()


# ==============================================================================
# 9. Database Failures Handled Safely
# ==============================================================================

def test_scenario_09_database_failures_handled_safely(tmp_path):
    """
    Scenario 9: Verifies that database write errors (e.g. read-only filesystem or invalid path)
    do not crash the logging call and increment error counters gracefully.
    """
    # Create invalid path that cannot be written as SQLite database
    # On Windows, using a directory path as the DB file path causes SQLite I/O failure
    bad_dir_as_file = tmp_path / "invalid_db_dir"
    bad_dir_as_file.mkdir()

    logger_inst = AuditLogger(db_path=bad_dir_as_file, enable_background_worker=False)

    # Logging synchronously to verify error catch
    rec = AuditRecord(
        event_type="FAULT_DETECTED",
        result="WARNING",
        reason="Testing bad DB path",
        lock_state="LOCKED",
    )
    ok = logger_inst.log_sync(rec)
    assert ok is False
    assert logger_inst.total_db_errors >= 1

    # Asynchronous call also returns safely
    ok_async = logger_inst.log(
        event_type="FAULT_DETECTED",
        result="WARNING",
        reason="Testing async with bad DB",
        lock_state="LOCKED",
    )
    assert ok_async is True  # Queue accepts it; worker handles write error safely


# ==============================================================================
# 10. Concurrent Events Do Not Corrupt SQLite Database
# ==============================================================================

def test_scenario_10_concurrent_events_integrity(tmp_path):
    """
    Scenario 10: Verifies that 10 concurrent threads logging simultaneously do not corrupt
    the SQLite database or cause database locked errors under WAL mode.
    """
    db_file = tmp_path / "audit_concurrency.db"
    logger_inst = AuditLogger(db_path=db_file, queue_capacity=5000)

    num_threads = 10
    events_per_thread = 50

    def worker_task(thread_id):
        for i in range(events_per_thread):
            logger_inst.log(
                event_type=AuditEventType.AUTH_SUBMISSION,
                result=AuditResult.INFO,
                reason=f"Worker {thread_id} event {i}",
                lock_state="PIN_ENTRY",
                user_id=f"user_{thread_id}",
                metadata={"thread": thread_id, "idx": i},
            )
            time.sleep(0.001)

    threads = [threading.Thread(target=worker_task, args=(tid,)) for tid in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Flush all events
    logger_inst.flush(timeout=5.0)

    # Verify database integrity with SQLite PRAGMA integrity_check
    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        check_result = cursor.fetchone()[0]
        assert check_result == "ok"

        cursor.execute("SELECT COUNT(*) FROM audit_events;")
        total_in_db = cursor.fetchone()[0]

    assert total_in_db == num_threads * events_per_thread
    assert logger_inst.total_db_errors == 0

    logger_inst.stop()


# ==============================================================================
# 11. Queue Overflow is Measurable
# ==============================================================================

def test_scenario_11_queue_overflow_is_measurable(tmp_path):
    """
    Scenario 11: Verifies that bounded queue saturation increments total_dropped_count
    and does not block the caller.
    """
    db_file = tmp_path / "audit_overflow.db"
    # Capacity = 5, background worker disabled so queue fills up completely
    logger_inst = AuditLogger(db_path=db_file, queue_capacity=5, enable_background_worker=False)

    # Push 15 items into a queue of size 5
    for i in range(15):
        logger_inst.log(
            event_type=AuditEventType.AUTH_SUBMISSION,
            result=AuditResult.INFO,
            reason=f"Event {i}",
            lock_state="LOCKED",
        )

    assert logger_inst.total_dropped_count > 0
    assert logger_inst.total_logged_count == 15
    # Dropped count + queue size == total logged
    assert logger_inst.total_dropped_count + logger_inst._queue.qsize() == 15


# ==============================================================================
# 12. Shutdown Flushes Pending Records
# ==============================================================================

def test_scenario_12_shutdown_flushes_pending_records(tmp_path):
    """
    Scenario 12: Verifies that invoking stop() flushes queued records to SQLite before stopping.
    """
    db_file = tmp_path / "audit_flush.db"
    logger_inst = AuditLogger(db_path=db_file, queue_capacity=100)

    for i in range(20):
        logger_inst.log(
            event_type=AuditEventType.AUTH_SUCCESS,
            result=AuditResult.SUCCESS,
            reason=f"Event {i}",
            lock_state="UNLOCKED",
        )

    # Calling stop() should join worker and flush queue
    logger_inst.stop()

    assert logger_inst._queue.empty()
    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM audit_events;")
        count = cursor.fetchone()[0]

    assert count == 20


# ==============================================================================
# 13. Privacy & Credential Leakage Prohibition
# ==============================================================================

def test_scenario_13_privacy_prohibits_plain_pins_and_hashes(tmp_path):
    """
    Scenario 13: Verifies that raw PINs ('1234') or salted PBKDF2 hashes are NEVER stored
    in the audit database, neither in text columns nor in metadata.
    """
    db_file = tmp_path / "audit_privacy.db"
    cred_file = tmp_path / "credentials.json"
    cfg = SecurityConfig(
        credentials_path=str(cred_file),
        audit_db_path=str(db_file),
        initial_demo_pin="1234",
    )
    controller = LockController(config=cfg)

    controller.handle_command("one")
    controller.handle_command("two")
    controller.handle_command("three")
    controller.handle_command("four")
    controller.handle_command("go")

    controller.audit_logger.flush(timeout=2.0)

    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT reason, command_type, metadata FROM audit_events;")
        rows = cursor.fetchall()

    for reason, cmd, meta_json in rows:
        # Plain PIN '1234' must never appear anywhere in audit trail
        assert "1234" not in reason
        assert "1234" not in str(cmd)
        assert "1234" not in str(meta_json)
        # Hash/salt fields must not appear
        assert "salt_hex" not in str(meta_json)
        assert "hash_hex" not in str(meta_json)

    controller.stop()


# ==============================================================================
# 14. Dashboard /api/audit Endpoint Integration
# ==============================================================================

def test_scenario_14_dashboard_audit_api_integration(tmp_path):
    """
    Scenario 14: Verifies that the dashboard HTTP server serves /api/audit with filtering.
    """
    db_file = tmp_path / "audit_dashboard.db"
    logger_inst = AuditLogger(db_path=db_file)

    logger_inst.log(
        event_type=AuditEventType.AUTH_SUCCESS,
        result=AuditResult.SUCCESS,
        reason="User access granted",
        lock_state="UNLOCKED",
    )
    logger_inst.log(
        event_type=AuditEventType.AUTH_FAILURE,
        result=AuditResult.FAILURE,
        reason="Wrong PIN entered",
        lock_state="LOCKED",
    )
    logger_inst.flush()

    provider = DashboardStatusProvider(audit_logger=logger_inst)
    server = DashboardServer(host="127.0.0.1", port=0, provider=provider)
    server.start()

    try:
        # Query /api/audit
        url = f"{server.url}/api/audit?limit=10"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert len(data) >= 2
            event_types = [d["event_type"] for d in data]
            assert "AUTH_SUCCESS" in event_types
            assert "AUTH_FAILURE" in event_types

        # Query /api/audit with filter result=FAILURE
        url_filter = f"{server.url}/api/audit?result=FAILURE"
        with urllib.request.urlopen(url_filter, timeout=3.0) as resp:
            assert resp.status == 200
            filtered = json.loads(resp.read().decode("utf-8"))
            assert len(filtered) == 1
            assert filtered[0]["result"] == "FAILURE"
            assert filtered[0]["event_type"] == "AUTH_FAILURE"

        # Query /api/audit with category filter
        url_cat = f"{server.url}/api/audit?category=AUTH"
        with urllib.request.urlopen(url_cat, timeout=3.0) as resp:
            assert resp.status == 200
            cat_data = json.loads(resp.read().decode("utf-8"))
            assert len(cat_data) >= 2

        # Query with invalid/unknown result filter (handled safely, returns empty)
        url_inv = f"{server.url}/api/audit?result=NONEXISTENT_RESULT"
        with urllib.request.urlopen(url_inv, timeout=3.0) as resp:
            assert resp.status == 200
            inv_data = json.loads(resp.read().decode("utf-8"))
            assert inv_data == []

        # Query with malformed limit and offset (handled safely, falls back to default)
        url_malformed = f"{server.url}/api/audit?limit=not_a_number&offset=bad"
        with urllib.request.urlopen(url_malformed, timeout=3.0) as resp:
            assert resp.status == 200
            mal_data = json.loads(resp.read().decode("utf-8"))
            assert len(mal_data) >= 2

    finally:
        server.stop()
        logger_inst.stop()


# ==============================================================================
# 15. Critical Event Saturation Edge-Case
# ==============================================================================

def test_scenario_15_critical_event_saturation_edge_case(tmp_path):
    """
    Scenario 15: Verifies behavior when bounded queue is completely saturated with critical events:
    1. New critical event evicts oldest record and is accepted.
    2. Evicted event increments total_dropped_count.
    3. Queue memory usage remains strictly bounded.
    4. Non-critical event attempting to enter full critical queue is dropped.
    """
    db_file = tmp_path / "audit_edge.db"
    capacity = 5
    logger_inst = AuditLogger(db_path=db_file, queue_capacity=capacity, enable_background_worker=False)

    # 1. Fill queue entirely with critical events
    for i in range(capacity):
        ok = logger_inst.log(
            event_type=AuditEventType.AUTH_FAILURE,
            result=AuditResult.FAILURE,
            reason=f"Critical Failure #{i}",
            lock_state="LOCKED",
        )
        assert ok is True

    assert logger_inst._queue.full() is True
    assert logger_inst._queue.qsize() == capacity
    assert logger_inst.total_dropped_count == 0

    # 2. Enqueue 6th critical event
    t0 = time.perf_counter()
    ok_6 = logger_inst.log(
        event_type=AuditEventType.EMERGENCY_STOP,
        result=AuditResult.WARNING,
        reason="Critical Emergency Stop #6",
        lock_state="LOCKED",
    )
    t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert ok_6 is True
    assert t_elapsed_ms < 50.0  # Responsive, non-blocking
    assert logger_inst.total_dropped_count == 1
    assert logger_inst._queue.qsize() == capacity

    # Verify FIFO eviction: #0 was evicted, #1 is first, #6 is last
    items = []
    while not logger_inst._queue.empty():
        items.append(logger_inst._queue.get_nowait())

    assert items[0].reason == "Critical Failure #1"
    assert items[-1].reason == "Critical Emergency Stop #6"
    assert len(items) == capacity

    # 3. Refill queue to full with critical items
    for it in items:
        logger_inst._queue.put_nowait(it)

    # 4. Attempt non-critical event when full of critical items
    ok_non_crit = logger_inst.log(
        event_type=AuditEventType.LOCK_STARTUP_INIT,
        result=AuditResult.INFO,
        reason="Non-critical info event",
        lock_state="LOCKED",
    )
    assert ok_non_crit is False  # Non-critical is rejected when queue is full of critical
    assert logger_inst.total_dropped_count == 2
    assert logger_inst._queue.qsize() == capacity

    logger_inst.stop()
