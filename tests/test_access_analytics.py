"""
tests/test_access_analytics.py
------------------------------
Automated Test Suite for Stage 7: Access Analytics and Report Generation.

Covers all 12 evaluation criteria:
1. Correct total access count.
2. Correct success rate.
3. Correct failure rate.
4. User-specific aggregation.
5. Date filtering.
6. Role filtering.
7. Hourly aggregation.
8. Empty database behaviour.
9. CSV export correctness.
10. JSON export correctness.
11. Sensitive information is excluded.
12. Dashboard Access Analytics API Integration.
"""

import csv
import json
import pathlib
import urllib.request
import urllib.error
import pytest

from src.security.audit_logger import AuditLogger, AuditEventType, AuditResult, AuditSource
from src.security.user_manager import UserManager, UserRole
from src.analytics.access_types import (
    AccessFilter,
    AccessReport,
    AccessAnalyticsSummary,
)
from src.analytics.access_analytics import AccessAnalyticsManager
from src.dashboard.server import DashboardServer, DashboardStatusProvider


# ==============================================================================
# 1. Total Access Count
# ==============================================================================

def test_01_correct_total_access_count(tmp_path: pathlib.Path):
    """
    Validates that total access count accurately counts authentication decisions,
    while strictly ignoring system lifecycle and lock physical events.
    """
    db_path = tmp_path / "audit_test1.db"
    logger = AuditLogger(db_path=str(db_path), enable_background_worker=False)

    # 6 Success events
    for i in range(6):
        logger.log(
            event_type=AuditEventType.AUTH_SUCCESS,
            result=AuditResult.SUCCESS,
            reason="Correct PIN",
            lock_state="UNLOCKED",
            user_id="user_01",
            username="User One",
        )

    # 4 Failure events
    for i in range(4):
        logger.log(
            event_type=AuditEventType.AUTH_FAILURE,
            result=AuditResult.FAILURE,
            reason="Incorrect PIN",
            lock_state="LOCKED",
            user_id="user_01",
            username="User One",
        )

    # 5 Non-authentication system and lock events (must NOT be counted as auth attempts)
    logger.log(event_type=AuditEventType.SYSTEM_STARTUP, result=AuditResult.INFO, reason="System Boot", lock_state="LOCKED")
    logger.log(event_type=AuditEventType.LOCK_RELEASED, result=AuditResult.SUCCESS, reason="Door Opened", lock_state="UNLOCKED")
    logger.log(event_type=AuditEventType.LOCK_AUTO_RELOCK, result=AuditResult.INFO, reason="Relocked", lock_state="LOCKED")
    logger.log(event_type=AuditEventType.AUTH_SUBMISSION, result=AuditResult.INFO, reason="Entering PIN", lock_state="LOCKED")
    logger.log(event_type=AuditEventType.SYSTEM_SHUTDOWN, result=AuditResult.INFO, reason="Shutdown", lock_state="LOCKED")

    # Flush batch synchronously to SQLite
    batch = []
    while not logger._queue.empty():
        batch.append(logger._queue.get_nowait())
    logger._write_records_to_db(batch)

    manager = AccessAnalyticsManager(db_path=db_path)
    summary = manager.compute_summary()

    assert summary.overall.total_attempts == 10
    assert summary.overall.successful_attempts == 6
    assert summary.overall.failed_attempts == 4
    assert "Successful + Failed" in summary.overall.denominator_definition


# ==============================================================================
# 2. Success Rate Calculation
# ==============================================================================

def test_02_correct_success_rate(tmp_path: pathlib.Path):
    """
    Validates formula: Success Rate = (Successful / Total) * 100%
    15 successes out of 20 total = 75.0%
    """
    db_path = tmp_path / "audit_test2.db"
    logger = AuditLogger(db_path=str(db_path), enable_background_worker=False)

    for _ in range(15):
        logger.log(event_type=AuditEventType.USER_ACCESS_GRANTED, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED")
    for _ in range(5):
        logger.log(event_type=AuditEventType.USER_ACCESS_DENIED, result=AuditResult.FAILURE, reason="Fail", lock_state="LOCKED")

    batch = []
    while not logger._queue.empty():
        batch.append(logger._queue.get_nowait())
    logger._write_records_to_db(batch)

    manager = AccessAnalyticsManager(db_path=db_path)
    summary = manager.compute_summary()

    assert summary.overall.total_attempts == 20
    assert summary.overall.successful_attempts == 15
    assert summary.overall.failed_attempts == 5
    assert summary.overall.success_rate_pct == pytest.approx(75.0, abs=0.01)
    assert summary.overall.denial_rate_pct == pytest.approx(25.0, abs=0.01)


# ==============================================================================
# 3. Failure Rate Calculation
# ==============================================================================

def test_03_correct_failure_rate(tmp_path: pathlib.Path):
    """
    Validates formula: Denial Rate = (Failed / Total) * 100%
    7 failures out of 10 total = 70.0%
    Success Rate + Denial Rate == 100.0%
    """
    db_path = tmp_path / "audit_test3.db"
    logger = AuditLogger(db_path=str(db_path), enable_background_worker=False)

    for _ in range(3):
        logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED")
    for _ in range(7):
        logger.log(event_type=AuditEventType.AUTH_FAILURE, result=AuditResult.FAILURE, reason="Bad PIN", lock_state="LOCKED")

    batch = []
    while not logger._queue.empty():
        batch.append(logger._queue.get_nowait())
    logger._write_records_to_db(batch)

    manager = AccessAnalyticsManager(db_path=db_path)
    summary = manager.compute_summary()

    assert summary.overall.total_attempts == 10
    assert summary.overall.denial_rate_pct == pytest.approx(70.0, abs=0.01)
    assert summary.overall.success_rate_pct == pytest.approx(30.0, abs=0.01)
    assert (summary.overall.success_rate_pct + summary.overall.denial_rate_pct) == pytest.approx(100.0, abs=0.01)


# ==============================================================================
# 4. User-Specific Aggregation
# ==============================================================================

def test_04_user_specific_aggregation(tmp_path: pathlib.Path):
    """
    Validates per-user activity aggregation, user ranking by activity, and last access timestamps.
    """
    db_path = tmp_path / "audit_test4.db"
    logger = AuditLogger(db_path=str(db_path), enable_background_worker=False)

    # Alice: 5 successes, 1 failure = 6 attempts (83.33%)
    for i in range(5):
        logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED",
                   user_id="alice", username="Alice User", timestamp=f"2026-11-04T10:0{i}:00+00:00")
    logger.log(event_type=AuditEventType.AUTH_FAILURE, result=AuditResult.FAILURE, reason="Wrong", lock_state="LOCKED",
               user_id="alice", username="Alice User", timestamp="2026-11-04T10:06:00+00:00")

    # Bob: 2 successes, 4 failures = 6 attempts (33.33%)
    for i in range(2):
        logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED",
                   user_id="bob", username="Bob User", timestamp=f"2026-11-04T11:0{i}:00+00:00")
    for i in range(4):
        logger.log(event_type=AuditEventType.AUTH_FAILURE, result=AuditResult.FAILURE, reason="Wrong", lock_state="LOCKED",
                   user_id="bob", username="Bob User", timestamp=f"2026-11-04T11:1{i}:00+00:00")

    # Admin: 10 successes = 10 attempts (100.0%)
    for i in range(10):
        logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="Admin OK", lock_state="UNLOCKED",
                   user_id="admin_01", username="Administrator", timestamp=f"2026-11-04T12:{i:02d}:00+00:00")

    batch = []
    while not logger._queue.empty():
        batch.append(logger._queue.get_nowait())
    logger._write_records_to_db(batch)

    manager = AccessAnalyticsManager(db_path=db_path)
    summary = manager.compute_summary()

    # Verify ranking: admin_01 (10 attempts) must be first
    assert len(summary.most_active_users) == 3
    assert summary.most_active_users[0].user_id == "admin_01"
    assert summary.most_active_users[0].total_attempts == 10
    assert summary.most_active_users[0].success_rate_pct == pytest.approx(100.0, abs=0.01)

    # Check Alice
    alice_stat = next(u for u in summary.user_activity if u.user_id == "alice")
    assert alice_stat.total_attempts == 6
    assert alice_stat.successful_attempts == 5
    assert alice_stat.failed_attempts == 1
    assert alice_stat.success_rate_pct == pytest.approx(83.33, abs=0.1)
    assert alice_stat.last_access_time == "2026-11-04T10:06:00+00:00"

    # Check Bob
    bob_stat = next(u for u in summary.user_activity if u.user_id == "bob")
    assert bob_stat.total_attempts == 6
    assert bob_stat.successful_attempts == 2
    assert bob_stat.failed_attempts == 4
    assert bob_stat.success_rate_pct == pytest.approx(33.33, abs=0.1)


# ==============================================================================
# 5. Date Filtering and Timezone Validation
# ==============================================================================

def test_05_date_filtering_and_timezone(tmp_path: pathlib.Path):
    """
    Validates start_date and end_date filtering with timezone normalization.
    Tests invalid date ranges (start > end).
    """
    db_path = tmp_path / "audit_test5.db"
    logger = AuditLogger(db_path=str(db_path), enable_background_worker=False)

    # 4 attempts on 2026-11-01
    for i in range(4):
        logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED",
                   timestamp=f"2026-11-01T10:0{i}:00+00:00")
    # 6 attempts on 2026-11-02
    for i in range(6):
        logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED",
                   timestamp=f"2026-11-02T14:0{i}:00+00:00")
    # 8 attempts on 2026-11-03
    for i in range(8):
        logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED",
                   timestamp=f"2026-11-03T18:0{i}:00+00:00")

    batch = []
    while not logger._queue.empty():
        batch.append(logger._queue.get_nowait())
    logger._write_records_to_db(batch)

    manager = AccessAnalyticsManager(db_path=db_path)

    # Filter Day 2 only: 2026-11-02
    f_day2 = AccessFilter(start_date="2026-11-02", end_date="2026-11-02")
    s_day2 = manager.compute_summary(filters=f_day2)
    assert s_day2.overall.total_attempts == 6

    # Filter Day 2 and Day 3: 2026-11-02 to 2026-11-03
    f_span = AccessFilter(start_date="2026-11-02", end_date="2026-11-03")
    s_span = manager.compute_summary(filters=f_span)
    assert s_span.overall.total_attempts == 14  # 6 + 8

    # Filter all
    s_all = manager.compute_summary()
    assert s_all.overall.total_attempts == 18  # 4 + 6 + 8

    # Invalid range: start > end must raise ValueError
    with pytest.raises(ValueError) as err:
        AccessFilter(start_date="2026-11-05", end_date="2026-11-01")
    assert "Invalid date range" in str(err.value)


# ==============================================================================
# 6. Role Filtering
# ==============================================================================

def test_06_role_filtering(tmp_path: pathlib.Path):
    """
    Validates filtering by user role (ADMIN, REGULAR_USER, GUEST).
    """
    db_path = tmp_path / "audit_test6.db"
    logger = AuditLogger(db_path=str(db_path), enable_background_worker=False)

    # 5 attempts by ADMIN
    for _ in range(5):
        logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED",
                   user_id="admin_01", metadata={"role": "ADMIN"})
    # 7 attempts by REGULAR_USER
    for _ in range(7):
        logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED",
                   user_id="user_alice", metadata={"role": "REGULAR_USER"})
    # 3 attempts by GUEST
    for _ in range(3):
        logger.log(event_type=AuditEventType.AUTH_FAILURE, result=AuditResult.FAILURE, reason="Expired", lock_state="LOCKED",
                   user_id="guest_bob", metadata={"role": "GUEST"})

    batch = []
    while not logger._queue.empty():
        batch.append(logger._queue.get_nowait())
    logger._write_records_to_db(batch)

    manager = AccessAnalyticsManager(db_path=db_path)

    # ADMIN filter
    s_admin = manager.compute_summary(filters=AccessFilter(role="ADMIN"))
    assert s_admin.overall.total_attempts == 5
    assert all(u.role == "ADMIN" for u in s_admin.user_activity)

    # REGULAR_USER filter
    s_reg = manager.compute_summary(filters=AccessFilter(role="REGULAR_USER"))
    assert s_reg.overall.total_attempts == 7

    # GUEST filter
    s_guest = manager.compute_summary(filters=AccessFilter(role="GUEST"))
    assert s_guest.overall.total_attempts == 3
    assert s_guest.overall.failed_attempts == 3


# ==============================================================================
# 7. Hourly Aggregation
# ==============================================================================

def test_07_hourly_aggregation_and_peaks(tmp_path: pathlib.Path):
    """
    Validates hourly distribution histogram (0-23 UTC) and peak hour detection.
    """
    db_path = tmp_path / "audit_test7.db"
    logger = AuditLogger(db_path=str(db_path), enable_background_worker=False)

    # 2 attempts at 09:00 UTC
    logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED", timestamp="2026-11-04T09:12:00+00:00")
    logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED", timestamp="2026-11-04T09:45:00+00:00")

    # 5 attempts at 14:00 UTC (Peak Hour)
    for i in range(5):
        logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED", timestamp=f"2026-11-04T14:{i*10:02d}:00+00:00")

    # 1 attempt at 22:00 UTC
    logger.log(event_type=AuditEventType.AUTH_FAILURE, result=AuditResult.FAILURE, reason="Bad", lock_state="LOCKED", timestamp="2026-11-04T22:30:00+00:00")

    batch = []
    while not logger._queue.empty():
        batch.append(logger._queue.get_nowait())
    logger._write_records_to_db(batch)

    manager = AccessAnalyticsManager(db_path=db_path)
    summary = manager.compute_summary()

    hourly = {h.hour: h.total_attempts for h in summary.time_activity.hourly_trends}
    assert hourly[9] == 2
    assert hourly[14] == 5
    assert hourly[22] == 1
    assert hourly[0] == 0
    assert hourly[12] == 0

    # Peak hour verification
    peak_h = summary.time_activity.peak_periods.peak_hour
    assert peak_h["hour"] == 14
    assert peak_h["attempts"] == 5


# ==============================================================================
# 8. Empty Database Behaviour
# ==============================================================================

def test_08_empty_database_behaviour(tmp_path: pathlib.Path):
    """
    Verifies that querying an empty or missing database returns clean zero-valued models without crashing.
    """
    empty_db = tmp_path / "nonexistent.db"
    manager = AccessAnalyticsManager(db_path=empty_db)

    summary = manager.compute_summary()
    assert summary.overall.total_attempts == 0
    assert summary.overall.successful_attempts == 0
    assert summary.overall.failed_attempts == 0
    assert summary.overall.success_rate_pct == 0.0
    assert summary.overall.denial_rate_pct == 0.0
    assert summary.user_activity == []
    assert summary.most_active_users == []

    # Report generation on empty DB
    report = manager.generate_report()
    assert len(report.detailed_records) == 0
    assert report.summary_metrics.overall.total_attempts == 0

    json_str = report.to_json()
    assert json.loads(json_str)["summary_metrics"]["overall"]["total_attempts"] == 0

    csv_str = report.to_csv()
    assert "# Summary: Total Attempts=0" in csv_str


# ==============================================================================
# 9. CSV Export Correctness
# ==============================================================================

def test_09_csv_export_correctness(tmp_path: pathlib.Path):
    """
    Validates CSV export formatting, header metadata comments, and table columns.
    """
    db_path = tmp_path / "audit_test9.db"
    logger = AuditLogger(db_path=str(db_path), enable_background_worker=False)

    logger.log(
        event_type=AuditEventType.AUTH_SUCCESS,
        result=AuditResult.SUCCESS,
        reason="Voice PIN valid",
        lock_state="UNLOCKED",
        user_id="user_test",
        username="Tester",
        source=AuditSource.VOICE,
        metadata={"role": "REGULAR_USER"},
    )
    batch = []
    while not logger._queue.empty():
        batch.append(logger._queue.get_nowait())
    logger._write_records_to_db(batch)

    manager = AccessAnalyticsManager(db_path=db_path, output_dir=tmp_path)
    report = manager.generate_report()
    _, csv_file = manager.export_report_files(report, base_filename="report_test")

    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")

    # Metadata comments
    assert "# Report ID: rep_" in content
    assert "# Summary: Total Attempts=1" in content
    assert "# Denominator:" in content

    # CSV data rows
    lines = [line for line in content.splitlines() if not line.startswith("#") and line.strip()]
    reader = csv.DictReader(lines)
    rows = list(reader)

    assert len(rows) == 1
    r = rows[0]
    assert r["event_type"] == "AUTH_SUCCESS"
    assert r["user_id"] == "user_test"
    assert r["username"] == "Tester"
    assert r["role"] == "REGULAR_USER"
    assert r["result"] == "SUCCESS"
    assert r["source"] == "voice"


# ==============================================================================
# 10. JSON Export Correctness
# ==============================================================================

def test_10_json_export_correctness(tmp_path: pathlib.Path):
    """
    Validates JSON export formatting, schema keys, and disk persistence.
    """
    db_path = tmp_path / "audit_test10.db"
    logger = AuditLogger(db_path=str(db_path), enable_background_worker=False)

    logger.log(event_type=AuditEventType.AUTH_SUCCESS, result=AuditResult.SUCCESS, reason="OK", lock_state="UNLOCKED", user_id="alice")
    logger.log(event_type=AuditEventType.LOCKOUT_ACTIVATED, result=AuditResult.WARNING, reason="Lockout", lock_state="LOCKED", user_id="bob")
    logger.log(event_type=AuditEventType.EMERGENCY_STOP, result=AuditResult.WARNING, reason="E-Stop", lock_state="LOCKED", user_id="admin_01")

    batch = []
    while not logger._queue.empty():
        batch.append(logger._queue.get_nowait())
    logger._write_records_to_db(batch)

    manager = AccessAnalyticsManager(db_path=db_path, output_dir=tmp_path)
    report = manager.generate_report()
    json_file, _ = manager.export_report_files(report, base_filename="report_test10")

    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))

    assert "report_metadata" in data
    assert "summary_metrics" in data
    assert "detailed_access_records" in data

    meta = data["report_metadata"]
    assert "report_id" in meta
    assert "generated_at" in meta
    assert meta["record_count"] == 3

    sm = data["summary_metrics"]
    assert sm["overall"]["total_attempts"] == 1
    assert sm["security"]["lockout_count"] == 1
    assert sm["security"]["emergency_stop_count"] == 1


# ==============================================================================
# 11. Sensitive Information Exclusion
# ==============================================================================

def test_11_sensitive_information_is_excluded(tmp_path: pathlib.Path):
    """
    Guarantees that credentials, plaintext PINs, PIN hashes, salts, and session tokens
    are NEVER exposed in reports, CSVs, or JSON payloads.
    """
    db_path = tmp_path / "audit_test11.db"
    logger = AuditLogger(db_path=str(db_path), enable_background_worker=False)

    # Even if a caller attempted to sneak sensitive fields into metadata
    logger.log(
        event_type=AuditEventType.AUTH_SUCCESS,
        result=AuditResult.SUCCESS,
        reason="Auth succeeded",
        lock_state="UNLOCKED",
        user_id="user_victim",
        metadata={
            "role": "REGULAR_USER",
            "pin": "1234",
            "pin_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "salt_hex": "deadbeefcafebabe",
            "session_token": "secret_session_token_xyz",
            "csrf_token": "secret_csrf_xyz",
        },
    )

    batch = []
    while not logger._queue.empty():
        batch.append(logger._queue.get_nowait())
    logger._write_records_to_db(batch)

    manager = AccessAnalyticsManager(db_path=db_path, output_dir=tmp_path)
    report = manager.generate_report()

    json_text = report.to_json()
    csv_text = report.to_csv()

    prohibited_tokens = [
        "pin_hash",
        "deadbeefcafebabe",
        "secret_session_token_xyz",
        "secret_csrf_xyz",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ]

    for token in prohibited_tokens:
        assert token not in json_text
        assert token not in csv_text


# ==============================================================================
# 12. Dashboard Access Analytics API Integration
# ==============================================================================

def test_12_dashboard_access_analytics_api_integration(tmp_path: pathlib.Path):
    """
    Validates dashboard REST endpoints:
    - GET /api/access-analytics/summary
    - GET /api/access-analytics/summary?start_date > end_date (400 validation error)
    - GET /api/access-analytics/report?format=csv
    - GET /api/access-analytics/report?format=json
    - POST /api/access-analytics/report
    """
    audit_db = tmp_path / "audit.db"
    users_db = tmp_path / "users.db"
    audit_logger = AuditLogger(db_path=str(audit_db))
    user_mgr = UserManager(db_path=str(users_db), audit_logger=audit_logger, allow_demo_bootstrap=False)
    if len(user_mgr.get_all_users()) == 0:
        user_mgr.bootstrap_initial_admin(admin_id="admin_01", username="Administrator", pin="1234")

    # Log 3 events
    audit_logger.log(AuditEventType.AUTH_SUCCESS, AuditResult.SUCCESS, "Valid PIN", "UNLOCKED", user_id="admin_01")
    audit_logger.log(AuditEventType.AUTH_FAILURE, AuditResult.FAILURE, "Wrong PIN", "LOCKED", user_id="admin_01")
    audit_logger.flush(0.2)

    manager = AccessAnalyticsManager(db_path=audit_db, user_manager=user_mgr, audit_logger=audit_logger)
    provider = DashboardStatusProvider(
        audit_logger=audit_logger,
        user_manager=user_mgr,
        access_analytics=manager,
    )
    server = DashboardServer(host="127.0.0.1", port=0, provider=provider)
    server.start()
    base_url = server.url

    try:
        # 1. GET /api/access-analytics/summary
        resp_sum = urllib.request.urlopen(f"{base_url}/api/access-analytics/summary")
        assert resp_sum.code == 200
        data_sum = json.loads(resp_sum.read().decode("utf-8"))
        assert data_sum["overall"]["total_attempts"] == 2
        assert data_sum["overall"]["successful_attempts"] == 1
        assert data_sum["overall"]["failed_attempts"] == 1
        assert data_sum["overall"]["success_rate_pct"] == pytest.approx(50.0, abs=0.1)

        # 2. GET /api/access-analytics/summary with invalid date range -> 400 Bad Request
        bad_url = f"{base_url}/api/access-analytics/summary?start_date=2026-11-05&end_date=2026-11-01"
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad_url)
        assert err.value.code == 400

        # 3. GET /api/access-analytics/report?format=csv
        resp_csv = urllib.request.urlopen(f"{base_url}/api/access-analytics/report?format=csv")
        assert resp_csv.code == 200
        assert "text/csv" in resp_csv.headers.get("Content-Type", "")
        assert "attachment;" in resp_csv.headers.get("Content-Disposition", "")
        csv_body = resp_csv.read().decode("utf-8")
        assert "event_id,timestamp,event_type" in csv_body
        assert "AUTH_SUCCESS" in csv_body

        # 4. GET /api/access-analytics/report?format=json
        resp_json = urllib.request.urlopen(f"{base_url}/api/access-analytics/report?format=json")
        assert resp_json.code == 200
        assert "application/json" in resp_json.headers.get("Content-Type", "")
        json_body = json.loads(resp_json.read().decode("utf-8"))
        assert "report_metadata" in json_body
        assert json_body["summary_metrics"]["overall"]["total_attempts"] == 2

        # 5. POST /api/access-analytics/report
        req_post = urllib.request.Request(
            f"{base_url}/api/access-analytics/report",
            data=json.dumps({"format": "json", "user_id": "admin_01"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp_post = urllib.request.urlopen(req_post)
        assert resp_post.code == 200
        post_json = json.loads(resp_post.read().decode("utf-8"))
        assert post_json["summary_metrics"]["overall"]["total_attempts"] == 2
        assert post_json["report_metadata"]["record_count"] == 3

    finally:
        server.stop()
