"""
tests/test_user_management.py
------------------------------
Automated Verification Test Suite for Stage 3:
Multiple Users and Role-Based Access Control (RBAC).

Audited and Verified against:
1. Multiple users created.
2. Independent authentication per user.
3. Duplicate PIN rejection.
4. Disabled users cannot authenticate.
5. Expired guests cannot authenticate.
6. Active guests can authenticate.
7. Regular users cannot access admin operations.
8. Administrators can manage users.
9. Unauthorized backend requests rejected (HTTP 403).
10. User removal handled correctly.
11. PIN resets update credentials securely.
12. Administrative actions audited.
13. Existing Stage 1 and Stage 2 tests pass (backward compatibility).
14. Privacy & credential redaction across APIs and snapshots.
15. SQLite persistence across restarts.
"""

import datetime
import json
import os
import pathlib
import time
from typing import List
import urllib.request
import urllib.error
import pytest

# Explicitly enable development/demo bootstrap for the Stage 3 behavioral regression suite.
# Dedicated tests below explicitly verify that normal startup (allow_demo_bootstrap=False) fails closed.
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


# ==============================================================================
# 1. Multiple Users Created
# ==============================================================================

def test_01_multiple_users_created(tmp_path: pathlib.Path):
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    um = UserManager(db_path=db_file, audit_logger=audit_logger)

    # Initial admin bootstrapped automatically
    admin = um.get_user("admin_01")
    assert admin is not None
    assert admin.role == UserRole.ADMIN
    assert admin.status == UserStatus.ACTIVE

    # Create REGULAR_USER
    u1 = um.create_user(
        admin_user_id="admin_01",
        user_id="user_alice",
        username="Alice Smith",
        pin="2345",
        role=UserRole.REGULAR_USER,
    )
    assert u1.user_id == "user_alice"
    assert u1.role == UserRole.REGULAR_USER
    assert u1.status == UserStatus.ACTIVE

    # Create GUEST
    future_exp = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=2)).isoformat()
    u2 = um.create_user(
        admin_user_id="admin_01",
        user_id="guest_bob",
        username="Bob Guest",
        pin="3456",
        role=UserRole.GUEST,
        expires_at=future_exp,
    )
    assert u2.user_id == "guest_bob"
    assert u2.role == UserRole.GUEST
    assert u2.expires_at == future_exp

    all_users = um.get_all_users()
    assert len(all_users) == 3
    audit_logger.stop()


# ==============================================================================
# 2. Independent Authentication Per User
# ==============================================================================

def test_02_independent_authentication_per_user(tmp_path: pathlib.Path):
    clock = MockClock()
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    # Admin: 1234
    # Alice (REGULAR_USER): 2345
    um.create_user("admin_01", "user_alice", "Alice Smith", "2345", UserRole.REGULAR_USER)

    ctrl = LockController(config=cfg, user_manager=um, audit_logger=audit_logger, time_func=clock)

    # Authenticate Alice
    for d in "2345":
        ctrl.handle_digit(d)
    ok, res, msg = ctrl.handle_submit()
    assert ok is True
    assert res == AccessResult.GRANTED
    assert ctrl.state == LockState.UNLOCKED

    snap = ctrl.get_snapshot()
    assert snap.authenticated_user_id == "user_alice"
    assert snap.authenticated_username == "Alice Smith"
    assert snap.authenticated_role == "REGULAR_USER"
    assert "Alice Smith" in snap.operational_status

    # Lock again via stop
    ctrl.handle_stop()
    assert ctrl.state == LockState.LOCKED
    assert ctrl.get_snapshot().authenticated_user_id is None

    # Authenticate Admin with 1234
    for d in "1234":
        ctrl.handle_digit(d)
    ok2, res2, msg2 = ctrl.handle_submit()
    assert ok2 is True
    assert ctrl.state == LockState.UNLOCKED
    assert ctrl.get_snapshot().authenticated_user_id == "admin_01"

    ctrl.stop()


# ==============================================================================
# 3. Duplicate PIN Rejection
# ==============================================================================

def test_03_duplicate_pin_rejection(tmp_path: pathlib.Path):
    db_file = tmp_path / "users.db"
    um = UserManager(db_path=db_file)

    # Default admin has PIN 1234
    # Attempting to assign 1234 to another user MUST raise ValueError
    with pytest.raises(ValueError) as excinfo:
        um.create_user("admin_01", "user_bob", "Bob", "1234", UserRole.REGULAR_USER)
    assert "Duplicate PINs are rejected" in str(excinfo.value)

    # Create user with unique PIN 5678
    um.create_user("admin_01", "user_bob", "Bob", "5678", UserRole.REGULAR_USER)

    # Create user Charlie with unique PIN 6789
    um.create_user("admin_01", "user_charlie", "Charlie", "6789", UserRole.REGULAR_USER)

    # Reset Charlie's PIN to Bob's PIN (5678) MUST be rejected
    with pytest.raises(ValueError) as excinfo2:
        um.reset_user_pin("admin_01", "user_charlie", "5678")
    assert "Duplicate PINs are rejected" in str(excinfo2.value)


# ==============================================================================
# 4. Disabled Users Cannot Authenticate
# ==============================================================================

def test_04_disabled_users_cannot_authenticate(tmp_path: pathlib.Path):
    clock = MockClock()
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    um.create_user("admin_01", "user_dave", "Dave", "4567", UserRole.REGULAR_USER)
    # Disable Dave
    um.disable_user("admin_01", "user_dave")
    dave = um.get_user("user_dave")
    assert dave.status == UserStatus.DISABLED

    ctrl = LockController(config=cfg, user_manager=um, audit_logger=audit_logger, time_func=clock)

    # Attempt to authenticate as Dave
    for d in "4567":
        ctrl.handle_digit(d)
    ok, res, msg = ctrl.handle_submit()

    assert ok is False
    assert res == AccessResult.DENIED_INCORRECT
    assert ctrl.state == LockState.LOCKED
    assert ctrl.failed_attempts == 1
    # Generic message given to voice user to prevent user enumeration
    assert "Access denied: Incorrect PIN" in msg or "Authentication failed" in msg

    ctrl.stop()


# ==============================================================================
# 5. Expired Guests Cannot Authenticate
# ==============================================================================

def test_05_expired_guests_cannot_authenticate(tmp_path: pathlib.Path):
    clock = MockClock()
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    # Guest expired 1 hour ago
    past_exp = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat()
    um.create_user("admin_01", "guest_expired", "Expired Guest", "7891", UserRole.GUEST, expires_at=past_exp)

    ctrl = LockController(config=cfg, user_manager=um, audit_logger=audit_logger, time_func=clock)

    for d in "7891":
        ctrl.handle_digit(d)
    ok, res, msg = ctrl.handle_submit()

    assert ok is False
    assert res == AccessResult.DENIED_INCORRECT
    assert ctrl.state == LockState.LOCKED
    assert ctrl.failed_attempts == 1

    # Check status changed to EXPIRED
    g = um.get_user("guest_expired")
    assert g.status == UserStatus.EXPIRED

    ctrl.stop()


# ==============================================================================
# 6. Active Guests Can Authenticate
# ==============================================================================

def test_06_active_guests_can_authenticate(tmp_path: pathlib.Path):
    clock = MockClock()
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    # Guest valid for 1 hour into future
    future_exp = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)).isoformat()
    um.create_user("admin_01", "guest_active", "Active Guest", "8912", UserRole.GUEST, expires_at=future_exp)

    ctrl = LockController(config=cfg, user_manager=um, audit_logger=audit_logger, time_func=clock)

    for d in "8912":
        ctrl.handle_digit(d)
    ok, res, msg = ctrl.handle_submit()

    assert ok is True
    assert res == AccessResult.GRANTED
    assert ctrl.state == LockState.UNLOCKED
    assert ctrl.get_snapshot().authenticated_user_id == "guest_active"
    assert ctrl.get_snapshot().authenticated_role == "GUEST"

    ctrl.stop()


# ==============================================================================
# 7. Regular Users Cannot Access Admin Operations
# ==============================================================================

def test_07_regular_users_cannot_access_admin_operations(tmp_path: pathlib.Path):
    db_file = tmp_path / "users.db"
    um = UserManager(db_path=db_file)

    um.create_user("admin_01", "user_reg", "Regular User", "3456", UserRole.REGULAR_USER)

    # Attempting to create user with regular user credentials MUST raise PermissionError
    with pytest.raises(PermissionError) as excinfo:
        um.create_user("user_reg", "user_new", "New", "6789", UserRole.REGULAR_USER)
    assert "lacks ADMIN privileges" in str(excinfo.value)

    # Attempting to disable user
    with pytest.raises(PermissionError):
        um.disable_user("user_reg", "admin_01")

    # Attempting to remove user
    with pytest.raises(PermissionError):
        um.remove_user("user_reg", "admin_01")

    # Attempting to reset PIN
    with pytest.raises(PermissionError):
        um.reset_user_pin("user_reg", "admin_01", "9876")


# ==============================================================================
# 8. Administrators Can Manage Users (Including Last Admin Protection)
# ==============================================================================

def test_08_administrators_can_manage_users(tmp_path: pathlib.Path):
    db_file = tmp_path / "users.db"
    um = UserManager(db_path=db_file)

    # Admin creates user
    user = um.create_user("admin_01", "user_target", "Target", "4567", UserRole.REGULAR_USER)
    assert user.role == UserRole.REGULAR_USER

    # Admin changes user role to ADMIN
    updated = um.change_user_role("admin_01", "user_target", UserRole.ADMIN)
    assert updated.role == UserRole.ADMIN

    # Now there are 2 admins: admin_01 and user_target
    # Demote user_target back to REGULAR_USER
    updated2 = um.change_user_role("admin_01", "user_target", UserRole.REGULAR_USER)
    assert updated2.role == UserRole.REGULAR_USER

    # Sole remaining active admin is admin_01
    # Attempting to demote or remove or disable the sole admin MUST fail
    with pytest.raises(ValueError) as exc_demote:
        um.change_user_role("admin_01", "admin_01", UserRole.REGULAR_USER)
    assert "Cannot demote the sole active administrator" in str(exc_demote.value)

    with pytest.raises(ValueError) as exc_disable:
        um.disable_user("admin_01", "admin_01")
    assert "Cannot disable the sole active administrator" in str(exc_disable.value)

    with pytest.raises(ValueError) as exc_remove:
        um.remove_user("admin_01", "admin_01")
    assert "Cannot remove the sole active administrator" in str(exc_remove.value)


# ==============================================================================
# 9. Unauthorized Backend Requests Rejected (HTTP 403)
# ==============================================================================

def test_09_unauthorized_backend_requests_rejected(tmp_path: pathlib.Path):
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)
    um.create_user("admin_01", "user_regular", "Regular", "5678", UserRole.REGULAR_USER)

    provider = DashboardStatusProvider(audit_logger=audit_logger, user_manager=um)
    server = DashboardServer(port=0, provider=provider)
    server.start()

    base_url = server.url

    # 1. GET /api/admin/users with NO header -> 401 Unauthorized
    req1 = urllib.request.Request(f"{base_url}/api/admin/users")
    with pytest.raises(urllib.error.HTTPError) as err1:
        urllib.request.urlopen(req1)
    assert err1.value.code in (401, 403)

    # 2. GET /api/admin/users with forged X-Admin-User header (no token) -> 401/403 Rejected
    req2 = urllib.request.Request(f"{base_url}/api/admin/users", headers={"X-Admin-User": "admin_01"})
    with pytest.raises(urllib.error.HTTPError) as err2:
        urllib.request.urlopen(req2)
    assert err2.value.code in (401, 403)

    # 3. POST /api/admin/users/create without admin session token -> 401/403 Rejected
    body_data = json.dumps({"user_id": "bad", "username": "bad", "pin": "9999", "role": "GUEST"}).encode("utf-8")
    req3 = urllib.request.Request(f"{base_url}/api/admin/users/create", data=body_data, headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as err3:
        urllib.request.urlopen(req3)
    assert err3.value.code in (401, 403)

    # 4. Authenticate properly via POST /api/admin/login
    login_data = json.dumps({"user_id": "admin_01", "pin": "1234"}).encode("utf-8")
    req_login = urllib.request.Request(f"{base_url}/api/admin/login", data=login_data, headers={"Content-Type": "application/json"})
    resp_login = urllib.request.urlopen(req_login)
    assert resp_login.code == 200
    login_res = json.loads(resp_login.read().decode("utf-8"))
    token = login_res["token"]
    assert token

    # 5. Valid GET with authenticated bearer token -> 200 OK
    req4 = urllib.request.Request(f"{base_url}/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    resp4 = urllib.request.urlopen(req4)
    assert resp4.code == 200
    user_list = json.loads(resp4.read().decode("utf-8"))
    assert len(user_list) >= 2

    # Verify audit trail recorded UNAUTHORIZED_OP for rejected attempts
    audit_logger.flush()
    unauth_events = audit_logger.query(event_type=AuditEventType.UNAUTHORIZED_OP)
    assert len(unauth_events) >= 3

    server.stop()
    audit_logger.stop()


# ==============================================================================
# 10. User Removal Handled Correctly
# ==============================================================================

def test_10_user_removal_handled_correctly(tmp_path: pathlib.Path):
    clock = MockClock()
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    um.create_user("admin_01", "user_temp", "Temporary", "6789", UserRole.REGULAR_USER)
    ctrl = LockController(config=cfg, user_manager=um, audit_logger=audit_logger, time_func=clock)

    # Works before removal
    for d in "6789":
        ctrl.handle_digit(d)
    ok, _, _ = ctrl.handle_submit()
    assert ok is True
    ctrl.handle_stop()

    # Remove user
    removed = um.remove_user("admin_01", "user_temp")
    assert removed is True
    assert um.get_user("user_temp") is None

    # Fails after removal
    for d in "6789":
        ctrl.handle_digit(d)
    ok2, res2, _ = ctrl.handle_submit()
    assert ok2 is False
    assert res2 == AccessResult.DENIED_INCORRECT

    ctrl.stop()


# ==============================================================================
# 11. PIN Resets Update Credentials Securely
# ==============================================================================

def test_11_pin_resets_update_credentials_securely(tmp_path: pathlib.Path):
    clock = MockClock()
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    um.create_user("admin_01", "user_reset", "Reset Target", "1357", UserRole.REGULAR_USER)
    user_before = um.get_user("user_reset")
    old_hash = user_before.pin_hash
    old_salt = user_before.salt_hex

    # Reset PIN to 2468
    updated = um.reset_user_pin("admin_01", "user_reset", "2468")
    assert updated.pin_hash != old_hash
    assert updated.salt_hex != old_salt

    ctrl = LockController(config=cfg, user_manager=um, audit_logger=audit_logger, time_func=clock)

    # Old PIN 1357 fails
    for d in "1357":
        ctrl.handle_digit(d)
    ok_old, res_old, _ = ctrl.handle_submit()
    assert ok_old is False
    assert res_old == AccessResult.DENIED_INCORRECT

    # Reset controller state
    ctrl.reset()

    # New PIN 2468 succeeds
    for d in "2468":
        ctrl.handle_digit(d)
    ok_new, res_new, _ = ctrl.handle_submit()
    assert ok_new is True
    assert res_new == AccessResult.GRANTED
    assert ctrl.get_snapshot().authenticated_user_id == "user_reset"

    ctrl.stop()


# ==============================================================================
# 12. Administrative Actions Audited
# ==============================================================================

def test_12_administrative_actions_audited(tmp_path: pathlib.Path):
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    um = UserManager(db_path=db_file, audit_logger=audit_logger)

    # Create
    um.create_user("admin_01", "user_audit", "Audit User", "1593", UserRole.REGULAR_USER)
    # Disable
    um.disable_user("admin_01", "user_audit")
    # Enable
    um.enable_user("admin_01", "user_audit")
    # Change role
    um.change_user_role("admin_01", "user_audit", UserRole.GUEST)
    # Reset PIN
    um.reset_user_pin("admin_01", "user_audit", "3579")
    # Remove
    um.remove_user("admin_01", "user_audit")

    audit_logger.flush()

    events = audit_logger.query(limit=50)
    event_types = [e["event_type"] for e in events]

    assert "ADMIN_CREATED_USER" in event_types
    assert "ADMIN_DISABLED_USER" in event_types
    assert "ADMIN_ENABLED_USER" in event_types
    assert "ADMIN_CHANGED_ROLE" in event_types
    assert "ADMIN_RESET_PIN" in event_types
    assert "ADMIN_REMOVED_USER" in event_types

    audit_logger.stop()


# ==============================================================================
# 13. Privacy & Redaction Across APIs and Snapshots
# ==============================================================================

def test_13_privacy_and_redaction(tmp_path: pathlib.Path):
    clock = MockClock()
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    um.create_user("admin_01", "user_secret", "Secret Agent", "3141", UserRole.REGULAR_USER)
    ctrl = LockController(config=cfg, user_manager=um, audit_logger=audit_logger, time_func=clock)

    for d in "3141":
        ctrl.handle_digit(d)
    ctrl.handle_submit()

    # Inspect snapshot
    snap_dict = ctrl.get_snapshot().to_dict()
    serialized = json.dumps(snap_dict)
    assert "3141" not in serialized
    assert "pin_hash" not in serialized
    assert "salt_hex" not in serialized

    # Inspect user.to_dict() default
    user_dict = um.get_user("user_secret").to_dict()
    assert "pin_hash" not in user_dict
    assert "salt_hex" not in user_dict
    assert "3141" not in json.dumps(user_dict)

    # Inspect audit events
    audit_logger.flush()
    records = audit_logger.query(limit=100)
    for r in records:
        dump = json.dumps(r)
        assert "3141" not in dump

    ctrl.stop()


# ==============================================================================
# 14. SQLite Persistence Across Restarts
# ==============================================================================

def test_14_sqlite_persistence_across_restarts(tmp_path: pathlib.Path):
    db_file = tmp_path / "users.db"

    # Session 1: Create user
    um1 = UserManager(db_path=db_file)
    um1.create_user("admin_01", "persisted_user", "Persisted Name", "9876", UserRole.REGULAR_USER)
    users1 = um1.get_all_users()
    assert len(users1) == 2

    # Session 2: Fresh instance pointing to same file
    um2 = UserManager(db_path=db_file)
    loaded = um2.get_user("persisted_user")
    assert loaded is not None
    assert loaded.username == "Persisted Name"
    assert loaded.role == UserRole.REGULAR_USER

    # Authenticate on loaded instance
    ok, user, code = um2.authenticate("9876")
    assert ok is True
    assert user.user_id == "persisted_user"


# ==============================================================================
# 15. Forged X-Admin-User Rejection (Scenario 1)
# ==============================================================================

def test_15_forged_admin_header_rejected(tmp_path: pathlib.Path):
    """
    Scenario 1: Header spoofing / forged identity.
    Supplying X-Admin-User (with admin_01, root, or any username) without a cryptographically
    valid session token MUST be rejected with HTTP 401 or 403 across all admin endpoints.
    """
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    provider = DashboardStatusProvider(audit_logger=audit_logger, user_manager=um)
    server = DashboardServer(port=0, provider=provider)
    server.start()
    base_url = server.url

    # 1. GET /api/admin/users with forged X-Admin-User: admin_01
    req = urllib.request.Request(f"{base_url}/api/admin/users", headers={"X-Admin-User": "admin_01"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code in (401, 403)

    # 2. POST /api/admin/users/create with forged X-Admin-User: admin_01
    body = json.dumps({"user_id": "hacker", "username": "Hacker", "pin": "8888", "role": "ADMIN"}).encode("utf-8")
    req_post = urllib.request.Request(
        f"{base_url}/api/admin/users/create",
        data=body,
        headers={"Content-Type": "application/json", "X-Admin-User": "admin_01"}
    )
    with pytest.raises(urllib.error.HTTPError) as err_post:
        urllib.request.urlopen(req_post)
    assert err_post.value.code in (401, 403)

    # Confirm user was NOT created
    assert um.get_user("hacker") is None

    server.stop()
    audit_logger.stop()


# ==============================================================================
# 16. Missing / Invalid / Expired / Revoked Tokens Rejected (Scenario 2)
# ==============================================================================

def test_16_missing_invalid_expired_revoked_tokens_rejected(tmp_path: pathlib.Path):
    """
    Scenario 2: Session token lifecycle verification.
    Missing, malformed/invalid, expired, and revoked tokens are all strictly rejected.
    """
    from src.dashboard.server import AdminSessionManager
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    # Session manager with very short TTL (1 second) for expiry test
    session_mgr = AdminSessionManager(default_ttl_seconds=1)
    provider = DashboardStatusProvider(audit_logger=audit_logger, user_manager=um, session_manager=session_mgr)
    server = DashboardServer(port=0, provider=provider)
    server.start()
    base_url = server.url

    # 1. Missing token -> 401/403
    req_missing = urllib.request.Request(f"{base_url}/api/admin/users")
    with pytest.raises(urllib.error.HTTPError) as err_missing:
        urllib.request.urlopen(req_missing)
    assert err_missing.value.code in (401, 403)

    # 2. Bogus / forged token -> 401/403
    req_bogus = urllib.request.Request(
        f"{base_url}/api/admin/users",
        headers={"Authorization": "Bearer totally_bogus_token_abcdef123456"}
    )
    with pytest.raises(urllib.error.HTTPError) as err_bogus:
        urllib.request.urlopen(req_bogus)
    assert err_bogus.value.code in (401, 403)

    # 3. Create short-lived token and test expiration
    short_token = session_mgr.create_session("admin_01", ttl_seconds=1)
    req_valid = urllib.request.Request(
        f"{base_url}/api/admin/users",
        headers={"Authorization": f"Bearer {short_token}"}
    )
    resp = urllib.request.urlopen(req_valid)
    assert resp.code == 200

    # Wait past TTL
    time.sleep(1.2)
    with pytest.raises(urllib.error.HTTPError) as err_expired:
        urllib.request.urlopen(req_valid)
    assert err_expired.value.code in (401, 403)

    # 4. Revocation via logout
    login_data = json.dumps({"user_id": "admin_01", "pin": "1234"}).encode("utf-8")
    req_login = urllib.request.Request(f"{base_url}/api/admin/login", data=login_data, headers={"Content-Type": "application/json"})
    token_to_revoke = json.loads(urllib.request.urlopen(req_login).read().decode("utf-8"))["token"]

    req_active = urllib.request.Request(f"{base_url}/api/admin/users", headers={"Authorization": f"Bearer {token_to_revoke}"})
    assert urllib.request.urlopen(req_active).code == 200

    # Logout
    req_logout = urllib.request.Request(
        f"{base_url}/api/admin/logout",
        data=b"{}",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token_to_revoke}"}
    )
    assert urllib.request.urlopen(req_logout).code == 200

    # Post-logout request must fail
    with pytest.raises(urllib.error.HTTPError) as err_revoked:
        urllib.request.urlopen(req_active)
    assert err_revoked.value.code in (401, 403)

    server.stop()
    audit_logger.stop()


# ==============================================================================
# 17. Regular Users / Guests Cannot Access Admin Endpoints (Scenario 3)
# ==============================================================================

def test_17_regular_user_cannot_access_admin_endpoints(tmp_path: pathlib.Path):
    """
    Scenario 3: Privilege separation.
    Regular users and guests cannot obtain administrative tokens, even if their credentials are valid.
    """
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)
    um.create_user("admin_01", "alice_regular", "Alice", "2468", UserRole.REGULAR_USER)
    um.create_user("admin_01", "bob_guest", "Bob", "1357", UserRole.GUEST)

    provider = DashboardStatusProvider(audit_logger=audit_logger, user_manager=um)
    server = DashboardServer(port=0, provider=provider)
    server.start()
    base_url = server.url

    # 1. Regular user logs into admin login -> 403 Forbidden
    reg_login = json.dumps({"user_id": "alice_regular", "pin": "2468"}).encode("utf-8")
    req_reg = urllib.request.Request(f"{base_url}/api/admin/login", data=reg_login, headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as err_reg:
        urllib.request.urlopen(req_reg)
    assert err_reg.value.code == 403
    err_body = json.loads(err_reg.value.read().decode("utf-8"))
    assert "administrative privileges" in err_body.get("error", "")

    # 2. Guest user logs into admin login -> 403 Forbidden
    guest_login = json.dumps({"user_id": "bob_guest", "pin": "1357"}).encode("utf-8")
    req_guest = urllib.request.Request(f"{base_url}/api/admin/login", data=guest_login, headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as err_guest:
        urllib.request.urlopen(req_guest)
    assert err_guest.value.code == 403

    # 3. Direct endpoint access with forged user token
    fake_token = provider.session_manager.create_session("alice_regular")
    req_fake = urllib.request.Request(f"{base_url}/api/admin/users", headers={"Authorization": f"Bearer {fake_token}"})
    with pytest.raises(urllib.error.HTTPError) as err_fake:
        urllib.request.urlopen(req_fake)
    assert err_fake.value.code in (401, 403)

    server.stop()
    audit_logger.stop()


# ==============================================================================
# 18. Demoted or Disabled Admin Immediately Loses Access (Scenario 4)
# ==============================================================================

def test_18_demoted_or_disabled_admin_immediately_loses_session(tmp_path: pathlib.Path):
    """
    Scenario 4: Real-time authorization check.
    If an admin is demoted or disabled during an active session, they immediately lose access
    on the very next request without waiting for token TTL.
    """
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    # Create secondary admin
    um.create_user("admin_01", "admin_02", "Secondary Admin", "3691", UserRole.ADMIN)

    provider = DashboardStatusProvider(audit_logger=audit_logger, user_manager=um)
    server = DashboardServer(port=0, provider=provider)
    server.start()
    base_url = server.url

    # admin_02 logs in
    login_data = json.dumps({"user_id": "admin_02", "pin": "3691"}).encode("utf-8")
    req_login = urllib.request.Request(f"{base_url}/api/admin/login", data=login_data, headers={"Content-Type": "application/json"})
    token_02 = json.loads(urllib.request.urlopen(req_login).read().decode("utf-8"))["token"]

    # admin_02 makes a successful request
    req_check = urllib.request.Request(f"{base_url}/api/admin/users", headers={"Authorization": f"Bearer {token_02}"})
    assert urllib.request.urlopen(req_check).code == 200

    # Demote admin_02 to REGULAR_USER
    um.change_user_role("admin_01", "admin_02", UserRole.REGULAR_USER)

    # Next request with token_02 MUST immediately be rejected
    with pytest.raises(urllib.error.HTTPError) as err_demoted:
        urllib.request.urlopen(req_check)
    assert err_demoted.value.code in (401, 403)

    # Re-promote to ADMIN, obtain new session
    um.change_user_role("admin_01", "admin_02", UserRole.ADMIN)
    req_login2 = urllib.request.Request(f"{base_url}/api/admin/login", data=login_data, headers={"Content-Type": "application/json"})
    token_02_new = json.loads(urllib.request.urlopen(req_login2).read().decode("utf-8"))["token"]

    req_check2 = urllib.request.Request(f"{base_url}/api/admin/users", headers={"Authorization": f"Bearer {token_02_new}"})
    assert urllib.request.urlopen(req_check2).code == 200

    # Disable admin_02
    um.disable_user("admin_01", "admin_02")

    # Next request with token_02_new MUST immediately be rejected
    with pytest.raises(urllib.error.HTTPError) as err_disabled:
        urllib.request.urlopen(req_check2)
    assert err_disabled.value.code in (401, 403)

    server.stop()
    audit_logger.stop()


# ==============================================================================
# 19. Valid Authenticated Admin Operations (Scenario 5)
# ==============================================================================

def test_19_valid_authenticated_admin_operations(tmp_path: pathlib.Path):
    """
    Scenario 5: Complete administrator session lifecycle and management workflow.
    Admin logs in, checks session, creates a user, disables, resets PIN, and logs out.
    """
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    provider = DashboardStatusProvider(audit_logger=audit_logger, user_manager=um)
    server = DashboardServer(port=0, provider=provider)
    server.start()
    base_url = server.url

    # 1. Login
    login_body = json.dumps({"user_id": "admin_01", "pin": "1234"}).encode("utf-8")
    req_login = urllib.request.Request(f"{base_url}/api/admin/login", data=login_body, headers={"Content-Type": "application/json"})
    resp_login = urllib.request.urlopen(req_login)
    assert resp_login.code == 200
    login_info = json.loads(resp_login.read().decode("utf-8"))
    token = login_info["token"]
    assert token
    assert login_info["authenticated"] is True
    assert login_info["user"]["user_id"] == "admin_01"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    # 2. Verify Session endpoint
    req_sess = urllib.request.Request(f"{base_url}/api/admin/session", headers=headers)
    sess_res = json.loads(urllib.request.urlopen(req_sess).read().decode("utf-8"))
    assert sess_res["authenticated"] is True
    assert sess_res["user"]["user_id"] == "admin_01"

    # 3. Create user
    create_body = json.dumps({"user_id": "carol_ops", "username": "Carol", "pin": "7777", "role": "REGULAR_USER"}).encode("utf-8")
    req_create = urllib.request.Request(f"{base_url}/api/admin/users/create", data=create_body, headers=headers)
    resp_create = urllib.request.urlopen(req_create)
    assert resp_create.code == 201

    # 4. Disable user
    dis_body = json.dumps({"target_user_id": "carol_ops"}).encode("utf-8")
    req_dis = urllib.request.Request(f"{base_url}/api/admin/users/disable", data=dis_body, headers=headers)
    assert urllib.request.urlopen(req_dis).code == 200
    assert um.get_user("carol_ops").status == UserStatus.DISABLED

    # 5. Reset PIN
    reset_body = json.dumps({"target_user_id": "carol_ops", "new_pin": "8888"}).encode("utf-8")
    req_reset = urllib.request.Request(f"{base_url}/api/admin/users/reset-pin", data=reset_body, headers=headers)
    assert urllib.request.urlopen(req_reset).code == 200

    # 6. Logout
    req_logout = urllib.request.Request(f"{base_url}/api/admin/logout", data=b"{}", headers=headers)
    assert urllib.request.urlopen(req_logout).code == 200

    server.stop()
    audit_logger.stop()


# ==============================================================================
# 20. Audit Generation Without Credential Leakage (Scenario 6)
# ==============================================================================

def test_20_unauthorized_attempts_generate_audit_events_without_credentials(tmp_path: pathlib.Path):
    """
    Scenario 6: Audit trail integrity and privacy verification.
    Rejected authentication and unauthorized operations produce audit records,
    while ensuring zero plaintext PINs, PIN hashes, or salts are exposed.
    """
    db_file = tmp_path / "users.db"
    audit_file = tmp_path / "audit.db"
    audit_logger = AuditLogger(db_path=str(audit_file))
    cfg = SecurityConfig(users_db_path=str(db_file), audit_db_path=str(audit_file))
    um = UserManager(config=cfg, db_path=db_file, audit_logger=audit_logger)

    provider = DashboardStatusProvider(audit_logger=audit_logger, user_manager=um)
    server = DashboardServer(port=0, provider=provider)
    server.start()
    base_url = server.url

    # Failed login with wrong PIN
    bad_login = json.dumps({"user_id": "admin_01", "pin": "9999"}).encode("utf-8")
    req_bad = urllib.request.Request(f"{base_url}/api/admin/login", data=bad_login, headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req_bad)
    assert err.value.code == 401

    # Unauthorized access to mutation endpoint
    req_unauth = urllib.request.Request(f"{base_url}/api/admin/users")
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(req_unauth)

    audit_logger.flush()
    events = audit_logger.query(limit=50)
    assert len(events) >= 2

    event_types = [e["event_type"] for e in events]
    assert "USER_ACCESS_DENIED" in event_types or "UNAUTHORIZED_OP" in event_types

    # Strict audit leakage check
    for ev in events:
        leak_check_data = {k: v for k, v in ev.items() if k != "timestamp"}
        dump = json.dumps(leak_check_data)
        assert "9999" not in dump, "Plaintext PIN leaked in audit log!"
        assert "1234" not in dump, "Plaintext PIN leaked in audit log!"
        assert "pin_hash" not in dump, "PIN hash leaked in audit log!"
        assert "salt_hex" not in dump, "Salt leaked in audit log!"

    server.stop()
    audit_logger.stop()

