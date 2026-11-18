"""
tests/test_lock_security.py
---------------------------
Automated Verification Test Suite for Voice-Based Smart Access Control & PIN Authentication.

Audited and Verified against:
1. Initial state is LOCKED.
2. Correct PIN grants access.
3. Incorrect PIN denies access.
4. Incomplete PIN is rejected.
5. Only numeric digits 1-9 are accepted (digit 0 rejected).
6. PIN buffer cannot exceed configured length.
7. PIN timeout clears the buffer.
8. Three failures activate LOCKOUT.
9. Authentication is rejected during LOCKOUT.
10. Lockout expires correctly.
11. Automatic relocking works.
12. STOP clears PIN entry.
13. STOP cancels pending unlock requests.
14. Restart initializes the lock in LOCKED state.
15. PIN is not stored in plaintext (salt + PBKDF2 hash only).
16. Duplicate voice events do not cause duplicate PIN digits.
17. PIN digit policy: digit 0 rejected in provisioning and entry.
18. Restart persistence: credentials reload from salted JSON without plaintext.
19. Redaction: entered PIN digits and credentials never leak to snapshots or APIs.
"""

import json
import pathlib
import threading
import time
from typing import List
import pytest

from src.security.config import SecurityConfig
from src.security.pin_manager import PINManager
from src.security.lock_controller import (
    AccessResult,
    LockController,
    LockState,
    LockStateSnapshot,
)
from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig


class MockClock:
    """Controllable clock for deterministic, zero-sleep timer testing."""

    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float):
        self.current_time += seconds


# ==============================================================================
# Scenario 1: Initial state is LOCKED
# ==============================================================================

def test_scenario_01_initial_state_is_locked():
    clock = MockClock()
    ctrl = LockController(time_func=clock)
    snap = ctrl.get_snapshot()

    assert snap.state == LockState.LOCKED
    assert snap.is_locked is True
    assert snap.is_unlocked is False
    assert snap.is_lockout is False
    assert snap.digits_entered == 0
    assert snap.failed_attempts == 0


# ==============================================================================
# Scenario 2: Correct PIN grants access
# ==============================================================================

def test_scenario_02_correct_pin_grants_access():
    clock = MockClock()
    ctrl = LockController(time_func=clock)

    for digit in "1234":
        assert ctrl.handle_digit(digit) is True
    assert ctrl.state == LockState.PIN_ENTRY

    success, result, msg = ctrl.handle_submit()
    assert success is True
    assert result == AccessResult.GRANTED
    assert ctrl.state == LockState.UNLOCKED
    assert ctrl.get_snapshot().is_unlocked is True
    assert ctrl.failed_attempts == 0
    assert ctrl.pin_manager.is_empty is True


# ==============================================================================
# Scenario 3: Incorrect PIN denies access
# ==============================================================================

def test_scenario_03_incorrect_pin_denies_access():
    clock = MockClock()
    ctrl = LockController(time_func=clock)

    for digit in "9999":
        ctrl.handle_digit(digit)

    success, result, msg = ctrl.handle_submit()
    assert success is False
    assert result == AccessResult.DENIED_INCORRECT
    assert ctrl.state == LockState.LOCKED
    assert ctrl.failed_attempts == 1
    assert ctrl.pin_manager.is_empty is True


# ==============================================================================
# Scenario 4: Incomplete PIN is rejected
# ==============================================================================

def test_scenario_04_incomplete_pin_is_rejected():
    clock = MockClock()
    ctrl = LockController(time_func=clock)

    # Enter only 2 digits
    ctrl.handle_digit("1")
    ctrl.handle_digit("2")
    assert ctrl.get_snapshot().digits_entered == 2

    # Submit early
    success, result, msg = ctrl.handle_submit()
    assert success is False
    assert result == AccessResult.DENIED_INCOMPLETE
    assert ctrl.state == LockState.LOCKED
    assert ctrl.failed_attempts == 1
    assert ctrl.pin_manager.is_empty is True


# ==============================================================================
# Scenario 5: Only numeric digits 1-9 are accepted (Digit '0' is rejected)
# ==============================================================================

def test_scenario_05_only_digits_1_to_9_accepted():
    clock = MockClock()
    pm = PINManager(time_func=clock)

    # Valid numeric digits 1 through 9
    for d in "123456789":
        pm.clear()
        assert pm.append_digit(d) is True

    # Digit '0' must be strictly rejected per Part B policy
    pm.clear()
    assert pm.append_digit("0") is False

    # Invalid non-digits
    for bad in ["a", "Z", "#", "12", "", " ", None]:
        assert pm.append_digit(bad) is False  # type: ignore


# ==============================================================================
# Scenario 6: PIN buffer cannot exceed configured length
# ==============================================================================

def test_scenario_06_pin_buffer_cannot_exceed_configured_length():
    clock = MockClock()
    ctrl = LockController(time_func=clock)

    # Fill buffer to capacity (4 digits)
    for d in "1234":
        assert ctrl.handle_digit(d) is True
    assert ctrl.pin_manager.current_length == 4
    assert ctrl.pin_manager.is_full is True

    # Attempt to append 5th digit
    assert ctrl.handle_digit("5") is False
    assert ctrl.pin_manager.current_length == 4


# ==============================================================================
# Scenario 7: PIN timeout clears the buffer
# ==============================================================================

def test_scenario_07_pin_timeout_clears_the_buffer():
    clock = MockClock(start_time=100.0)
    config = SecurityConfig(pin_length=4, pin_timeout_s=10.0)
    ctrl = LockController(config=config, time_func=clock)

    ctrl.handle_digit("1")
    ctrl.handle_digit("2")
    assert ctrl.state == LockState.PIN_ENTRY
    assert ctrl.pin_manager.current_length == 2

    # Advance 5s (less than 10s timeout)
    clock.advance(5.0)
    ctrl.tick()
    assert ctrl.state == LockState.PIN_ENTRY
    assert ctrl.pin_manager.current_length == 2

    # Advance beyond 10s timeout (additional 6s = 11s elapsed)
    clock.advance(6.0)
    ctrl.tick()
    assert ctrl.state == LockState.LOCKED
    assert ctrl.pin_manager.is_empty is True
    assert ctrl.last_access_result == AccessResult.TIMEOUT


# ==============================================================================
# Scenario 8: Three failures activate LOCKOUT
# ==============================================================================

def test_scenario_08_three_failures_activate_lockout():
    clock = MockClock()
    config = SecurityConfig(max_failed_attempts=3)
    ctrl = LockController(config=config, time_func=clock)

    # Attempt 1: Fail with invalid PIN '9999'
    for d in "9999": ctrl.handle_digit(d)
    ctrl.handle_submit()
    assert ctrl.state == LockState.LOCKED
    assert ctrl.failed_attempts == 1

    # Attempt 2: Fail with invalid PIN '8888'
    for d in "8888": ctrl.handle_digit(d)
    ctrl.handle_submit()
    assert ctrl.state == LockState.LOCKED
    assert ctrl.failed_attempts == 2

    # Attempt 3: Fail with invalid PIN '7777' -> Triggers LOCKOUT
    for d in "7777": ctrl.handle_digit(d)
    success, result, msg = ctrl.handle_submit()
    assert success is False
    assert result == AccessResult.DENIED_LOCKOUT
    assert ctrl.state == LockState.LOCKOUT
    assert ctrl.get_snapshot().is_lockout is True


# ==============================================================================
# Scenario 9: Authentication is rejected during LOCKOUT
# ==============================================================================

def test_scenario_09_authentication_is_rejected_during_lockout():
    clock = MockClock()
    config = SecurityConfig(max_failed_attempts=3, lockout_duration_s=30.0)
    ctrl = LockController(config=config, time_func=clock)

    # Trigger lockout
    for _ in range(3):
        for d in "9999": ctrl.handle_digit(d)
        ctrl.handle_submit()
    assert ctrl.state == LockState.LOCKOUT

    # Digits are rejected
    assert ctrl.handle_digit("1") is False

    # Submission is rejected
    success, result, msg = ctrl.handle_submit()
    assert success is False
    assert result == AccessResult.DENIED_LOCKOUT


# ==============================================================================
# Scenario 10: Lockout expires correctly
# ==============================================================================

def test_scenario_10_lockout_expires_correctly():
    clock = MockClock(start_time=500.0)
    config = SecurityConfig(max_failed_attempts=3, lockout_duration_s=30.0)
    ctrl = LockController(config=config, time_func=clock)

    # Trigger lockout
    for _ in range(3):
        for d in "9999": ctrl.handle_digit(d)
        ctrl.handle_submit()
    assert ctrl.state == LockState.LOCKOUT

    # Advance 20 seconds (lockout still active)
    clock.advance(20.0)
    ctrl.tick()
    assert ctrl.state == LockState.LOCKOUT
    assert ctrl.get_snapshot().lockout_remaining_s == pytest.approx(10.0, 0.1)

    # Advance 11 seconds (31s elapsed > 30s)
    clock.advance(11.0)
    ctrl.tick()
    assert ctrl.state == LockState.LOCKED
    assert ctrl.failed_attempts == 0
    assert ctrl.get_snapshot().is_lockout is False

    # New authentication succeeds after lockout expires
    for d in "1234": ctrl.handle_digit(d)
    success, _, _ = ctrl.handle_submit()
    assert success is True
    assert ctrl.state == LockState.UNLOCKED


# ==============================================================================
# Scenario 11: Automatic relocking works
# ==============================================================================

def test_scenario_11_automatic_relocking_works():
    clock = MockClock(start_time=100.0)
    config = SecurityConfig(unlock_duration_s=5.0)
    ctrl = LockController(config=config, time_func=clock)

    for d in "1234": ctrl.handle_digit(d)
    ctrl.handle_submit()
    assert ctrl.state == LockState.UNLOCKED

    # Advance 3s (relock pending)
    clock.advance(3.0)
    ctrl.tick()
    assert ctrl.state == LockState.UNLOCKED
    assert ctrl.get_snapshot().unlock_remaining_s == pytest.approx(2.0, 0.1)

    # Advance 2.5s (5.5s total > 5.0s unlock duration)
    clock.advance(2.5)
    ctrl.tick()
    assert ctrl.state == LockState.LOCKED
    assert ctrl.get_snapshot().is_locked is True
    assert ctrl.last_access_result == AccessResult.RELOCKED_AUTO


# ==============================================================================
# Scenario 12: STOP clears PIN entry
# ==============================================================================

def test_scenario_12_stop_clears_pin_entry():
    clock = MockClock()
    ctrl = LockController(time_func=clock)

    ctrl.handle_digit("1")
    ctrl.handle_digit("2")
    assert ctrl.state == LockState.PIN_ENTRY
    assert ctrl.pin_manager.current_length == 2

    # Issue stop command
    ctrl.handle_command("stop")
    assert ctrl.state == LockState.LOCKED
    assert ctrl.pin_manager.is_empty is True
    assert ctrl.last_access_result == AccessResult.CLEARED


# ==============================================================================
# Scenario 13: STOP cancels pending unlock requests
# ==============================================================================

def test_scenario_13_stop_cancels_pending_unlock():
    clock = MockClock()
    ctrl = LockController(time_func=clock)

    for d in "1234": ctrl.handle_digit(d)
    ctrl.handle_submit()
    assert ctrl.state == LockState.UNLOCKED

    # User speaks stop while unlocked
    ctrl.handle_command("stop")
    assert ctrl.state == LockState.LOCKED
    assert ctrl.get_snapshot().is_locked is True
    assert ctrl.get_snapshot().unlock_remaining_s == 0.0


# ==============================================================================
# Scenario 14: Restart initializes the lock in LOCKED state
# ==============================================================================

def test_scenario_14_restart_initializes_lock_in_locked_state():
    config = OrchestratorConfig(enable_dashboard=False)
    app = VoiceRecognitionApp(config=config)

    # Simulate unlocking
    for d in "1234":
        app.lock_controller.handle_digit(d)
    app.lock_controller.handle_submit()
    assert app.lock_controller.state == LockState.UNLOCKED

    # Restart application
    app.restart(use_microphone=False)
    snap = app.lock_controller.get_snapshot()

    assert snap.state == LockState.LOCKED
    assert snap.is_locked is True
    assert snap.is_unlocked is False
    assert snap.digits_entered == 0
    app.stop()


# ==============================================================================
# Scenario 15: PIN is not stored in plaintext (Salt + Hash storage verified)
# ==============================================================================

def test_scenario_15_pin_is_not_stored_in_plaintext():
    pm = PINManager()

    # 1. Inspect internal attributes
    assert pm.is_plaintext_stored() is False

    # 2. Check salt and hash existence
    assert len(pm._salt_hex) == 32  # 16 bytes = 32 hex chars
    assert len(pm._hash_hex) == 64  # sha256 = 32 bytes = 64 hex chars
    assert pm._salt_hex != "1234"
    assert pm._hash_hex != "1234"

    # 3. Masked representation does not leak plaintext
    pm.append_digit("1")
    pm.append_digit("2")
    assert pm.masked_display == "**"
    assert "1" not in pm.masked_display
    assert "2" not in pm.masked_display

    # 4. Verify credentials file stores only cryptographic hash & salt
    cred_file = pathlib.Path("configs/security_credentials.json")
    assert cred_file.exists(), "configs/security_credentials.json must exist"
    with open(cred_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "salt_hex" in data
    assert "hash_hex" in data
    assert "1234" not in json.dumps(data)

    # 5. Verify default.yaml contains no plaintext PIN
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        yaml_content = f.read()
    assert "default_pin" not in yaml_content


# ==============================================================================
# Scenario 16: Duplicate voice events do not cause duplicate PIN digits
# ==============================================================================

def test_scenario_16_duplicate_voice_events_do_not_duplicate_digits():
    clock = MockClock()
    ctrl = LockController(time_func=clock)

    # Event 1: First arrival of command record with ID 101
    snap1 = ctrl.handle_command("one", command_id=101)
    assert snap1.digits_entered == 1

    # Event 2: Duplicate delivery of same command record ID 101
    snap2 = ctrl.handle_command("one", command_id=101)
    assert snap2.digits_entered == 1  # Not duplicated!

    # Event 3: Genuinely new distinct command record with ID 102
    snap3 = ctrl.handle_command("two", command_id=102)
    assert snap3.digits_entered == 2


# ==============================================================================
# Scenario 17: PART B — Digit '0' policy enforcement
# ==============================================================================

def test_part_b_pin_digit_policy_rejects_zero():
    pm = PINManager()

    # 1. Provisioning must reject any PIN containing '0'
    for invalid_pin in ["0123", "1034", "1204", "1230", "0000"]:
        with pytest.raises(ValueError, match="Only digits 1 through 9 are permitted"):
            pm.provision_pin(invalid_pin)

    # 2. Append digit must reject '0'
    pm.clear()
    assert pm.append_digit("0") is False
    assert pm.current_length == 0

    # 3. LockController must reject digit '0' and spoken 'zero'
    ctrl = LockController()
    assert ctrl.handle_digit("0") is False
    snap = ctrl.handle_command("zero")
    assert snap.digits_entered == 0
    assert snap.state == LockState.LOCKED


# ==============================================================================
# Scenario 18: Credentials Persistence Across Restart
# ==============================================================================

def test_part_a_credentials_persistence_across_restart():
    # Instantiate fresh PINManager without passing any plaintext PIN
    config = SecurityConfig(credentials_path="configs/security_credentials.json")
    pm = PINManager(config=config)

    # Enter demo PIN '1234'
    for d in "1234":
        pm.append_digit(d)
    valid, reason = pm.verify()
    assert valid is True
    assert reason == "CORRECT_PIN"

    # Enter incorrect PIN '5678'
    for d in "5678":
        pm.append_digit(d)
    valid_bad, reason_bad = pm.verify()
    assert valid_bad is False
    assert reason_bad == "INCORRECT_PIN"


# ==============================================================================
# Scenario 19: Sensitive Data Redaction from Dashboard Snapshot & API
# ==============================================================================

def test_part_a_snapshot_and_api_redacts_credentials():
    ctrl = LockController()

    # Enter digits via voice commands
    ctrl.handle_command("one")
    ctrl.handle_command("two")

    snap = ctrl.get_snapshot()
    snap_dict = snap.to_dict()

    # Verify no raw digits or salt/hash leak
    assert snap_dict["last_command"] == "[DIGIT_ENTERED]"
    assert "salt_hex" not in snap_dict
    assert "hash_hex" not in snap_dict
    assert "1" not in snap_dict["masked_entry"]
    assert "2" not in snap_dict["masked_entry"]
    assert snap_dict["masked_entry"] == "**"


# ==============================================================================
# Integration Test: Spoken Command Stream ('one', 'two', 'three', 'four', 'go')
# ==============================================================================

def test_voice_command_stream_unlock_cycle():
    clock = MockClock(start_time=2000.0)
    config = SecurityConfig(unlock_duration_s=5.0)
    ctrl = LockController(config=config, time_func=clock)

    # Speak digits 'one', 'two', 'three', 'four'
    for cid, word in enumerate(["one", "two", "three", "four"], start=1):
        snap = ctrl.handle_command(word, command_id=cid)
        assert snap.state == LockState.PIN_ENTRY

    assert ctrl.get_snapshot().digits_entered == 4
    assert ctrl.get_snapshot().masked_entry == "****"

    # Speak 'go'
    snap_go = ctrl.handle_command("go", command_id=5)
    assert snap_go.state == LockState.UNLOCKED
    assert snap_go.last_access_result == AccessResult.GRANTED

    # Advance time to test auto-relock
    clock.advance(5.5)
    ctrl.tick()
    assert ctrl.state == LockState.LOCKED
    assert ctrl.last_access_result == AccessResult.RELOCKED_AUTO


# ==============================================================================
# Thread Safety Test: Concurrent Digit Appending & Verification
# ==============================================================================

def test_lock_controller_thread_safety():
    ctrl = LockController()

    def worker(d: str):
        for _ in range(50):
            ctrl.handle_digit(d)
            time.sleep(0.0001)
            ctrl.tick()

    threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(1, 5)]
    for t in threads: t.start()
    for t in threads: t.join()

    # State must remain valid and bounded
    snap = ctrl.get_snapshot()
    assert snap.digits_entered <= 4
    assert snap.state in (LockState.LOCKED, LockState.PIN_ENTRY)
