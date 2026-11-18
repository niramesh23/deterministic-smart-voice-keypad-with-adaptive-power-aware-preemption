"""
src/security/lock_controller.py
-------------------------------
Software-Simulated Voice-Based Smart Lock Controller and Finite-State Machine.

States:
- LOCKED: Default startup state. Barrier active. Awaiting voice authentication.
- PIN_ENTRY: User is speaking digits; digits are appended to the PINManager buffer.
- UNLOCKED: Access granted following successful PIN verification. Auto-relock timer active.
- LOCKOUT: Temporary lockout active after 3 consecutive failed authentication attempts.

Responsibilities:
- Enforce strict state transition rules and reject unauthorized bypasses.
- Guarantee startup state is ALWAYS LOCKED (never UNLOCKED after restart).
- Coordinate PINManager for salted cryptographic authentication.
- Handle automatic relocking after configured unlock duration (default: 5.0s).
- Handle temporary security lockout after consecutive failures (default: 30.0s).
- Handle PIN entry inactivity timeout (default: 10.0s).
- Prevent duplicate command processing across overlapping inference windows.
- Provide thread-safe state snapshots and telemetry callbacks.
"""

from collections import deque
from dataclasses import dataclass, field
import datetime
from enum import Enum
import hashlib
import hmac
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

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

logger = logging.getLogger("rtos.security.lock")


class LockState(str, Enum):
    """Documented states of the simulated Smart Lock FSM."""
    LOCKED = "LOCKED"
    PIN_ENTRY = "PIN_ENTRY"
    UNLOCKED = "UNLOCKED"
    LOCKOUT = "LOCKOUT"


class AccessResult(str, Enum):
    """Categorized outcome of access requests and PIN evaluations."""
    NONE = "NONE"
    GRANTED = "GRANTED"
    DENIED_INCORRECT = "DENIED_INCORRECT"
    DENIED_INCOMPLETE = "DENIED_INCOMPLETE"
    DENIED_LOCKOUT = "DENIED_LOCKOUT"
    CLEARED = "CLEARED"
    TIMEOUT = "TIMEOUT"
    RELOCKED_AUTO = "RELOCKED_AUTO"


@dataclass(frozen=True)
class LockStateSnapshot:
    """Immutable, thread-safe snapshot of lock controller telemetry and status."""
    state: LockState
    is_locked: bool
    is_unlocked: bool
    is_lockout: bool
    digits_entered: int
    pin_length: int
    masked_entry: str
    pin_progress: str
    failed_attempts: int
    max_failed_attempts: int
    lockout_remaining_s: float
    unlock_remaining_s: float
    last_access_result: AccessResult
    last_access_message: str
    last_command: Optional[str]
    last_state_change_time: float
    operational_status: str
    authenticated_user_id: Optional[str] = None
    authenticated_username: Optional[str] = None
    authenticated_role: Optional[str] = None
    emergency_active: bool = False
    pending_cancelled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes snapshot to JSON-compatible dictionary for dashboard/REST API."""
        return {
            "state": self.state.value,
            "is_locked": self.is_locked,
            "is_unlocked": self.is_unlocked,
            "is_lockout": self.is_lockout,
            "digits_entered": self.digits_entered,
            "pin_length": self.pin_length,
            "masked_entry": self.masked_entry,
            "pin_progress": self.pin_progress,
            "failed_attempts": self.failed_attempts,
            "max_failed_attempts": self.max_failed_attempts,
            "lockout_remaining_s": round(self.lockout_remaining_s, 1),
            "unlock_remaining_s": round(self.unlock_remaining_s, 1),
            "last_access_result": self.last_access_result.value,
            "last_access_message": self.last_access_message,
            "last_command": self.last_command,
            "last_state_change_time": round(self.last_state_change_time, 3),
            "operational_status": self.operational_status,
            "authenticated_user_id": self.authenticated_user_id,
            "authenticated_username": self.authenticated_username,
            "authenticated_role": self.authenticated_role,
            "emergency_active": self.emergency_active,
            "pending_cancelled": self.pending_cancelled,
        }


class LockController:
    """
    Simulated Smart Lock Controller coordinating state transitions,
    voice command routing, auto-relocking, tamper lockout, and persistent audit logging.
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        pin_manager: Optional[PINManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        user_manager: Optional[UserManager] = None,
        time_func: Optional[Callable[[], float]] = None,
        on_state_change: Optional[Callable[[LockStateSnapshot], None]] = None,
        voice_feedback: Optional[Any] = None,
    ):
        self.config = config or SecurityConfig()
        self._time_func = time_func or time.monotonic
        self._on_state_change = on_state_change
        self.voice_feedback = voice_feedback
        self._lock = threading.RLock()

        # Audit Logger instance
        if audit_logger is not None:
            self.audit_logger: Optional[AuditLogger] = audit_logger
        elif self.config.enable_audit_logging:
            self.audit_logger = AuditLogger(
                db_path=self.config.audit_db_path,
                queue_capacity=self.config.audit_queue_capacity,
            )
        else:
            self.audit_logger = None

        # PIN Manager instance
        self.pin_manager = pin_manager or PINManager(config=self.config, time_func=self._time_func)

        # User Manager instance (RBAC)
        if user_manager is not None:
            self.user_manager: Optional[UserManager] = user_manager
        else:
            self.user_manager = UserManager(
                config=self.config,
                audit_logger=self.audit_logger,
                db_path=self.config.users_db_path,
                time_func=self._time_func,
            )

        self.authenticated_user: Optional[UserRecord] = None

        # Core FSM state (Default startup state MUST be LOCKED)
        self.state: LockState = LockState.LOCKED
        self.failed_attempts: int = 0

        # Timers
        self._unlock_timestamp: float = 0.0
        self._lockout_timestamp: float = 0.0
        self._last_state_change_time: float = self._time_func()

        # Historical status tracking
        self.last_access_result: AccessResult = AccessResult.NONE
        self.last_access_message: str = "System initialized in LOCKED state."
        self.last_command: Optional[str] = None

        # Operation sequencing & Emergency interlock state
        self._operation_id: int = 0
        self._emergency_active: bool = False
        self._pending_cancelled: bool = False
        self._pending_unlock_request: Optional[Dict[str, Any]] = None

        # Processed command ID cache to prevent duplicate triggering
        self._processed_command_ids: deque = deque(maxlen=200)

        # Notify initial state and log startup audit event
        self._notify_state_change()
        self._log_audit(
            AuditEventType.LOCK_STARTUP_INIT,
            AuditResult.INFO,
            "Simulated smart lock initialized in LOCKED state.",
            source=AuditSource.SYSTEM,
        )
        logger.info("[LOCK CONTROLLER] Initialized smart lock in LOCKED state.")

    def _log_audit(
        self,
        event_type: AuditEventType | str,
        result: AuditResult | str,
        reason: str,
        command_type: Optional[str] = None,
        source: AuditSource | str = AuditSource.VOICE,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ):
        """Dispatches an audit log record asynchronously without blocking lock operations."""
        if self.audit_logger is not None:
            uid = user_id or (self.authenticated_user.user_id if self.authenticated_user else "voice_user")
            uname = username or (self.authenticated_user.username if self.authenticated_user else "Voice User")
            try:
                self.audit_logger.log(
                    event_type=event_type,
                    result=result,
                    reason=reason,
                    lock_state=self.state.value,
                    user_id=uid,
                    username=uname,
                    command_type=command_type,
                    source=source,
                    metadata=metadata or {},
                )
            except Exception as e:
                logger.error(f"[LOCK CONTROLLER] Failed to enqueue audit log: {e}")

    def stop(self):
        """Flushes and cleanly stops the audit logger."""
        if self.audit_logger is not None:
            try:
                self.audit_logger.stop()
            except Exception as e:
                logger.warning(f"[LOCK CONTROLLER] Error stopping audit logger: {e}")

    def set_voice_feedback(self, voice_feedback: Any) -> None:
        """Configures or updates the voice feedback subsystem."""
        with self._lock:
            self.voice_feedback = voice_feedback

    def set_pin_timeout(self, timeout_s: float) -> None:
        """
        Updates PIN entry timeout within safe bounded limits [5.0s, 60.0s].
        Enforces security bounds to prevent weakening authentication security.
        """
        with self._lock:
            clamped = max(5.0, min(60.0, float(timeout_s)))
            self.config.pin_timeout_s = clamped
            if hasattr(self.pin_manager, "config"):
                self.pin_manager.config.pin_timeout_s = clamped
            logger.info(f"[LOCK CONTROLLER] Updated PIN entry timeout to {clamped:.1f}s.")

    def reset(self):
        """
        Safely resets lock controller to initial LOCKED state.
        Guarantees system is never left in UNLOCKED state.
        """
        with self._lock:
            self.state = LockState.LOCKED
            self.failed_attempts = 0
            self.authenticated_user = None
            self._unlock_timestamp = 0.0
            self._lockout_timestamp = 0.0
            self._last_state_change_time = self._time_func()
            self.last_access_result = AccessResult.NONE
            self.last_access_message = "System reset to LOCKED state."
            self.last_command = None
            self._operation_id += 1
            self._emergency_active = False
            self._pending_cancelled = False
            self._pending_unlock_request = None
            self.pin_manager.clear()
            self._processed_command_ids.clear()
            self._notify_state_change()
            self._log_audit(
                AuditEventType.LOCK_ACTIVATED,
                AuditResult.INFO,
                "System reset executed; confirmed in LOCKED state.",
                source=AuditSource.SYSTEM,
            )
            logger.info("[LOCK CONTROLLER] Reset executed; state confirmed as LOCKED.")

    # ==========================================================================
    # State Inspection & Snapshot
    # ==========================================================================

    def get_snapshot(self) -> LockStateSnapshot:
        """Returns an immutable thread-safe snapshot of the current lock controller state."""
        with self._lock:
            now = self._time_func()

            # Calculate remaining timer durations
            lockout_rem = 0.0
            if self.state == LockState.LOCKOUT and self._lockout_timestamp > 0.0:
                elapsed = now - self._lockout_timestamp
                lockout_rem = max(0.0, self.config.lockout_duration_s - elapsed)

            unlock_rem = 0.0
            if self.state == LockState.UNLOCKED and self._unlock_timestamp > 0.0:
                elapsed = now - self._unlock_timestamp
                unlock_rem = max(0.0, self.config.unlock_duration_s - elapsed)

            auth_uid = self.authenticated_user.user_id if self.authenticated_user else None
            auth_name = self.authenticated_user.username if self.authenticated_user else None
            auth_role = self.authenticated_user.role.value if self.authenticated_user else None

            # Determine operational status string
            if self.state == LockState.LOCKOUT:
                op_status = f"SECURITY LOCKOUT ({lockout_rem:.1f}s remaining)"
            elif self.state == LockState.UNLOCKED:
                user_str = f" by {auth_name} ({auth_role})" if auth_name else ""
                op_status = f"ACCESS GRANTED{user_str} (Relock in {unlock_rem:.1f}s)"
            elif self.state == LockState.PIN_ENTRY:
                op_status = f"PIN ENTRY ACTIVE ({self.pin_manager.progress_string})"
            else:
                op_status = "LOCKED (Ready for PIN)"

            return LockStateSnapshot(
                state=self.state,
                is_locked=(self.state in (LockState.LOCKED, LockState.PIN_ENTRY, LockState.LOCKOUT)),
                is_unlocked=(self.state == LockState.UNLOCKED),
                is_lockout=(self.state == LockState.LOCKOUT),
                digits_entered=self.pin_manager.current_length,
                pin_length=self.config.pin_length,
                masked_entry=self.pin_manager.masked_display,
                pin_progress=self.pin_manager.progress_string,
                failed_attempts=self.failed_attempts,
                max_failed_attempts=self.config.max_failed_attempts,
                lockout_remaining_s=lockout_rem,
                unlock_remaining_s=unlock_rem,
                last_access_result=self.last_access_result,
                last_access_message=self.last_access_message,
                last_command=self.last_command,
                last_state_change_time=self._last_state_change_time,
                operational_status=op_status,
                authenticated_user_id=auth_uid,
                authenticated_username=auth_name,
                authenticated_role=auth_role,
                emergency_active=self._emergency_active,
                pending_cancelled=self._pending_cancelled,
            )

    # ==========================================================================
    # Voice Command Handlers
    # ==========================================================================

    def handle_command(self, command: str, command_id: Optional[Any] = None) -> LockStateSnapshot:
        """
        Main entry point for confirmed voice commands from the speech recognition pipeline.
        Routes digits, 'go', and 'stop' appropriately while rejecting duplicates.
        """
        with self._lock:
            # Check duplicate command ID
            if command_id is not None:
                if command_id in self._processed_command_ids:
                    logger.debug(f"[LOCK CONTROLLER] Suppressed duplicate command ID: {command_id}")
                    return self.get_snapshot()
                self._processed_command_ids.append(command_id)

            # Refresh internal timers before command evaluation
            self.tick()

            # Reject digit '0' and 'zero' per PIN policy
            if command in ("zero", "0"):
                logger.warning("[LOCK CONTROLLER] Spoken command 'zero' rejected by PIN policy (digits 1-9 only).")
                self.last_command = "[REJECTED_ZERO]"
                self._log_audit(
                    AuditEventType.UNAUTHORIZED_OP,
                    AuditResult.WARNING,
                    "Spoken command 'zero' rejected by PIN policy (digits 1-9 only).",
                    command_type="zero",
                    source=AuditSource.VOICE,
                )
                return self.get_snapshot()

            # Word to digit mapping (digits 1-9 only)
            number_word_map = {
                "one": "1", "1": "1",
                "two": "2", "2": "2",
                "three": "3", "3": "3",
                "four": "4", "4": "4",
                "five": "5", "5": "5",
                "six": "6", "6": "6",
                "seven": "7", "7": "7",
                "eight": "8", "8": "8",
                "nine": "9", "9": "9",
            }

            if command in number_word_map:
                digit = number_word_map[command]
                # Redact digit from snapshot to prevent exposing credentials via dashboard/logs
                self.last_command = "[DIGIT_ENTERED]"
                self.handle_digit(digit)
            elif command == "go":
                self.last_command = "go"
                self.handle_submit()
            elif command == "stop":
                self.last_command = "stop"
                self.handle_stop()
            else:
                self.last_command = command
                logger.debug(f"[LOCK CONTROLLER] Command '{command}' not applicable to smart lock.")

            return self.get_snapshot()

    def handle_digit(self, digit: str) -> bool:
        """
        Appends a spoken digit ('1'-'9') to the PIN buffer.
        Transitions LOCKED -> PIN_ENTRY upon first digit.
        Rejects input if system is UNLOCKED, in LOCKOUT, or digit is '0'.
        """
        with self._lock:
            self.tick()

            # Guard 0: Reject digit '0' per PIN policy
            if digit == "0":
                logger.warning("[LOCK CONTROLLER] Digit '0' rejected by PIN policy (digits 1-9 only).")
                self._log_audit(
                    AuditEventType.UNAUTHORIZED_OP,
                    AuditResult.WARNING,
                    "Digit '0' rejected by PIN policy (digits 1-9 only).",
                    command_type="digit",
                    source=AuditSource.VOICE,
                )
                return False

            # Guard 1: Reject during LOCKOUT
            if self.state == LockState.LOCKOUT:
                logger.warning("[LOCK CONTROLLER] Rejected digit during active LOCKOUT.")
                self._log_audit(
                    AuditEventType.UNAUTHORIZED_OP,
                    AuditResult.WARNING,
                    f"Digit entry rejected during active security LOCKOUT ({self._get_lockout_remaining():.1f}s remaining).",
                    command_type="digit",
                    source=AuditSource.VOICE,
                )
                return False

            # Guard 2: Reject if already UNLOCKED
            if self.state == LockState.UNLOCKED:
                logger.warning("[LOCK CONTROLLER] Rejected digit; lock is already UNLOCKED.")
                return False

            # Transition LOCKED -> PIN_ENTRY on first digit
            if self.state == LockState.LOCKED:
                self._emergency_active = False
                self._transition_to(LockState.PIN_ENTRY, "First digit entered; starting PIN entry")
                self._log_audit(
                    AuditEventType.AUTH_SUBMISSION,
                    AuditResult.INFO,
                    "PIN entry initiated by user.",
                    command_type="digit",
                    source=AuditSource.VOICE,
                    metadata={"digits_entered": 1},
                )

            # Append digit to PIN manager
            accepted = self.pin_manager.append_digit(digit, current_time=self._time_func())
            if accepted:
                if self.voice_feedback is not None:
                    try:
                        self.voice_feedback.on_digit_accepted()
                        if self.pin_manager.current_length == self.config.pin_length:
                            self.voice_feedback.on_pin_complete()
                    except Exception as e:
                        logger.debug(f"[LOCK CONTROLLER] Voice feedback digit error: {e}")
                self._notify_state_change()
            return accepted

    def handle_submit(self) -> Tuple[bool, AccessResult, str]:
        """
        Evaluates entered PIN buffer when user speaks 'go'.
        Grants access on match, increments failed attempts on mismatch,
        and triggers LOCKOUT after 3 consecutive failures.
        """
        with self._lock:
            self.tick()
            op_id = self.begin_access_operation()

            # Guard 1: Reject during LOCKOUT
            if self.state == LockState.LOCKOUT:
                msg = f"Access denied: System is in temporary LOCKOUT ({self._get_lockout_remaining():.1f}s remaining)."
                self.last_access_result = AccessResult.DENIED_LOCKOUT
                self.last_access_message = msg
                logger.warning(f"[LOCK CONTROLLER] {msg}")
                self._notify_state_change()
                self._log_audit(
                    AuditEventType.UNAUTHORIZED_OP,
                    AuditResult.WARNING,
                    "PIN submission rejected during active LOCKOUT.",
                    command_type="go",
                    source=AuditSource.VOICE,
                )
                return False, AccessResult.DENIED_LOCKOUT, msg

            # Guard 2: If already UNLOCKED
            if self.state == LockState.UNLOCKED:
                msg = "System is already UNLOCKED."
                self.last_access_message = msg
                return True, AccessResult.GRANTED, msg

            # Guard 3: Incomplete PIN submission
            if self.pin_manager.current_length < self.config.pin_length:
                entered = self.pin_manager.current_length
                self.pin_manager.clear()
                self.authenticated_user = None
                self.failed_attempts += 1
                msg = f"Access denied: Incomplete PIN ({entered}/{self.config.pin_length} digits)."
                self.last_access_result = AccessResult.DENIED_INCOMPLETE
                self.last_access_message = msg
                logger.warning(f"[LOCK CONTROLLER] {msg} Attempt {self.failed_attempts}/{self.config.max_failed_attempts}.")
                if self.voice_feedback is not None:
                    try:
                        self.voice_feedback.on_access_denied()
                    except Exception as e:
                        logger.debug(f"[LOCK CONTROLLER] Voice feedback access denied error: {e}")

                self._log_audit(
                    AuditEventType.AUTH_INCOMPLETE,
                    AuditResult.FAILURE,
                    msg,
                    command_type="go",
                    source=AuditSource.VOICE,
                    metadata={"digits_entered": entered, "failed_attempts": self.failed_attempts},
                )

                if self.failed_attempts >= self.config.max_failed_attempts:
                    self._activate_lockout()
                    return False, AccessResult.DENIED_LOCKOUT, f"Lockout activated: {self.config.max_failed_attempts} failed attempts."
                else:
                    self._transition_to(LockState.LOCKED, "Incomplete PIN submitted; returning to LOCKED")
                    return False, AccessResult.DENIED_INCOMPLETE, msg

            # Guard 4: Full Multi-User Authentication via salted PBKDF2 hash
            candidate_pin = self.pin_manager.consume_buffer()
            if candidate_pin is None:
                return False, AccessResult.DENIED_INCOMPLETE, "Access denied: Incomplete PIN."

            authenticated = False
            matched_user: Optional[UserRecord] = None
            denial_reason = "Incorrect PIN."
            is_expired = False
            is_disabled = False

            # Multi-User RBAC evaluation
            if self.user_manager is not None:
                auth_ok, user_rec, reason_code = self.user_manager.authenticate(candidate_pin)
                if auth_ok:
                    authenticated = True
                    matched_user = user_rec
                elif reason_code == "GUEST_EXPIRED":
                    is_expired = True
                    matched_user = user_rec
                    denial_reason = "Guest account has expired."
                elif reason_code == "ACCOUNT_DISABLED":
                    is_disabled = True
                    matched_user = user_rec
                    denial_reason = "Account is disabled."
                else:
                    denial_reason = "Incorrect PIN."

            # Backward-compatibility fallback to PINManager if not matched in UserManager
            if not authenticated and not is_expired and not is_disabled:
                if (
                    hasattr(self.pin_manager, "_hash_hex")
                    and self.pin_manager._hash_hex
                    and hasattr(self.pin_manager, "_salt_hex")
                    and self.pin_manager._salt_hex
                ):
                    salt = bytes.fromhex(self.pin_manager._salt_hex)
                    expected_hash = bytes.fromhex(self.pin_manager._hash_hex)
                    cand_hash = hashlib.pbkdf2_hmac(
                        hash_name=self.config.hash_algorithm,
                        password=candidate_pin.encode("utf-8"),
                        salt=salt,
                        iterations=self.config.hash_iterations,
                    )
                    if hmac.compare_digest(cand_hash, expected_hash):
                        authenticated = True
                        if self.user_manager is not None:
                            matched_user = self.user_manager.get_user(self.config.default_admin_id)
                        if matched_user is None:
                            matched_user = UserRecord(
                                user_id=self.config.default_admin_id,
                                username=self.config.default_admin_name,
                                pin_hash=self.pin_manager._hash_hex,
                                salt_hex=self.pin_manager._salt_hex,
                                role=UserRole.ADMIN,
                                status=UserStatus.ACTIVE,
                                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            )

            if authenticated and matched_user is not None:
                return self.commit_unlock(op_id, matched_user, command_type="go", source=AuditSource.VOICE)
            else:
                # ACCESS DENIED
                self.authenticated_user = None
                self.failed_attempts += 1
                msg = f"Access denied: Incorrect PIN (Attempt {self.failed_attempts}/{self.config.max_failed_attempts})."
                self.last_access_result = AccessResult.DENIED_INCORRECT
                self.last_access_message = msg
                logger.warning(f"[LOCK CONTROLLER] {msg} Reason: {denial_reason}")
                if self.voice_feedback is not None:
                    try:
                        self.voice_feedback.on_access_denied()
                    except Exception as e:
                        logger.debug(f"[LOCK CONTROLLER] Voice feedback access denied error: {e}")

                self._log_audit(
                    AuditEventType.AUTH_FAILURE,
                    AuditResult.FAILURE,
                    f"Authentication failed: {denial_reason}",
                    command_type="go",
                    source=AuditSource.VOICE,
                    metadata={
                        "failed_attempts": self.failed_attempts,
                        "max_failed_attempts": self.config.max_failed_attempts,
                        "reason_detail": denial_reason,
                    },
                )

                if self.failed_attempts >= self.config.max_failed_attempts:
                    self._activate_lockout()
                    return False, AccessResult.DENIED_LOCKOUT, f"Lockout activated: {self.config.max_failed_attempts} failed attempts."
                else:
                    self._transition_to(LockState.LOCKED, "Incorrect PIN; returning to LOCKED")
                    return False, AccessResult.DENIED_INCORRECT, msg

    # ==========================================================================
    # Operation Sequencing & Emergency Precedence
    # ==========================================================================

    def begin_access_operation(self) -> int:
        """
        Begins a new tracked authentication or unlock operation.
        Returns monotonically increasing operation ID.
        """
        with self._lock:
            self._operation_id += 1
            self._emergency_active = False
            return self._operation_id

    def is_operation_valid(self, op_id: int) -> bool:
        """Checks if operation ID matches current generation and no emergency stop cancelled it."""
        with self._lock:
            return (
                op_id == self._operation_id
                and self.state != LockState.LOCKOUT
            )

    def commit_unlock(
        self,
        op_id: int,
        user_rec: UserRecord,
        unlock_duration_s: Optional[float] = None,
        command_type: Optional[str] = "go",
        source: AuditSource = AuditSource.VOICE,
    ) -> Tuple[bool, AccessResult, str]:
        """
        Commits an authenticated unlock operation if its op_id is still valid.
        Rejects stale unlocks cancelled by emergency STOP.
        """
        with self._lock:
            self.tick()

            # Guard 1: Lockout
            if self.state == LockState.LOCKOUT:
                msg = f"Access denied: System is in temporary LOCKOUT ({self._get_lockout_remaining():.1f}s remaining)."
                self.last_access_result = AccessResult.DENIED_LOCKOUT
                self.last_access_message = msg
                return False, AccessResult.DENIED_LOCKOUT, msg

            # Guard 2: Stale operation / Cancelled by emergency STOP
            if op_id != self._operation_id:
                msg = f"Access denied: Operation {op_id} was cancelled or superseded by emergency STOP."
                self.last_access_result = AccessResult.CLEARED
                self.last_access_message = msg
                logger.warning(f"[LOCK CONTROLLER] {msg}")
                self._log_audit(
                    AuditEventType.EMERGENCY_CANCEL_PENDING,
                    AuditResult.WARNING,
                    f"Stale unlock attempt (op_id={op_id}, current={self._operation_id}) blocked by emergency state.",
                    command_type=command_type,
                    source=source,
                    user_id=user_rec.user_id,
                    username=user_rec.username,
                    metadata={"op_id": op_id, "current_op_id": self._operation_id},
                )
                self._notify_state_change()
                return False, AccessResult.CLEARED, msg

            # Valid commit
            duration = unlock_duration_s if unlock_duration_s is not None else self.config.unlock_duration_s
            self.authenticated_user = user_rec
            self.failed_attempts = 0
            now = self._time_func()
            self._unlock_timestamp = now
            msg = (
                f"ACCESS GRANTED: User '{user_rec.username}' ({user_rec.role.value}) authenticated. "
                f"Door unlocked for {duration}s."
            )
            self.last_access_result = AccessResult.GRANTED
            self.last_access_message = msg
            logger.info(f"[LOCK CONTROLLER] *** {msg} ***")
            if self.voice_feedback is not None:
                try:
                    self.voice_feedback.on_access_granted()
                except Exception as e:
                    logger.debug(f"[LOCK CONTROLLER] Voice feedback access granted error: {e}")
            self._transition_to(LockState.UNLOCKED, f"Correct PIN for {user_rec.username}")

            self._log_audit(
                AuditEventType.AUTH_SUCCESS,
                AuditResult.SUCCESS,
                f"Voice PIN authentication succeeded for '{user_rec.username}'.",
                command_type=command_type,
                source=source,
                user_id=user_rec.user_id,
                username=user_rec.username,
                metadata={"user_id": user_rec.user_id, "role": user_rec.role.value, "op_id": op_id},
            )
            self._log_audit(
                AuditEventType.LOCK_RELEASED,
                AuditResult.SUCCESS,
                f"Door lock released for {duration}s.",
                command_type=command_type,
                source=AuditSource.SYSTEM,
                user_id=user_rec.user_id,
                username=user_rec.username,
                metadata={"unlock_duration_s": duration, "user_id": user_rec.user_id, "op_id": op_id},
            )
            return True, AccessResult.GRANTED, msg

    def request_pending_unlock(self, user_rec: UserRecord, delay_s: float = 0.0) -> int:
        """
        Stages a pending unlock request.
        Returns assigned operation ID.
        """
        with self._lock:
            self._operation_id += 1
            op_id = self._operation_id
            self._pending_unlock_request = {
                "op_id": op_id,
                "user": user_rec,
                "request_time": self._time_func(),
                "delay_s": delay_s,
            }
            logger.info(f"[LOCK CONTROLLER] Staged pending unlock request (op_id={op_id}) for '{user_rec.username}'.")
            return op_id

    def execute_pending_unlock(self, op_id: Optional[int] = None) -> Tuple[bool, AccessResult, str]:
        """Executes a previously staged pending unlock request."""
        with self._lock:
            if self._pending_unlock_request is None:
                return False, AccessResult.NONE, "No pending unlock request found."
            req = self._pending_unlock_request
            self._pending_unlock_request = None
            target_id = op_id if op_id is not None else req["op_id"]
            return self.commit_unlock(target_id, req["user"])

    def cancel_pending_operations(self) -> int:
        """
        Explicitly invalidates all pending operations and clears active PIN entry.
        Returns count of cancelled pending items.
        """
        with self._lock:
            cancelled_count = 0
            if self._pending_unlock_request is not None:
                cancelled_count += 1
                self._pending_unlock_request = None
            if self.pin_manager.current_length > 0 or self.state == LockState.PIN_ENTRY:
                cancelled_count += 1
                self.pin_manager.clear()
                if self.state == LockState.PIN_ENTRY:
                    self._transition_to(LockState.LOCKED, "Pending operations cancelled; returning to LOCKED")
            self._operation_id += 1
            return cancelled_count

    def emergency_stop(
        self,
        source: AuditSource = AuditSource.VOICE,
        reason: Optional[str] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Tuple[LockStateSnapshot, bool]:
        """
        High-priority emergency STOP:
        1. Increments operation ID, immediately invalidating any in-flight or queued unlock requests.
        2. Clears active PIN entry buffer.
        3. Cancels any staged pending unlock request.
        4. If UNLOCKED, immediately cancels unlock duration and relocks.
        5. If in LOCKOUT, strictly PRESERVES the lockout state and timer (does not bypass).
        6. Guarantees system ends in LOCKED (or LOCKOUT) state.
        7. Logs structured audit events.
        """
        with self._lock:
            self.tick()

            # Invalidate any in-flight or queued operations
            self._operation_id += 1

            # Detect if a pending operation was cancelled
            had_pending = False
            if self._pending_unlock_request is not None:
                had_pending = True
                self._pending_unlock_request = None
            if self.pin_manager.current_length > 0 or self.state == LockState.PIN_ENTRY:
                had_pending = True
                self.pin_manager.clear()
            if self.state == LockState.UNLOCKED:
                had_pending = True

            self._emergency_active = True
            self._pending_cancelled = had_pending
            self.authenticated_user = None

            # Handle LOCKOUT preservation
            if self.state == LockState.LOCKOUT:
                logger.warning("[LOCK CONTROLLER] Emergency STOP received during active LOCKOUT; lockout preserved.")
                self.last_access_result = AccessResult.DENIED_LOCKOUT
                msg = reason or "Emergency STOP received during active LOCKOUT; security lockout preserved."
                self.last_access_message = msg
                if self.voice_feedback is not None:
                    try:
                        self.voice_feedback.on_emergency_stop_accepted()
                    except Exception as e:
                        logger.debug(f"[LOCK CONTROLLER] Voice feedback emergency stop error: {e}")
                self._log_audit(
                    AuditEventType.EMERGENCY_STOP,
                    AuditResult.WARNING,
                    msg,
                    command_type="stop",
                    source=source,
                    user_id=user_id,
                    username=username,
                    metadata={
                        "lockout_remaining_s": self._get_lockout_remaining(),
                        "failed_attempts": self.failed_attempts,
                        "pending_cancelled": had_pending,
                    },
                )
                self._notify_state_change()
                return self.get_snapshot(), had_pending

            # Normal states (LOCKED, PIN_ENTRY, UNLOCKED)
            was_unlocked = (self.state == LockState.UNLOCKED)
            self._unlock_timestamp = 0.0

            msg = reason or ("Lock manually re-engaged by emergency STOP." if was_unlocked else "PIN buffer cleared and lock secured by emergency STOP.")
            self.last_access_result = AccessResult.CLEARED
            self.last_access_message = msg

            if was_unlocked:
                self._transition_to(LockState.LOCKED, "Emergency STOP: forced relock from UNLOCKED")
                self._log_audit(
                    AuditEventType.LOCK_MANUAL_RELOCK,
                    AuditResult.INFO,
                    "Lock manually re-engaged by emergency STOP.",
                    command_type="stop",
                    source=source,
                    user_id=user_id,
                    username=username,
                )
            elif self.state == LockState.PIN_ENTRY:
                self._transition_to(LockState.LOCKED, "Emergency STOP: PIN entry cancelled")
            else:
                self._transition_to(LockState.LOCKED, "Emergency STOP: secure state confirmed")

            self._log_audit(
                AuditEventType.EMERGENCY_STOP,
                AuditResult.SUCCESS,
                msg,
                command_type="stop",
                source=source,
                user_id=user_id,
                username=username,
                metadata={"pending_cancelled": had_pending, "was_unlocked": was_unlocked},
            )

            if had_pending:
                self._log_audit(
                    AuditEventType.EMERGENCY_CANCEL_PENDING,
                    AuditResult.INFO,
                    "Emergency STOP cancelled active access operation or PIN entry.",
                    command_type="stop",
                    source=source,
                    user_id=user_id,
                    username=username,
                )

            if self.voice_feedback is not None:
                try:
                    self.voice_feedback.on_emergency_stop_accepted()
                    if was_unlocked:
                        self.voice_feedback.on_lock_activated()
                except Exception as e:
                    logger.debug(f"[LOCK CONTROLLER] Voice feedback emergency stop error: {e}")

            self._notify_state_change()
            logger.info(f"[LOCK CONTROLLER] Emergency STOP executed. Pending cancelled: {had_pending}. State: {self.state.value}")
            return self.get_snapshot(), had_pending

    def handle_stop(self) -> None:
        """Executes 'stop' command via unified emergency_stop procedure."""
        self.emergency_stop(source=AuditSource.VOICE, reason="Voice 'stop' command executed.")

    def clear_emergency(self):
        """Clears the active emergency status flag."""
        with self._lock:
            self._emergency_active = False
            self._pending_cancelled = False
            self._notify_state_change()
            logger.info("[LOCK CONTROLLER] Emergency state cleared.")

    # ==========================================================================
    # Periodic Timer & State Maintenance
    # ==========================================================================

    def tick(self, current_time: Optional[float] = None) -> None:
        """
        Performs periodic timer maintenance:
        1. Checks whether LOCKOUT duration (default: 30s) has expired.
        2. Checks whether UNLOCKED duration (default: 5s) has expired (auto-relock).
        3. Checks whether PIN_ENTRY timeout (default: 10s) has expired.
        """
        with self._lock:
            now = self._time_func() if current_time is None else current_time

            # 1. Check LOCKOUT expiration
            if self.state == LockState.LOCKOUT:
                elapsed = now - self._lockout_timestamp
                if elapsed >= self.config.lockout_duration_s:
                    logger.info(
                        f"[LOCK CONTROLLER] Security lockout duration elapsed ({elapsed:.1f}s >= "
                        f"{self.config.lockout_duration_s}s). Returning to LOCKED."
                    )
                    self.failed_attempts = 0
                    self._lockout_timestamp = 0.0
                    self.authenticated_user = None
                    self.last_access_message = "Lockout expired. Ready for authentication."
                    if self.voice_feedback is not None:
                        try:
                            self.voice_feedback.on_lockout_expired()
                        except Exception as e:
                            logger.debug(f"[LOCK CONTROLLER] Voice feedback lockout expired error: {e}")
                    self._transition_to(LockState.LOCKED, "Lockout duration expired")
                    self._log_audit(
                        AuditEventType.LOCKOUT_EXPIRED,
                        AuditResult.INFO,
                        "Security lockout duration elapsed; system ready for authentication.",
                        source=AuditSource.SYSTEM,
                    )

            # 2. Check Auto-Relock expiration
            elif self.state == LockState.UNLOCKED:
                elapsed = now - self._unlock_timestamp
                if elapsed >= self.config.unlock_duration_s:
                    logger.info(
                        f"[LOCK CONTROLLER] Unlock duration elapsed ({elapsed:.1f}s >= "
                        f"{self.config.unlock_duration_s}s). Automatically relocking system."
                    )
                    self._unlock_timestamp = 0.0
                    self.authenticated_user = None
                    self.last_access_result = AccessResult.RELOCKED_AUTO
                    self.last_access_message = "System automatically relocked after timeout."
                    if self.voice_feedback is not None:
                        try:
                            self.voice_feedback.on_lock_activated()
                        except Exception as e:
                            logger.debug(f"[LOCK CONTROLLER] Voice feedback lock activated error: {e}")
                    self._transition_to(LockState.LOCKED, "Automatic relock timer expired")
                    self._log_audit(
                        AuditEventType.LOCK_AUTO_RELOCK,
                        AuditResult.INFO,
                        f"Unlock duration expired ({self.config.unlock_duration_s}s); automatically relocked.",
                        source=AuditSource.SYSTEM,
                    )

            # 3. Check PIN entry inactivity timeout
            elif self.state == LockState.PIN_ENTRY:
                if self.pin_manager.check_timeout(current_time=now):
                    self.last_access_result = AccessResult.TIMEOUT
                    self.last_access_message = "PIN entry timed out due to user inactivity."
                    self._transition_to(LockState.LOCKED, "PIN entry inactivity timeout")
                    self._log_audit(
                        AuditEventType.AUTH_TIMEOUT,
                        AuditResult.WARNING,
                        f"PIN entry timed out due to user inactivity ({self.config.pin_timeout_s}s).",
                        source=AuditSource.SYSTEM,
                    )

    # ==========================================================================
    # Internal Helpers
    # ==========================================================================

    def _activate_lockout(self):
        """Activates temporary security lockout following maximum failed attempts."""
        now = self._time_func()
        self._lockout_timestamp = now
        self.pin_manager.clear()
        self.authenticated_user = None
        self.last_access_result = AccessResult.DENIED_LOCKOUT
        self.last_access_message = (
            f"MAXIMUM FAILED ATTEMPTS ({self.config.max_failed_attempts}) REACHED. "
            f"System locked out for {self.config.lockout_duration_s}s."
        )
        logger.warning(f"[LOCK CONTROLLER] *** {self.last_access_message} ***")
        if self.voice_feedback is not None:
            try:
                self.voice_feedback.on_lockout_activated()
            except Exception as e:
                logger.debug(f"[LOCK CONTROLLER] Voice feedback lockout activated error: {e}")
        self._transition_to(LockState.LOCKOUT, f"Failed attempts reached limit ({self.config.max_failed_attempts})")

        self._log_audit(
            AuditEventType.LOCKOUT_ACTIVATED,
            AuditResult.WARNING,
            f"Security lockout activated after {self.config.max_failed_attempts} consecutive failed attempts.",
            source=AuditSource.SYSTEM,
            metadata={"lockout_duration_s": self.config.lockout_duration_s, "failed_attempts": self.failed_attempts},
        )

    def change_pin(self, new_pin: str, user_id: str = "admin") -> Tuple[bool, str]:
        """Admin function to update the authentication PIN."""
        import pathlib
        with self._lock:
            try:
                self.pin_manager.provision_pin(new_pin, persist_path=pathlib.Path(self.config.credentials_path))
                if self.user_manager is not None:
                    admin_rec = self.user_manager.get_user(user_id) or self.user_manager.get_user(self.config.default_admin_id)
                    if admin_rec is not None:
                        self.user_manager.reset_user_pin(admin_rec.user_id, admin_rec.user_id, new_pin)
                self._log_audit(
                    AuditEventType.PIN_CHANGED,
                    AuditResult.SUCCESS,
                    "Authentication PIN successfully updated.",
                    user_id=user_id,
                    username=user_id.capitalize(),
                    source=AuditSource.ADMINISTRATOR,
                )
                return True, "PIN updated successfully."
            except Exception as e:
                self._log_audit(
                    AuditEventType.PIN_CHANGED,
                    AuditResult.FAILURE,
                    f"Failed to update PIN: {e}",
                    user_id=user_id,
                    username=user_id.capitalize(),
                    source=AuditSource.ADMINISTRATOR,
                )
                return False, str(e)

    def log_admin_action(self, action: str, details: str, user_id: str = "admin") -> None:
        """Logs an administrative action (e.g. user management, configuration changes)."""
        self._log_audit(
            AuditEventType.ADMIN_ACTION,
            AuditResult.INFO,
            f"Admin action: {action} - {details}",
            user_id=user_id,
            username=user_id.capitalize(),
            source=AuditSource.ADMINISTRATOR,
        )

    def log_security_event(
        self,
        event_type: AuditEventType | str,
        result: AuditResult | str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Dispatches an arbitrary security or system event."""
        self._log_audit(event_type, result, reason, source=AuditSource.SYSTEM, metadata=metadata)

    def _get_lockout_remaining(self) -> float:
        """Returns remaining seconds in current lockout, or 0.0."""
        if self.state != LockState.LOCKOUT or self._lockout_timestamp <= 0.0:
            return 0.0
        elapsed = self._time_func() - self._lockout_timestamp
        return max(0.0, self.config.lockout_duration_s - elapsed)

    def _transition_to(self, new_state: LockState, reason: str):
        """Executes explicit state transition and notifies listeners."""
        if self.state != new_state:
            prev = self.state
            self.state = new_state
            self._last_state_change_time = self._time_func()
            logger.info(f"[LOCK CONTROLLER] Transition: {prev.value} -> {new_state.value} ({reason})")
            self._notify_state_change()

    def _notify_state_change(self):
        """Notifies optional subscriber/dashboard of state update."""
        if self._on_state_change is not None:
            try:
                snapshot = self.get_snapshot()
                self._on_state_change(snapshot)
            except Exception as e:
                logger.error(f"[LOCK CONTROLLER] Error in state change callback: {e}")

