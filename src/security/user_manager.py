"""
src/security/user_manager.py
----------------------------
Persistent User Management and Role-Based Access Control (RBAC) Subsystem.

Responsibilities:
1. Implements Role-Based Access Control with three primary roles:
   - ADMIN: Full administrative privileges (user management, audit logs, PIN resets, policy configuration).
   - REGULAR_USER: Standard voice authentication, lock unlocking, manual locking.
   - GUEST: Temporary access with mandatory account expiration dates.
2. Cryptographic Credential Protection:
   - Plaintext PINs are NEVER stored in database columns or memory structures.
   - Each user record maintains a unique cryptographically random salt and PBKDF2-HMAC-SHA256 hash.
   - Strictly enforces digits 1-9 only (digit '0' rejected) and configured PIN length.
3. Duplicate PIN Rejection:
   - Evaluates candidate PINs against all registered accounts during user creation and PIN reset.
   - Rejects duplicate PINs to eliminate user ambiguity during voice authentication.
4. Administrative Protection:
   - All user management operations require authenticated administrator privileges.
   - Prevents deletion or disabling of the last remaining administrator.
   - Dispatches structured audit events for every administrative action.
5. SQLite Persistence:
   - Uses SQLite with WAL mode (`data/users.db`) ensuring persistence across application restarts.
"""

from dataclasses import dataclass
import datetime
from enum import Enum
import hashlib
import hmac
import logging
import os
import pathlib
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.security.config import SecurityConfig
from src.security.audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditResult,
    AuditSource,
)

logger = logging.getLogger("rtos.security.user")


class UserRole(str, Enum):
    """Supported roles in the RBAC system."""
    ADMIN = "ADMIN"
    REGULAR_USER = "REGULAR_USER"
    GUEST = "GUEST"


class UserStatus(str, Enum):
    """Account operational states."""
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    EXPIRED = "EXPIRED"


@dataclass
class UserRecord:
    """Represents an individual user account with credentials and metadata."""
    user_id: str
    username: str
    pin_hash: str
    salt_hex: str
    role: UserRole
    status: UserStatus
    created_at: str
    expires_at: Optional[str] = None
    last_access_at: Optional[str] = None
    hash_algorithm: str = "sha256"
    hash_iterations: int = 100_000

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Serializes user record to dictionary. Excludes sensitive hashes and salts by default."""
        data = {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role.value if isinstance(self.role, Enum) else str(self.role),
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_access_at": self.last_access_at,
        }
        if include_sensitive:
            data["pin_hash"] = self.pin_hash
            data["salt_hex"] = self.salt_hex
            data["hash_algorithm"] = self.hash_algorithm
            data["hash_iterations"] = self.hash_iterations
        return data


class UserManager:
    """
    Thread-safe, persistent User Management and Role-Based Access Control manager.
    Coordinates user provisioning, duplicate PIN detection, credential hashing, and administrative operations.
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        db_path: Optional[str | pathlib.Path] = None,
        time_func: Optional[Callable[[], float]] = None,
        allow_demo_bootstrap: Optional[bool] = None,
    ):
        self.config = config or SecurityConfig()
        self.audit_logger = audit_logger
        self.db_path = pathlib.Path(db_path or self.config.users_db_path)
        self._time_func = time_func or time.monotonic
        self._lock = threading.RLock()

        if allow_demo_bootstrap is not None:
            self.allow_demo_bootstrap = allow_demo_bootstrap
        elif os.environ.get("RTOS_ALLOW_DEMO_BOOTSTRAP", "").lower() in ("1", "true", "yes"):
            self.allow_demo_bootstrap = True
        else:
            self.allow_demo_bootstrap = getattr(self.config, "allow_demo_bootstrap", False)

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database schema
        self._init_db()

        # Only bootstrap initial administrator if explicitly permitted by demo/dev option
        if self.allow_demo_bootstrap:
            self._bootstrap_demo_admin_if_empty()
        else:
            logger.info(f"[USER MANAGER] Normal startup. User database: '{self.db_path}'. Demo bootstrap disabled.")

    def _get_connection(self) -> sqlite3.Connection:
        """Creates an SQLite connection with WAL mode enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        """Creates the users table and indexes if they do not exist."""
        conn = None
        try:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    pin_hash TEXT NOT NULL,
                    salt_hex TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    last_access_at TEXT,
                    hash_algorithm TEXT NOT NULL DEFAULT 'sha256',
                    hash_iterations INTEGER NOT NULL DEFAULT 100000
                );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users (role);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_users_status ON users (status);")
            logger.info(f"[USER MANAGER] Initialized SQLite schema at '{self.db_path}'.")
        except sqlite3.Error as e:
            logger.error(f"[USER MANAGER] Failed to initialize SQLite database: {e}")
            raise
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _bootstrap_demo_admin_if_empty(self):
        """Bootstraps default development administrator account if empty and demo flag is active."""
        with self._lock:
            users = self.get_all_users()
            if len(users) == 0:
                demo_pin = self.config.initial_demo_pin or "1234"
                admin_id = self.config.default_admin_id
                admin_name = self.config.default_admin_name
                logger.info(
                    f"[USER MANAGER] Demo bootstrap active. Initializing administrator '{admin_id}' ({admin_name})."
                )
                self.bootstrap_initial_admin(admin_id=admin_id, username=admin_name, pin=demo_pin)

    def bootstrap_initial_admin(
        self,
        admin_id: Optional[str] = None,
        username: Optional[str] = None,
        pin: Optional[str] = None,
    ) -> UserRecord:
        """
        Explicit first-time administrator provisioning procedure.
        Requires deliberate operator action and a valid PIN.
        Fails if users already exist in the database.
        """
        with self._lock:
            users = self.get_all_users()
            if len(users) > 0:
                raise ValueError("Cannot bootstrap initial administrator: User database is not empty.")

            admin_id = admin_id or self.config.default_admin_id
            username = username or self.config.default_admin_name
            if not pin:
                raise ValueError("PIN is required for first-time administrator provisioning.")

            self._validate_pin_policy(pin)
            salt = os.urandom(self.config.salt_bytes)
            pin_hash = hashlib.pbkdf2_hmac(
                hash_name=self.config.hash_algorithm,
                password=pin.encode("utf-8"),
                salt=salt,
                iterations=self.config.hash_iterations,
            ).hex()

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            admin_user = UserRecord(
                user_id=admin_id,
                username=username,
                pin_hash=pin_hash,
                salt_hex=salt.hex(),
                role=UserRole.ADMIN,
                status=UserStatus.ACTIVE,
                created_at=now_iso,
                expires_at=None,
                last_access_at=None,
                hash_algorithm=self.config.hash_algorithm,
                hash_iterations=self.config.hash_iterations,
            )
            self._insert_user_record(admin_user)
            self._log_audit(
                AuditEventType.ADMIN_CREATED_USER,
                AuditResult.SUCCESS,
                f"Initial administrator '{admin_id}' provisioned via explicit bootstrap.",
                admin_id=admin_id,
                source=AuditSource.SYSTEM,
                metadata={"user_id": admin_id, "role": "ADMIN"},
            )
            logger.info(f"[USER MANAGER] Explicitly provisioned initial administrator '{admin_id}' ({username}).")
            return admin_user

    def _log_audit(
        self,
        event_type: AuditEventType | str,
        result: AuditResult | str,
        reason: str,
        admin_id: Optional[str] = None,
        source: AuditSource | str = AuditSource.ADMINISTRATOR,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Asynchronously dispatches an audit log event if AuditLogger is configured."""
        if self.audit_logger is not None:
            try:
                self.audit_logger.log(
                    event_type=event_type,
                    result=result,
                    reason=reason,
                    lock_state="LOCKED",
                    user_id=admin_id or "system",
                    username="Administrator" if admin_id else "System",
                    source=source,
                    metadata=metadata or {},
                )
            except Exception as e:
                logger.error(f"[USER MANAGER] Error dispatching audit event: {e}")

    # ==========================================================================
    # User Retrieval & Query
    # ==========================================================================

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        """Retrieves a single user record by unique user_id."""
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
            SELECT user_id, username, pin_hash, salt_hex, role, status, created_at, expires_at, last_access_at, hash_algorithm, hash_iterations
            FROM users WHERE user_id = ?;
            """, (user_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_all_users(self) -> List[UserRecord]:
        """Retrieves all registered user records."""
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
            SELECT user_id, username, pin_hash, salt_hex, role, status, created_at, expires_at, last_access_at, hash_algorithm, hash_iterations
            FROM users ORDER BY created_at ASC;
            """)
            return [self._row_to_record(row) for row in cur.fetchall()]
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _row_to_record(self, row: sqlite3.Row) -> UserRecord:
        """Converts an SQLite row to a UserRecord object."""
        return UserRecord(
            user_id=row["user_id"],
            username=row["username"],
            pin_hash=row["pin_hash"],
            salt_hex=row["salt_hex"],
            role=UserRole(row["role"]),
            status=UserStatus(row["status"]),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            last_access_at=row["last_access_at"],
            hash_algorithm=row["hash_algorithm"],
            hash_iterations=row["hash_iterations"],
        )

    # ==========================================================================
    # PIN Policy & Duplicate PIN Checking
    # ==========================================================================

    def _validate_pin_policy(self, pin: str):
        """Enforces length and digit policy: exact length, digits 1-9 only (digit '0' rejected)."""
        if not pin.isdigit():
            raise ValueError("PIN must contain only numeric digits (1-9).")
        if not all(c in "123456789" for c in pin):
            raise ValueError("PIN policy violation: Only digits 1 through 9 are permitted (digit '0' rejected).")
        if len(pin) != self.config.pin_length:
            raise ValueError(f"PIN length must match configured length {self.config.pin_length}. Got {len(pin)} digits.")

    def _is_duplicate_pin(self, candidate_pin: str, exclude_user_id: Optional[str] = None) -> bool:
        """
        Evaluates whether a proposed PIN is already assigned to any existing account.
        Performs constant-time comparison across all users.
        """
        users = self.get_all_users()
        for u in users:
            if exclude_user_id is not None and u.user_id == exclude_user_id:
                continue
            salt = bytes.fromhex(u.salt_hex)
            expected_hash = bytes.fromhex(u.pin_hash)
            derived_hash = hashlib.pbkdf2_hmac(
                hash_name=u.hash_algorithm,
                password=candidate_pin.encode("utf-8"),
                salt=salt,
                iterations=u.hash_iterations,
            )
            if hmac.compare_digest(derived_hash, expected_hash):
                return True
        return False

    # ==========================================================================
    # Authentication & Access Verification
    # ==========================================================================

    def authenticate(
        self,
        candidate_pin: str,
        current_time_iso: Optional[str] = None,
    ) -> Tuple[bool, Optional[UserRecord], str]:
        """
        Authenticates an entered PIN string across all active users.
        Returns:
            (True, UserRecord, "AUTHENTICATION_SUCCESS")
            (False, None, "INCORRECT_CREDENTIALS")
            (False, UserRecord, "ACCOUNT_DISABLED")
            (False, UserRecord, "GUEST_EXPIRED")
        """
        with self._lock:
            matched_user: Optional[UserRecord] = None
            users = self.get_all_users()

            for u in users:
                salt = bytes.fromhex(u.salt_hex)
                expected_hash = bytes.fromhex(u.pin_hash)
                derived_hash = hashlib.pbkdf2_hmac(
                    hash_name=u.hash_algorithm,
                    password=candidate_pin.encode("utf-8"),
                    salt=salt,
                    iterations=u.hash_iterations,
                )
                if hmac.compare_digest(derived_hash, expected_hash):
                    matched_user = u
                    break

            if matched_user is None:
                logger.warning("[USER MANAGER] Authentication failed: No account matched candidate PIN.")
                self._log_audit(
                    AuditEventType.USER_ACCESS_DENIED,
                    AuditResult.FAILURE,
                    "Access denied: Incorrect authentication PIN.",
                    source=AuditSource.VOICE,
                )
                return False, None, "INCORRECT_CREDENTIALS"

            # Check account disabled status
            if matched_user.status == UserStatus.DISABLED:
                logger.warning(f"[USER MANAGER] Authentication rejected: Account '{matched_user.user_id}' is DISABLED.")
                self._log_audit(
                    AuditEventType.USER_ACCESS_DENIED,
                    AuditResult.FAILURE,
                    f"Access denied: Account '{matched_user.user_id}' is disabled.",
                    admin_id=matched_user.user_id,
                    source=AuditSource.VOICE,
                    metadata={"target_user_id": matched_user.user_id, "role": matched_user.role.value},
                )
                return False, matched_user, "ACCOUNT_DISABLED"

            # Check guest expiration
            if matched_user.role == UserRole.GUEST:
                now_str = current_time_iso or datetime.datetime.now(datetime.timezone.utc).isoformat()
                if matched_user.expires_at is not None and now_str > matched_user.expires_at:
                    logger.warning(
                        f"[USER MANAGER] Authentication rejected: Guest account '{matched_user.user_id}' has expired ({matched_user.expires_at})."
                    )
                    # Mark account expired
                    self._update_user_status(matched_user.user_id, UserStatus.EXPIRED)
                    self._log_audit(
                        AuditEventType.GUEST_ACCOUNT_EXPIRED,
                        AuditResult.WARNING,
                        f"Guest account '{matched_user.user_id}' expired at {matched_user.expires_at}.",
                        admin_id=matched_user.user_id,
                        source=AuditSource.SYSTEM,
                        metadata={"target_user_id": matched_user.user_id, "expires_at": matched_user.expires_at},
                    )
                    self._log_audit(
                        AuditEventType.USER_ACCESS_DENIED,
                        AuditResult.FAILURE,
                        f"Access denied: Guest account '{matched_user.user_id}' has expired.",
                        admin_id=matched_user.user_id,
                        source=AuditSource.VOICE,
                        metadata={"target_user_id": matched_user.user_id},
                    )
                    return False, matched_user, "GUEST_EXPIRED"

            # Check expired status
            if matched_user.status == UserStatus.EXPIRED:
                logger.warning(f"[USER MANAGER] Authentication rejected: Account '{matched_user.user_id}' is EXPIRED.")
                self._log_audit(
                    AuditEventType.USER_ACCESS_DENIED,
                    AuditResult.FAILURE,
                    f"Access denied: Account '{matched_user.user_id}' has expired.",
                    admin_id=matched_user.user_id,
                    source=AuditSource.VOICE,
                )
                return False, matched_user, "GUEST_EXPIRED"

            # SUCCESSFUL AUTHENTICATION
            now_iso = current_time_iso or datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._update_last_access(matched_user.user_id, now_iso)
            logger.info(
                f"[USER MANAGER] Authentication SUCCESS for '{matched_user.user_id}' ({matched_user.username}, Role: {matched_user.role.value})."
            )
            self._log_audit(
                AuditEventType.USER_ACCESS_GRANTED,
                AuditResult.SUCCESS,
                f"Access granted to '{matched_user.username}' (Role: {matched_user.role.value}).",
                admin_id=matched_user.user_id,
                source=AuditSource.VOICE,
                metadata={"user_id": matched_user.user_id, "role": matched_user.role.value},
            )
            return True, matched_user, "AUTHENTICATION_SUCCESS"

    def verify_user_credentials(
        self,
        user_id: str,
        candidate_pin: str,
    ) -> Tuple[bool, Optional[UserRecord], str]:
        """
        Cryptographically verifies credentials for a specific user ID and candidate PIN.
        Enforces active account status and constant-time PBKDF2 hash comparison.
        Returns:
            (True, UserRecord, "SUCCESS")
            (False, None, "USER_NOT_FOUND")
            (False, UserRecord, "ACCOUNT_NOT_ACTIVE")
            (False, UserRecord, "INCORRECT_CREDENTIALS")
        """
        with self._lock:
            user = self.get_user(user_id)
            if user is None:
                return False, None, "USER_NOT_FOUND"
            if user.status != UserStatus.ACTIVE:
                return False, user, "ACCOUNT_NOT_ACTIVE"

            salt = bytes.fromhex(user.salt_hex)
            expected_hash = bytes.fromhex(user.pin_hash)
            derived_hash = hashlib.pbkdf2_hmac(
                hash_name=user.hash_algorithm,
                password=candidate_pin.encode("utf-8"),
                salt=salt,
                iterations=user.hash_iterations,
            )
            if hmac.compare_digest(derived_hash, expected_hash):
                return True, user, "SUCCESS"
            else:
                return False, user, "INCORRECT_CREDENTIALS"

    # ==========================================================================
    # Administrative Actions (Requires ADMIN Authorization)
    # ==========================================================================

    def _require_admin(self, admin_user_id: str) -> UserRecord:
        """Validates that the acting user has active ADMIN privileges."""
        admin = self.get_user(admin_user_id)
        if admin is None:
            self._log_audit(
                AuditEventType.UNAUTHORIZED_OP,
                AuditResult.WARNING,
                f"Administrative action rejected: User '{admin_user_id}' does not exist.",
                admin_id=admin_user_id,
            )
            raise PermissionError(f"Administrative access denied: User '{admin_user_id}' does not exist.")
        if admin.status != UserStatus.ACTIVE:
            self._log_audit(
                AuditEventType.UNAUTHORIZED_OP,
                AuditResult.WARNING,
                f"Administrative action rejected: Admin account '{admin_user_id}' is not active.",
                admin_id=admin_user_id,
            )
            raise PermissionError(f"Administrative access denied: Admin account '{admin_user_id}' is not active.")
        if admin.role != UserRole.ADMIN:
            self._log_audit(
                AuditEventType.UNAUTHORIZED_OP,
                AuditResult.WARNING,
                f"Administrative action rejected: User '{admin_user_id}' lacks ADMIN privileges (Role: {admin.role.value}).",
                admin_id=admin_user_id,
            )
            raise PermissionError(
                f"Administrative access denied: User '{admin_user_id}' lacks ADMIN privileges (Role: {admin.role.value})."
            )
        return admin

    def create_user(
        self,
        admin_user_id: str,
        user_id: str,
        username: str,
        pin: str,
        role: UserRole | str,
        expires_at: Optional[str] = None,
    ) -> UserRecord:
        """
        Provisions a new user account under administrator authorization.
        Enforces PIN policy and strict duplicate PIN rejection.
        """
        with self._lock:
            admin = self._require_admin(admin_user_id)

            # Check if user_id already exists
            if self.get_user(user_id) is not None:
                raise ValueError(f"User ID '{user_id}' is already registered.")

            # Validate PIN policy
            self._validate_pin_policy(pin)

            # Enforce duplicate PIN rejection
            if self._is_duplicate_pin(pin):
                raise ValueError(
                    "PIN policy violation: Duplicate PINs are rejected. This PIN is already assigned to an existing account."
                )

            # Parse role
            target_role = UserRole(role) if isinstance(role, str) else role

            # Guest expiration validation
            if target_role == UserRole.GUEST and expires_at is None:
                # Default guest expiration: 24 hours from creation
                expires_at = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)).isoformat()

            # Hash credentials
            salt = os.urandom(self.config.salt_bytes)
            pin_hash = hashlib.pbkdf2_hmac(
                hash_name=self.config.hash_algorithm,
                password=pin.encode("utf-8"),
                salt=salt,
                iterations=self.config.hash_iterations,
            ).hex()

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            new_user = UserRecord(
                user_id=user_id,
                username=username,
                pin_hash=pin_hash,
                salt_hex=salt.hex(),
                role=target_role,
                status=UserStatus.ACTIVE,
                created_at=now_iso,
                expires_at=expires_at,
                last_access_at=None,
                hash_algorithm=self.config.hash_algorithm,
                hash_iterations=self.config.hash_iterations,
            )

            self._insert_user_record(new_user)

            self._log_audit(
                AuditEventType.ADMIN_CREATED_USER,
                AuditResult.SUCCESS,
                f"Administrator '{admin.username}' created user '{username}' ({user_id}, Role: {target_role.value}).",
                admin_id=admin.user_id,
                metadata={"created_user_id": user_id, "role": target_role.value},
            )
            logger.info(f"[USER MANAGER] Administrator '{admin.user_id}' created user '{user_id}'.")
            return new_user

    def disable_user(self, admin_user_id: str, target_user_id: str) -> UserRecord:
        """Disables a user account, preventing subsequent voice authentications."""
        with self._lock:
            admin = self._require_admin(admin_user_id)
            target = self.get_user(target_user_id)
            if target is None:
                raise ValueError(f"User '{target_user_id}' not found.")

            # Prevent disabling the sole remaining active admin
            if target.role == UserRole.ADMIN:
                active_admins = [u for u in self.get_all_users() if u.role == UserRole.ADMIN and u.status == UserStatus.ACTIVE]
                if len(active_admins) <= 1 and target.user_id in [u.user_id for u in active_admins]:
                    raise ValueError("Operation rejected: Cannot disable the sole active administrator account.")

            self._update_user_status(target_user_id, UserStatus.DISABLED)
            updated = self.get_user(target_user_id)

            self._log_audit(
                AuditEventType.ADMIN_DISABLED_USER,
                AuditResult.SUCCESS,
                f"Administrator '{admin.username}' disabled account '{target.username}' ({target_user_id}).",
                admin_id=admin.user_id,
                metadata={"target_user_id": target_user_id},
            )
            return updated

    def enable_user(self, admin_user_id: str, target_user_id: str) -> UserRecord:
        """Enables a previously disabled user account."""
        with self._lock:
            admin = self._require_admin(admin_user_id)
            target = self.get_user(target_user_id)
            if target is None:
                raise ValueError(f"User '{target_user_id}' not found.")

            self._update_user_status(target_user_id, UserStatus.ACTIVE)
            updated = self.get_user(target_user_id)

            self._log_audit(
                AuditEventType.ADMIN_ENABLED_USER,
                AuditResult.SUCCESS,
                f"Administrator '{admin.username}' enabled account '{target.username}' ({target_user_id}).",
                admin_id=admin.user_id,
                metadata={"target_user_id": target_user_id},
            )
            return updated

    def remove_user(self, admin_user_id: str, target_user_id: str) -> bool:
        """Permanently removes a user account."""
        with self._lock:
            admin = self._require_admin(admin_user_id)
            target = self.get_user(target_user_id)
            if target is None:
                raise ValueError(f"User '{target_user_id}' not found.")

            # Prevent removing the sole active admin
            if target.role == UserRole.ADMIN:
                active_admins = [u for u in self.get_all_users() if u.role == UserRole.ADMIN and u.status == UserStatus.ACTIVE]
                if len(active_admins) <= 1 and target.user_id in [u.user_id for u in active_admins]:
                    raise ValueError("Operation rejected: Cannot remove the sole active administrator account.")

            conn = None
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("DELETE FROM users WHERE user_id = ?;", (target_user_id,))
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

            self._log_audit(
                AuditEventType.ADMIN_REMOVED_USER,
                AuditResult.SUCCESS,
                f"Administrator '{admin.username}' deleted account '{target.username}' ({target_user_id}).",
                admin_id=admin.user_id,
                metadata={"target_user_id": target_user_id},
            )
            return True

    def change_user_role(self, admin_user_id: str, target_user_id: str, new_role: UserRole | str) -> UserRecord:
        """Changes the role assigned to a user account."""
        with self._lock:
            admin = self._require_admin(admin_user_id)
            target = self.get_user(target_user_id)
            if target is None:
                raise ValueError(f"User '{target_user_id}' not found.")

            parsed_role = UserRole(new_role) if isinstance(new_role, str) else new_role

            # Prevent demoting the sole active admin
            if target.role == UserRole.ADMIN and parsed_role != UserRole.ADMIN:
                active_admins = [u for u in self.get_all_users() if u.role == UserRole.ADMIN and u.status == UserStatus.ACTIVE]
                if len(active_admins) <= 1 and target.user_id in [u.user_id for u in active_admins]:
                    raise ValueError("Operation rejected: Cannot demote the sole active administrator account.")

            conn = None
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("UPDATE users SET role = ? WHERE user_id = ?;", (parsed_role.value, target_user_id))
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

            updated = self.get_user(target_user_id)
            self._log_audit(
                AuditEventType.ADMIN_CHANGED_ROLE,
                AuditResult.SUCCESS,
                f"Administrator '{admin.username}' changed role of '{target.username}' to {parsed_role.value}.",
                admin_id=admin.user_id,
                metadata={"target_user_id": target_user_id, "new_role": parsed_role.value},
            )
            return updated

    def reset_user_pin(self, admin_user_id: str, target_user_id: str, new_pin: str) -> UserRecord:
        """
        Securely updates a user's authentication PIN under administrator authorization.
        Enforces policy and duplicate PIN rejection. Discards plaintext PIN immediately.
        """
        with self._lock:
            admin = self._require_admin(admin_user_id)
            target = self.get_user(target_user_id)
            if target is None:
                raise ValueError(f"User '{target_user_id}' not found.")

            self._validate_pin_policy(new_pin)

            # Check duplicate against all OTHER users
            if self._is_duplicate_pin(new_pin, exclude_user_id=target_user_id):
                raise ValueError(
                    "PIN policy violation: Duplicate PINs are rejected. This PIN is already assigned to an existing account."
                )

            # Generate fresh salt and hash
            salt = os.urandom(self.config.salt_bytes)
            pin_hash = hashlib.pbkdf2_hmac(
                hash_name=self.config.hash_algorithm,
                password=new_pin.encode("utf-8"),
                salt=salt,
                iterations=self.config.hash_iterations,
            ).hex()

            conn = None
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("""
                    UPDATE users SET pin_hash = ?, salt_hex = ? WHERE user_id = ?;
                    """, (pin_hash, salt.hex(), target_user_id))
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

            self._log_audit(
                AuditEventType.ADMIN_RESET_PIN,
                AuditResult.SUCCESS,
                f"Administrator '{admin.username}' reset credentials for user '{target.username}' ({target_user_id}).",
                admin_id=admin.user_id,
                metadata={"target_user_id": target_user_id},
            )
            return self.get_user(target_user_id)

    def set_guest_expiration(self, admin_user_id: str, target_user_id: str, expires_at: str) -> UserRecord:
        """Sets or extends the expiration timestamp for a guest account."""
        with self._lock:
            admin = self._require_admin(admin_user_id)
            target = self.get_user(target_user_id)
            if target is None:
                raise ValueError(f"User '{target_user_id}' not found.")

            conn = None
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("UPDATE users SET expires_at = ? WHERE user_id = ?;", (expires_at, target_user_id))
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

            self._log_audit(
                AuditEventType.ADMIN_ACTION,
                AuditResult.SUCCESS,
                f"Administrator '{admin.username}' updated expiration for '{target.username}' to {expires_at}.",
                admin_id=admin.user_id,
                metadata={"target_user_id": target_user_id, "expires_at": expires_at},
            )
            return self.get_user(target_user_id)

    # ==========================================================================
    # Internal Helpers
    # ==========================================================================

    def _insert_user_record(self, user: UserRecord):
        """Inserts a new user record into the database."""
        conn = None
        try:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                INSERT INTO users (
                    user_id, username, pin_hash, salt_hex, role, status,
                    created_at, expires_at, last_access_at, hash_algorithm, hash_iterations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    user.user_id,
                    user.username,
                    user.pin_hash,
                    user.salt_hex,
                    user.role.value,
                    user.status.value,
                    user.created_at,
                    user.expires_at,
                    user.last_access_at,
                    user.hash_algorithm,
                    user.hash_iterations,
                ))
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _update_user_status(self, user_id: str, status: UserStatus):
        """Updates account operational status."""
        conn = None
        try:
            conn = self._get_connection()
            with conn:
                conn.execute("UPDATE users SET status = ? WHERE user_id = ?;", (status.value, user_id))
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _update_last_access(self, user_id: str, access_time_iso: str):
        """Updates the last successful access timestamp for a user."""
        conn = None
        try:
            conn = self._get_connection()
            with conn:
                conn.execute("UPDATE users SET last_access_at = ? WHERE user_id = ?;", (access_time_iso, user_id))
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

