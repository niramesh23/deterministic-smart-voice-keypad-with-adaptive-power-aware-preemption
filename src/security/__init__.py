"""
src/security package
--------------------
Voice-Based Smart Access Control, PIN Management, and Persistent Audit Logging Subsystem.
"""

from src.security.config import SecurityConfig
from src.security.pin_manager import PINManager
from src.security.lock_controller import (
    AccessResult,
    LockController,
    LockState,
    LockStateSnapshot,
)
from src.security.audit_logger import (
    AuditLogger,
    AuditRecord,
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
from src.security.emergency_handler import (
    EmergencyHandler,
    EmergencyStatus,
)

__all__ = [
    "SecurityConfig",
    "PINManager",
    "LockController",
    "LockState",
    "LockStateSnapshot",
    "AccessResult",
    "AuditLogger",
    "AuditRecord",
    "AuditEventType",
    "AuditResult",
    "AuditSource",
    "UserManager",
    "UserRecord",
    "UserRole",
    "UserStatus",
    "EmergencyHandler",
    "EmergencyStatus",
]
