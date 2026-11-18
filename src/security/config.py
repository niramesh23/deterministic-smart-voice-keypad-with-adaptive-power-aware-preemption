"""
src/security/config.py
----------------------
Configuration dataclass for Voice-Based Smart Access Control & PIN Authentication.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SecurityConfig:
    """Configurable security, PIN authentication, and lock timing parameters."""
    # Required PIN length (default 4 digits)
    pin_length: int = 4

    # Path to persistent credentials JSON file containing only salt and PBKDF2 hash
    credentials_path: str = "configs/security_credentials.json"

    # Optional initial plaintext PIN supplied exclusively for programmatic tests/bootstrapping
    # (Scrubbed from memory immediately upon salted key derivation; never written to config files)
    initial_demo_pin: Optional[str] = None

    # Maximum consecutive failed PIN attempts before activating LOCKOUT
    max_failed_attempts: int = 3

    # PIN entry timeout in seconds (clears buffer if inactive)
    pin_timeout_s: float = 10.0

    # Automatic unlock duration in seconds (relocks automatically after expiration)
    unlock_duration_s: float = 5.0

    # Lockout duration in seconds following 3 consecutive failures
    lockout_duration_s: float = 30.0

    # Cryptographic parameters for salted PBKDF2 key derivation
    hash_algorithm: str = "sha256"
    hash_iterations: int = 100_000
    salt_bytes: int = 16

    # Persistent SQLite security audit logging parameters
    audit_db_path: str = "data/security_audit.db"
    audit_queue_capacity: int = 1000
    enable_audit_logging: bool = True

    # User Management and Role-Based Access Control (RBAC) parameters
    users_db_path: str = "data/users.db"
    default_admin_id: str = "admin_01"
    default_admin_name: str = "Administrator"

    # Explicit development/demo flag: when True, permits bootstrap of demo administrator credentials
    # Default is False to ensure normal application startup fails closed without creating default accounts
    allow_demo_bootstrap: bool = False

    # Administrative Dashboard login rate-limiting parameters
    max_login_failures: int = 5
    login_cooldown_seconds: float = 60.0

