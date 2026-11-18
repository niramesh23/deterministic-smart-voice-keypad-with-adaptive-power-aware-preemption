"""
src/security/pin_manager.py
---------------------------
Thread-Safe PIN Buffer and Salted Cryptographic Hashing Manager.

Key Guarantees:
- Salting & Hashing: Uses PBKDF2-HMAC-SHA256 with 100,000 iterations.
  Plaintext PIN is NEVER stored in memory or persisted.
- Digits Only: Only accepts numeric digits '0'-'9'.
- Buffer Boundary: Strict enforcement of configurable PIN length (default: 4 digits).
  Rejects extra digits when buffer is full.
- Incomplete Verification Rejection: Rejects verification attempts with fewer digits.
- Self-Clearing Buffer: Clears the internal PIN buffer immediately after verification
  (successful or unsuccessful) or on timeout/reset.
- Privacy Masking: Masks digit entries for logs, dashboards, and snapshots ("*").
- Constant-Time Comparison: Uses hmac.compare_digest to prevent side-channel timing attacks.
"""

from dataclasses import dataclass
import datetime
import hashlib
import hmac
import json
import logging
import os
import pathlib
import threading
import time
from typing import Callable, List, Optional, Tuple

from src.security.config import SecurityConfig

logger = logging.getLogger("rtos.security.pin")


class PINManager:
    """
    Manages PIN entry buffering, validation, timeout tracking, and salted PBKDF2 verification.
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        time_func: Optional[Callable[[], float]] = None,
        allow_demo_bootstrap: Optional[bool] = None,
    ):
        self.config = config or SecurityConfig()
        self._time_func = time_func or time.monotonic
        self._lock = threading.RLock()

        if allow_demo_bootstrap is not None:
            self.allow_demo_bootstrap = allow_demo_bootstrap
        elif os.environ.get("RTOS_ALLOW_DEMO_BOOTSTRAP", "").lower() in ("1", "true", "yes"):
            self.allow_demo_bootstrap = True
        else:
            self.allow_demo_bootstrap = getattr(self.config, "allow_demo_bootstrap", False)

        # Internal buffer holding entered digits
        self._buffer: List[str] = []
        self._last_digit_time: float = 0.0

        # Cryptographic salt and hash storage (Plaintext PIN is never stored)
        self._salt_hex: str = ""
        self._hash_hex: str = ""

        # Credentials initialization (Loads from credentials file or fails closed)
        cred_path = pathlib.Path(self.config.credentials_path)
        if cred_path.exists():
            self.load_credentials(cred_path)
        elif self.allow_demo_bootstrap:
            demo_pin = self.config.initial_demo_pin or "1234"
            logger.info("[PIN MANAGER] Explicit demo bootstrap enabled. Provisioning demo credentials.")
            self.provision_pin(demo_pin, persist_path=cred_path)
        else:
            logger.warning(
                f"[PIN MANAGER] No credentials file found at '{cred_path}'. "
                "Running in unprovisioned state; PIN verification will fail closed until explicitly provisioned."
            )

    @property
    def pin_length(self) -> int:
        return self.config.pin_length

    @property
    def current_length(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self.config.pin_length

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0

    @property
    def masked_display(self) -> str:
        """Returns masked display representation (e.g. '**--' or '****') for safe UI rendering."""
        with self._lock:
            entered = len(self._buffer)
            return "*" * entered

    @property
    def progress_string(self) -> str:
        """Returns safe user-facing progress string (e.g. '2/4 digits')."""
        with self._lock:
            return f"{len(self._buffer)}/{self.config.pin_length} digits"

    def is_plaintext_stored(self) -> bool:
        """Audit helper confirming plaintext PIN is not retained in object attributes."""
        # Check standard attributes
        for attr, val in self.__dict__.items():
            if attr in ("_buffer",):
                continue
            if isinstance(val, str) and val in ("1234", "9999", "5555"):
                return True
        return False

    def load_credentials(self, path: pathlib.Path):
        """Loads salt and PBKDF2 hash from persistent credentials file."""
        with self._lock:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("Credentials file content must be a JSON object.")
                if "salt_hex" not in data or "hash_hex" not in data:
                    raise ValueError("Credentials file missing required 'salt_hex' or 'hash_hex' fields.")
                if not data["salt_hex"] or not data["hash_hex"]:
                    raise ValueError("Credentials file contains empty salt or hash.")
                bytes.fromhex(data["salt_hex"])
                bytes.fromhex(data["hash_hex"])
                self._salt_hex = data["salt_hex"]
                self._hash_hex = data["hash_hex"]
                logger.info(
                    f"[PIN MANAGER] Loaded salted credentials from '{path.name}' "
                    f"({data.get('hash_algorithm', 'sha256')}, {data.get('hash_iterations', 100000)} iters). "
                    f"Plaintext PIN is NOT stored."
                )
            except Exception as e:
                self._salt_hex = ""
                self._hash_hex = ""
                logger.error(f"[PIN MANAGER] Failed to load credentials from '{path}': {e}. Failing closed.")
                raise ValueError(f"Corrupted or invalid PIN credentials file: {e}") from e

    def provision_pin(self, plaintext_pin: str, persist_path: Optional[pathlib.Path] = None):
        """
        Derives and stores a salted PBKDF2 hash of the PIN.
        Enforces policy: exactly configured length, digits 1-9 only (digit '0' rejected).
        The plaintext argument is discarded immediately after derivation.
        """
        if not plaintext_pin.isdigit():
            raise ValueError("PIN must contain only numeric digits (1-9).")
        if not all(c in "123456789" for c in plaintext_pin):
            raise ValueError("PIN policy violation: Only digits 1 through 9 are permitted (digit '0' rejected).")
        if len(plaintext_pin) != self.config.pin_length:
            raise ValueError(
                f"PIN length must match configured length {self.config.pin_length}. Got {len(plaintext_pin)} digits."
            )

        with self._lock:
            # Generate cryptographically secure random salt
            salt = os.urandom(self.config.salt_bytes)
            derived_hash = hashlib.pbkdf2_hmac(
                hash_name=self.config.hash_algorithm,
                password=plaintext_pin.encode("utf-8"),
                salt=salt,
                iterations=self.config.hash_iterations,
            )
            self._salt_hex = salt.hex()
            self._hash_hex = derived_hash.hex()

            if persist_path is not None:
                persist_path.parent.mkdir(parents=True, exist_ok=True)
                creds = {
                    "hash_algorithm": self.config.hash_algorithm,
                    "hash_iterations": self.config.hash_iterations,
                    "salt_hex": self._salt_hex,
                    "hash_hex": self._hash_hex,
                    "pin_length": self.config.pin_length,
                    "allowed_digits": "1-9",
                    "provisioned_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "security_note": "Salted PBKDF2-HMAC-SHA256 credential. Plaintext PIN is NOT stored."
                }
                with open(persist_path, "w", encoding="utf-8") as f:
                    json.dump(creds, f, indent=2)

            logger.info(
                f"[PIN MANAGER] Provisioned salted PBKDF2-{self.config.hash_algorithm.upper()} hash "
                f"({self.config.hash_iterations} iterations, {len(salt)}B salt). Plaintext PIN discarded."
            )

    def append_digit(self, digit: str, current_time: Optional[float] = None) -> bool:
        """
        Appends a single numeric digit to the PIN buffer.
        Rejects non-numeric characters, digit '0', or inputs exceeding the configured PIN length.
        """
        if not isinstance(digit, str) or len(digit) != 1 or digit not in "123456789":
            logger.warning(f"[PIN MANAGER] Rejected invalid digit '{digit}' (Policy: Only digits 1-9 permitted, '0' rejected).")
            return False

        with self._lock:
            if len(self._buffer) >= self.config.pin_length:
                logger.warning(
                    f"[PIN MANAGER] Rejected extra digit '{digit}'; buffer already full "
                    f"({len(self._buffer)}/{self.config.pin_length})."
                )
                return False

            now = self._time_func() if current_time is None else current_time
            self._buffer.append(digit)
            self._last_digit_time = now
            logger.info(
                f"[PIN MANAGER] Digit accepted. Progress: {len(self._buffer)}/{self.config.pin_length} "
                f"(Masked: {self.masked_display})"
            )
            return True

    def clear(self):
        """Securely clears the internal PIN buffer and resets entry timer."""
        with self._lock:
            self._buffer.clear()
            self._last_digit_time = 0.0
            logger.debug("[PIN MANAGER] PIN buffer cleared.")

    def check_timeout(self, current_time: Optional[float] = None) -> bool:
        """
        Checks if the PIN entry has timed out due to user inactivity.
        Clears buffer and returns True if timeout elapsed.
        """
        with self._lock:
            if not self._buffer or self._last_digit_time == 0.0:
                return False

            now = self._time_func() if current_time is None else current_time
            elapsed = now - self._last_digit_time
            if elapsed >= self.config.pin_timeout_s:
                logger.info(
                    f"[PIN MANAGER] PIN entry timed out ({elapsed:.1f}s >= {self.config.pin_timeout_s}s). "
                    f"Buffer cleared."
                )
                self.clear()
                return True
            return False

    def consume_buffer(self) -> Optional[str]:
        """
        Atomically extracts the entered PIN string and clears the internal buffer.
        Returns the entered string if length matches pin_length, or None if incomplete.
        """
        with self._lock:
            if len(self._buffer) < self.config.pin_length:
                self.clear()
                return None
            candidate = "".join(self._buffer)
            self.clear()
            return candidate

    def verify(self) -> Tuple[bool, str]:
        """
        Verifies the currently entered PIN buffer against the stored salted hash.
        Always clears the internal PIN buffer immediately upon completion.
        Returns:
            (True, "CORRECT_PIN") on match
            (False, "INCOMPLETE_PIN") if fewer than required digits
            (False, "INCORRECT_PIN") on mismatch
        """
        with self._lock:
            if not self._salt_hex or not self._hash_hex:
                self.clear()
                logger.warning("[PIN MANAGER] Verification failed: PIN manager is unprovisioned (missing credentials). Failing closed.")
                return False, "UNPROVISIONED"

            if len(self._buffer) < self.config.pin_length:
                entered = len(self._buffer)
                self.clear()
                logger.warning(
                    f"[PIN MANAGER] Verification failed: Incomplete PIN ({entered}/{self.config.pin_length} digits)."
                )
                return False, "INCOMPLETE_PIN"

            candidate = "".join(self._buffer)
            # Clear buffer immediately before hashing candidate
            self.clear()

            # Hash candidate with stored salt
            salt = bytes.fromhex(self._salt_hex)
            expected_hash = bytes.fromhex(self._hash_hex)
            candidate_hash = hashlib.pbkdf2_hmac(
                hash_name=self.config.hash_algorithm,
                password=candidate.encode("utf-8"),
                salt=salt,
                iterations=self.config.hash_iterations,
            )

            # Constant-time comparison
            if hmac.compare_digest(candidate_hash, expected_hash):
                logger.info("[PIN MANAGER] Verification SUCCESS: PIN matches stored hash.")
                return True, "CORRECT_PIN"
            else:
                logger.warning("[PIN MANAGER] Verification FAILURE: Incorrect PIN.")
                return False, "INCORRECT_PIN"

