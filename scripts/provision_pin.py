"""
scripts/provision_pin.py
------------------------
Secure Provisioning Utility for Voice-Based Smart Lock Credentials.

Responsibilities:
1. Provisions or updates the Smart Lock PIN.
2. Enforces security policy: exactly 4 digits, digits 1-9 only (digit '0' rejected).
3. Generates a cryptographically random 16-byte salt via os.urandom().
4. Derives PBKDF2-HMAC-SHA256 hash using 100,000 iterations.
5. Saves only salt, hash, and metadata to configs/security_credentials.json.
6. Plaintext PIN is never stored in persistent configuration files.
"""

import argparse
import datetime
import getpass
import hashlib
import json
import os
import pathlib
import sys
from typing import Optional

# Ensure project root in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CREDENTIALS_PATH = PROJECT_ROOT / "configs" / "security_credentials.json"


def provision_credentials(
    pin: str,
    output_path: pathlib.Path = DEFAULT_CREDENTIALS_PATH,
    iterations: int = 100_000,
    salt_bytes: int = 16,
) -> pathlib.Path:
    """
    Derives salted PBKDF2 hash for the given PIN and saves to credentials file.
    Validates that PIN contains only digits 1-9.
    """
    if len(pin) != 4:
        raise ValueError(f"PIN must be exactly 4 digits. Got length {len(pin)}.")

    if not all(c in "123456789" for c in pin):
        raise ValueError("PIN policy violation: Only digits 1 through 9 are permitted (digit '0' is rejected).")

    # Generate 16 bytes cryptographically secure salt
    salt = os.urandom(salt_bytes)
    derived_hash = hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=pin.encode("utf-8"),
        salt=salt,
        iterations=iterations,
    )

    credentials = {
        "hash_algorithm": "sha256",
        "hash_iterations": iterations,
        "salt_hex": salt.hex(),
        "hash_hex": derived_hash.hex(),
        "pin_length": len(pin),
        "allowed_digits": "1-9",
        "provisioned_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "security_note": "Salted PBKDF2-HMAC-SHA256 credential. Plaintext PIN is NOT stored."
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(credentials, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Provision Smart Lock Salted PIN Credentials")
    parser.add_argument("--pin", type=str, default=None, help="4-digit PIN (digits 1-9 only). If omitted, prompts securely.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_CREDENTIALS_PATH), help="Destination credentials path")
    parser.add_argument("--demo-default", action="store_true", help="Provision development demo PIN '1234'")
    args = parser.parse_args()

    out_file = pathlib.Path(args.output)

    if args.demo_default:
        pin = "1234"
        print("[PROVISION] Using initial development/demo PIN: '1234'")
    elif args.pin:
        pin = args.pin.strip()
    else:
        pin = getpass.getpass("Enter 4-digit PIN (digits 1-9 only): ").strip()
        confirm = getpass.getpass("Confirm PIN: ").strip()
        if pin != confirm:
            print("Error: PIN entries do not match.", file=sys.stderr)
            sys.exit(1)

    try:
        path = provision_credentials(pin, output_path=out_file)
        print(f"[PROVISION SUCCESS] Salted credentials written to: {path}")
        print(f"Algorithm: PBKDF2-HMAC-SHA256 (100,000 iterations)")
        print(f"Plaintext PIN discarded. Only cryptographic salt and hash persisted.")
    except Exception as e:
        print(f"Error provisioning PIN: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

