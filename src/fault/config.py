"""
src/fault/config.py
-------------------
Configurable parameters and threshold specifications for Fault Detection
and Recovery Subsystems.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.fault.types import FaultType, FaultSeverity, DEFAULT_SEVERITY_MAP


@dataclass
class FaultDetectorConfig:
    """Configurable thresholds for detecting system anomalies and threshold breaches."""

    # --- Audio Subsystem Thresholds ---
    queue_occupancy_warning_threshold: int = 15     # Blocks in audio queue before WARNING
    queue_overflow_drop_warning: int = 1            # Single drop triggers WARNING
    repeated_drops_threshold: int = 5               # >= 5 dropped audio blocks triggers ERROR
    max_audio_read_failures: int = 3                # Consecutive read errors triggers ERROR
    empty_block_threshold: int = 3                  # Consecutive empty/zero-len blocks triggers WARNING

    # --- Processing & Real-Time Scheduler Thresholds ---
    max_inference_latency_ms: float = 75.0          # Inference exceeding 75ms triggers WARNING
    max_consecutive_deadline_misses: int = 3        # >= 3 consecutive misses triggers WARNING
    worker_heartbeat_timeout_s: float = 3.0         # Scheduler/Worker silent > 3s triggers CRITICAL
    max_feature_extraction_errors: int = 2          # Consecutive FFT/Spectrogram errors triggers ERROR

    # --- Application & Persistence Thresholds ---
    max_db_retry_attempts: int = 3                  # Bounded database write retries
    db_retry_backoff_s: float = 0.05                # Delay between retries
    dashboard_health_timeout_s: float = 2.0         # Unresponsive dashboard check threshold

    # --- Security Subsystem Thresholds ---
    max_failed_auth_attempts: int = 3               # Matches tamper lockout threshold
    max_repeated_invalid_commands: int = 5          # Spurious commands threshold

    # --- Rate Limiting & Debouncing ---
    fault_dedup_window_s: float = 1.0               # Window to suppress repeated identical alerts

    # Custom severity overrides
    severity_overrides: Dict[FaultType, FaultSeverity] = field(default_factory=dict)

    def get_severity(self, fault_type: FaultType) -> FaultSeverity:
        """Returns the configured severity for a fault type, applying overrides if set."""
        if fault_type in self.severity_overrides:
            return self.severity_overrides[fault_type]
        return DEFAULT_SEVERITY_MAP.get(fault_type, FaultSeverity.WARNING)


@dataclass
class RecoveryConfig:
    """Configurable parameters governing automated and supervised recovery policies."""

    max_recovery_attempts: int = 3                  # Max automated retry attempts per component
    recovery_cooloff_s: float = 0.5                 # Cool-off period before retrying recovery
    stale_queue_drain_on_audio_recovery: bool = True # Clear audio queue backlog on mic reinit
    fail_safe_lock_on_critical: bool = True         # Return lock to LOCKED on CRITICAL fault
    verify_inference_with_synthetic_tensor: bool = True # Run test tensor to verify ML engine
    auto_recover_dashboard: bool = True             # Try re-binding dashboard on port collision
    max_db_retry_attempts: int = 3                  # Bounded database write retries
    db_retry_backoff_s: float = 0.05                # Delay between DB retries
