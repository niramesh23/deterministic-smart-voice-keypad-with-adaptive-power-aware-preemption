"""
src/fault/types.py
------------------
Data structures, enums, and models for Fault Detection and Safe Recovery.

Defines:
- FaultSeverity: INFO, WARNING, ERROR, CRITICAL
- FaultCategory: AUDIO, PROCESSING, APPLICATION, SECURITY
- SystemHealthState: HEALTHY, DEGRADED, RECOVERING, FAULT
- ComponentHealth: OPERATIONAL, DEGRADED, RECOVERING, FAULT, UNAVAILABLE
- FaultType: Enumeration of all categorized operational, audio, processing, and security faults
- FaultEvent & RecoveryRecord: Structured telemetry models
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from enum import Enum
import time
from typing import Any, Dict, List, Optional


class FaultSeverity(str, Enum):
    """Hierarchical severity levels for detected operational and security faults."""
    INFO = "INFO"          # Informational anomaly; no immediate impact (e.g. single frame jitter)
    WARNING = "WARNING"    # Degraded operation; system still functional (e.g. queue occupancy elevated)
    ERROR = "ERROR"        # Component failure; feature unavailable, recoverable (e.g. mic stream drop)
    CRITICAL = "CRITICAL"  # System-wide or security failure requiring safe fail-closed response


class FaultCategory(str, Enum):
    """Categorized subsystem origin of faults."""
    AUDIO = "AUDIO"
    PROCESSING = "PROCESSING"
    APPLICATION = "APPLICATION"
    SECURITY = "SECURITY"


class SystemHealthState(str, Enum):
    """
    Global system-level health status.
    Kept separate from lock controller state (e.g., lock may be LOCKED while health is DEGRADED).
    """
    HEALTHY = "HEALTHY"        # All components operating within normal bounds
    DEGRADED = "DEGRADED"      # Operating with non-critical warnings or fallback modes
    RECOVERING = "RECOVERING"  # Active automated or supervised recovery action in flight
    FAULT = "FAULT"            # Critical failure or unrecovered error requiring fail-safe lockdown


class ComponentHealth(str, Enum):
    """Health status of an individual subsystem component."""
    OPERATIONAL = "OPERATIONAL"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    FAULT = "FAULT"
    UNAVAILABLE = "UNAVAILABLE"


class FaultType(str, Enum):
    """Comprehensive taxonomy of detectable operational and security faults."""
    # --- Audio Faults ---
    MIC_INIT_FAILURE = "MIC_INIT_FAILURE"
    MIC_DISCONNECTED = "MIC_DISCONNECTED"
    AUDIO_READ_FAILURE = "AUDIO_READ_FAILURE"
    INVALID_AUDIO_BLOCK = "INVALID_AUDIO_BLOCK"
    EXCESSIVE_QUEUE_OCCUPANCY = "EXCESSIVE_QUEUE_OCCUPANCY"
    AUDIO_QUEUE_OVERFLOW = "AUDIO_QUEUE_OVERFLOW"
    REPEATED_AUDIO_DROPS = "REPEATED_AUDIO_DROPS"

    # --- Processing Faults ---
    EXCESSIVE_INFERENCE_LATENCY = "EXCESSIVE_INFERENCE_LATENCY"
    FEATURE_EXTRACTION_FAILURE = "FEATURE_EXTRACTION_FAILURE"
    TASK_EXECUTION_EXCEPTION = "TASK_EXECUTION_EXCEPTION"
    REPEATED_DEADLINE_MISSES = "REPEATED_DEADLINE_MISSES"
    UNRESPONSIVE_WORKER = "UNRESPONSIVE_WORKER"
    INFERENCE_EXCEPTION = "INFERENCE_EXCEPTION"

    # --- Application Faults ---
    DATABASE_CONNECTION_FAILURE = "DATABASE_CONNECTION_FAILURE"
    AUDIT_WORKER_FAILURE = "AUDIT_WORKER_FAILURE"
    DASHBOARD_UNAVAILABLE = "DASHBOARD_UNAVAILABLE"
    CONFIG_LOAD_FAILURE = "CONFIG_LOAD_FAILURE"
    UNEXPECTED_APP_EXCEPTION = "UNEXPECTED_APP_EXCEPTION"

    # --- Security Faults ---
    REPEATED_AUTH_FAILURE = "REPEATED_AUTH_FAILURE"
    UNEXPECTED_LOCK_TRANSITION = "UNEXPECTED_LOCK_TRANSITION"
    UNAUTHORIZED_ADMIN_REQUEST = "UNAUTHORIZED_ADMIN_REQUEST"
    REPEATED_INVALID_COMMANDS = "REPEATED_INVALID_COMMANDS"


# Default severity mapping per fault type
DEFAULT_SEVERITY_MAP: Dict[FaultType, FaultSeverity] = {
    # Audio
    FaultType.MIC_INIT_FAILURE: FaultSeverity.ERROR,
    FaultType.MIC_DISCONNECTED: FaultSeverity.ERROR,
    FaultType.AUDIO_READ_FAILURE: FaultSeverity.WARNING,
    FaultType.INVALID_AUDIO_BLOCK: FaultSeverity.WARNING,
    FaultType.EXCESSIVE_QUEUE_OCCUPANCY: FaultSeverity.WARNING,
    FaultType.AUDIO_QUEUE_OVERFLOW: FaultSeverity.WARNING,
    FaultType.REPEATED_AUDIO_DROPS: FaultSeverity.ERROR,

    # Processing
    FaultType.EXCESSIVE_INFERENCE_LATENCY: FaultSeverity.WARNING,
    FaultType.FEATURE_EXTRACTION_FAILURE: FaultSeverity.ERROR,
    FaultType.TASK_EXECUTION_EXCEPTION: FaultSeverity.ERROR,
    FaultType.REPEATED_DEADLINE_MISSES: FaultSeverity.WARNING,
    FaultType.UNRESPONSIVE_WORKER: FaultSeverity.CRITICAL,
    FaultType.INFERENCE_EXCEPTION: FaultSeverity.ERROR,

    # Application
    FaultType.DATABASE_CONNECTION_FAILURE: FaultSeverity.ERROR,
    FaultType.AUDIT_WORKER_FAILURE: FaultSeverity.ERROR,
    FaultType.DASHBOARD_UNAVAILABLE: FaultSeverity.WARNING,
    FaultType.CONFIG_LOAD_FAILURE: FaultSeverity.ERROR,
    FaultType.UNEXPECTED_APP_EXCEPTION: FaultSeverity.CRITICAL,

    # Security
    FaultType.REPEATED_AUTH_FAILURE: FaultSeverity.WARNING,
    FaultType.UNEXPECTED_LOCK_TRANSITION: FaultSeverity.CRITICAL,
    FaultType.UNAUTHORIZED_ADMIN_REQUEST: FaultSeverity.WARNING,
    FaultType.REPEATED_INVALID_COMMANDS: FaultSeverity.INFO,
}


@dataclass
class FaultEvent:
    """Structured record of a detected fault condition."""
    fault_id: int
    fault_type: FaultType
    category: FaultCategory
    severity: FaultSeverity
    component: str
    description: str
    timestamp: float = field(default_factory=time.time)
    iso_time: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    resolution_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "component": self.component,
            "description": self.description,
            "timestamp": round(self.timestamp, 6),
            "iso_time": self.iso_time,
            "metrics": self.metrics,
            "resolved": self.resolved,
            "resolution_timestamp": round(self.resolution_timestamp, 6) if self.resolution_timestamp else None,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class RecoveryRecord:
    """Structured record of an automated or manual recovery action."""
    recovery_id: int
    fault_id: int
    fault_type: FaultType
    component: str
    action_taken: str
    success: bool
    verified_healthy: bool
    timestamp: float = field(default_factory=time.time)
    iso_time: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recovery_id": self.recovery_id,
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "component": self.component,
            "action_taken": self.action_taken,
            "success": self.success,
            "verified_healthy": self.verified_healthy,
            "timestamp": round(self.timestamp, 6),
            "iso_time": self.iso_time,
            "details": self.details,
        }

