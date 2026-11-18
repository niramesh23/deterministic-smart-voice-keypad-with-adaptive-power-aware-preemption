"""
src/fault/__init__.py
---------------------
Package entry point for Stage 5: Fault Detection and Safe Recovery.
"""

from src.fault.types import (
    FaultSeverity,
    FaultCategory,
    SystemHealthState,
    ComponentHealth,
    FaultType,
    FaultEvent,
    RecoveryRecord,
    DEFAULT_SEVERITY_MAP,
)
from src.fault.config import FaultDetectorConfig, RecoveryConfig
from src.fault.detector import FaultDetector
from src.fault.recovery import RecoveryManager

__all__ = [
    "FaultSeverity",
    "FaultCategory",
    "SystemHealthState",
    "ComponentHealth",
    "FaultType",
    "FaultEvent",
    "RecoveryRecord",
    "DEFAULT_SEVERITY_MAP",
    "FaultDetectorConfig",
    "RecoveryConfig",
    "FaultDetector",
    "RecoveryManager",
]

