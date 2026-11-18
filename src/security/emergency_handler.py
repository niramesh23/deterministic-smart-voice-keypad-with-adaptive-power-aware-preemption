"""
src/security/emergency_handler.py
---------------------------------
High-Priority Emergency Stop and Interlock Management Subsystem.

Primary Objectives:
1. High-Priority Preemption: Valid STOP commands take absolute priority over normal
   access-control requests, pending authentication, and background reporting.
2. Race-Condition Defense: Prevents stale in-flight or delayed unlock operations from
   ever unlatching the door following an emergency STOP.
3. Lockout Policy Preservation: Ensures emergency STOP does NOT inadvertently disable
   tamper lockout, reset failure counters, or bypass security penalties.
4. Non-Blocking Execution: Audio capture loops and real-time scheduling remain unblocked.
5. Persistent Audit Trail: Records every accepted, rejected, or completed emergency action.
6. Telemetry & Monitoring: Provides structured operational status for dashboard and APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.application.command_fsm import CommandRecord

from src.security.lock_controller import (
    AccessResult,
    LockController,
    LockState,
    LockStateSnapshot,
)
from src.security.audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditResult,
    AuditSource,
)

logger = logging.getLogger("rtos.security.emergency")


@dataclass
class EmergencyStatus:
    """Structured telemetry data representing current emergency subsystem status."""
    is_active: bool
    last_stop_timestamp: Optional[str]
    last_stop_source: Optional[str]
    last_stop_reason: Optional[str]
    last_lock_state: str
    pending_cancelled: bool
    total_received: int
    total_accepted: int
    total_rejected: int
    total_pending_cancelled: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_active": self.is_active,
            "last_stop_timestamp": self.last_stop_timestamp,
            "last_stop_source": self.last_stop_source,
            "last_stop_reason": self.last_stop_reason,
            "last_lock_state": self.last_lock_state,
            "pending_cancelled": self.pending_cancelled,
            "total_received": self.total_received,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "total_pending_cancelled": self.total_pending_cancelled,
        }


class EmergencyHandler:
    """
    Coordinates high-priority STOP event verification, lock preemption,
    stale unlock suppression, and persistent security auditing.
    """

    def __init__(
        self,
        lock_controller: LockController,
        audit_logger: Optional[AuditLogger] = None,
        min_confidence_threshold: float = 0.65,
        min_margin_threshold: float = 0.15,
        time_func: Optional[Callable[[], float]] = None,
    ):
        self.lock_controller = lock_controller
        self.audit_logger = audit_logger or getattr(lock_controller, "audit_logger", None)
        self.min_confidence_threshold = min_confidence_threshold
        self.min_margin_threshold = min_margin_threshold
        self._time_func = time_func or time.time
        self._lock = threading.RLock()

        # Internal state metrics
        self._emergency_active: bool = False
        self._last_stop_event: Optional[Dict[str, Any]] = None
        self._total_stops_received: int = 0
        self._total_stops_accepted: int = 0
        self._total_stops_rejected: int = 0
        self._total_pending_cancelled: int = 0

    @property
    def is_emergency_active(self) -> bool:
        """Returns True if emergency stop status is currently active."""
        with self._lock:
            return self._emergency_active

    def _log_audit(
        self,
        event_type: AuditEventType | str,
        result: AuditResult | str,
        reason: str,
        source: AuditSource | str = AuditSource.VOICE,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Asynchronously dispatches an audit event without blocking emergency handling."""
        if self.audit_logger is not None:
            try:
                self.audit_logger.log(
                    event_type=event_type,
                    result=result,
                    reason=reason,
                    lock_state=self.lock_controller.state.value,
                    source=source,
                    user_id=user_id or "system",
                    username=username or "System",
                    command_type="stop",
                    metadata=metadata or {},
                )
            except Exception as e:
                logger.error(f"[EMERGENCY HANDLER] Failed to dispatch audit log: {e}")

    def handle_voice_stop(self, record: CommandRecord) -> Tuple[bool, LockStateSnapshot, str]:
        """
        Processes a candidate or confirmed STOP command from the speech recognition pipeline.
        Enforces confidence and verification thresholds to prevent arbitrary low-confidence triggering.
        Returns:
            (accepted: bool, snapshot: LockStateSnapshot, decision: str)
        """
        with self._lock:
            self._total_stops_received += 1

            if record.command != "stop":
                return False, self.lock_controller.get_snapshot(), "IGNORED_NON_STOP"

            # ------------------------------------------------------------------
            # Verification Guard: Reject Low-Confidence or Low-Margin STOPs
            # ------------------------------------------------------------------
            if (
                not record.is_confirmed
                or record.confidence < self.min_confidence_threshold
                or record.margin < self.min_margin_threshold
            ):
                self._total_stops_rejected += 1
                reason = (
                    f"Spoken STOP command rejected: confidence ({record.confidence:.2f} < {self.min_confidence_threshold:.2f}) "
                    f"or margin ({record.margin:.2f} < {self.min_margin_threshold:.2f}) insufficient."
                )
                logger.warning(f"[EMERGENCY HANDLER] {reason}")
                self._log_audit(
                    AuditEventType.EMERGENCY_STOP_REJECTED,
                    AuditResult.WARNING,
                    reason,
                    source=AuditSource.VOICE,
                    metadata={
                        "confidence": round(record.confidence, 4),
                        "margin": round(record.margin, 4),
                        "threshold": self.min_confidence_threshold,
                        "record_id": record.record_id,
                    },
                )
                return False, self.lock_controller.get_snapshot(), "REJECTED_LOW_CONFIDENCE"

            # ------------------------------------------------------------------
            # Valid Confirmed STOP Command: Submit High-Priority Lock Request
            # ------------------------------------------------------------------
            reason_str = (
                f"Voice STOP confirmed (Conf: {record.confidence*100:.1f}%, Margin: {record.margin:.2f}, "
                f"Reaction: {record.reaction_latency_ms}ms)."
            )
            snap, had_pending = self.lock_controller.emergency_stop(
                source=AuditSource.VOICE,
                reason=reason_str,
            )

            self._emergency_active = True
            self._total_stops_accepted += 1
            if had_pending:
                self._total_pending_cancelled += 1

            now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            self._last_stop_event = {
                "timestamp": now_iso,
                "source": "VOICE",
                "confidence": round(record.confidence, 4),
                "margin": round(record.margin, 4),
                "reaction_latency_ms": record.reaction_latency_ms,
                "pending_cancelled": had_pending,
                "lock_state": snap.state.value,
                "status": "ACCEPTED",
                "reason": reason_str,
            }

            logger.info(
                f"[EMERGENCY HANDLER] Successfully executed voice STOP. "
                f"State: {snap.state.value} | Pending Cancelled: {had_pending}"
            )
            return True, snap, "ACCEPTED"

    def trigger_emergency_stop(
        self,
        source: AuditSource = AuditSource.SYSTEM,
        reason: str = "Manual emergency stop triggered.",
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Tuple[bool, LockStateSnapshot]:
        """
        Directly initiates an emergency lock request from an authorized programmatic
        or dashboard control interface.
        """
        with self._lock:
            self._total_stops_received += 1
            snap, had_pending = self.lock_controller.emergency_stop(
                source=source,
                reason=reason,
                user_id=user_id,
                username=username,
            )

            self._emergency_active = True
            self._total_stops_accepted += 1
            if had_pending:
                self._total_pending_cancelled += 1

            now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            self._last_stop_event = {
                "timestamp": now_iso,
                "source": str(source.value) if hasattr(source, "value") else str(source),
                "user_id": user_id or "system",
                "username": username or "System",
                "pending_cancelled": had_pending,
                "lock_state": snap.state.value,
                "status": "ACCEPTED",
                "reason": reason,
            }

            logger.info(
                f"[EMERGENCY HANDLER] Manual emergency STOP executed ({source}). "
                f"State: {snap.state.value} | Pending Cancelled: {had_pending}"
            )
            return True, snap

    def get_emergency_status(self) -> Dict[str, Any]:
        """Returns unified dictionary representation for status providers and dashboard."""
        with self._lock:
            status = EmergencyStatus(
                is_active=self._emergency_active,
                last_stop_timestamp=self._last_stop_event.get("timestamp") if self._last_stop_event else None,
                last_stop_source=self._last_stop_event.get("source") if self._last_stop_event else None,
                last_stop_reason=self._last_stop_event.get("reason") if self._last_stop_event else None,
                last_lock_state=self.lock_controller.state.value,
                pending_cancelled=self._last_stop_event.get("pending_cancelled", False) if self._last_stop_event else False,
                total_received=self._total_stops_received,
                total_accepted=self._total_stops_accepted,
                total_rejected=self._total_stops_rejected,
                total_pending_cancelled=self._total_pending_cancelled,
            )
            data = status.to_dict()
            data["last_stop_event"] = self._last_stop_event
            return data

    def clear_emergency_flag(self):
        """Clears the active emergency alert flag (operator reset)."""
        with self._lock:
            self._emergency_active = False
            self.lock_controller.clear_emergency()
            logger.info("[EMERGENCY HANDLER] Emergency status flag cleared.")
