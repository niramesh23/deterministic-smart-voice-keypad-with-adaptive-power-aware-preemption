"""
src/fault/detector.py
---------------------
Real-Time Fault Detection, Metric Evaluation, and System Health Monitor.

Responsibilities:
1. Ingests operational telemetry from audio, DSP, ML inference, scheduler, persistence, and security subsystems.
2. Evaluates observations against configurable threshold specifications.
3. Classifies fault severity (INFO, WARNING, ERROR, CRITICAL).
4. Implements rate-limiting and debouncing to prevent event flooding.
5. Computes dynamic system-level health states (HEALTHY, DEGRADED, RECOVERING, FAULT).
6. Dispatches structured audit and telemetry records without blocking critical paths.
"""

from __future__ import annotations

from collections import defaultdict
import datetime
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.fault.types import (
    FaultCategory,
    FaultEvent,
    FaultSeverity,
    FaultType,
    SystemHealthState,
    ComponentHealth,
)
from src.fault.config import FaultDetectorConfig
from src.security.audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditResult,
    AuditSource,
)

logger = logging.getLogger("rtos.fault.detector")


class FaultDetector:
    """
    Thread-safe engine for detecting, debouncing, and tracking operational faults.
    """

    def __init__(
        self,
        config: Optional[FaultDetectorConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        time_func: Optional[Callable[[], float]] = None,
        on_fault_detected: Optional[Callable[[FaultEvent], None]] = None,
        voice_feedback: Optional[Any] = None,
    ):
        self.config = config or FaultDetectorConfig()
        self.audit_logger = audit_logger
        self._time_func = time_func or time.time
        self.on_fault_detected = on_fault_detected
        self.voice_feedback = voice_feedback
        self._lock = threading.RLock()

        # Telemetry & State Tracking
        self._fault_counter: int = 0
        self._active_faults: Dict[int, FaultEvent] = {}
        self._fault_history: List[FaultEvent] = []
        self._last_reported_times: Dict[FaultType, float] = {}
        self._suppressed_counts: Dict[FaultType, int] = defaultdict(int)

        # Subsystem Component Health
        self._component_health: Dict[str, ComponentHealth] = {
            "microphone": ComponentHealth.OPERATIONAL,
            "audio_queue": ComponentHealth.OPERATIONAL,
            "inference": ComponentHealth.OPERATIONAL,
            "scheduler": ComponentHealth.OPERATIONAL,
            "database": ComponentHealth.OPERATIONAL,
            "dashboard": ComponentHealth.OPERATIONAL,
            "lock": ComponentHealth.OPERATIONAL,
        }

        self._overall_health: SystemHealthState = SystemHealthState.HEALTHY
        self._is_recovering: bool = False
        self._lock_controller: Any = None

    def set_lock_controller(self, lock_controller: Any):
        with self._lock:
            self._lock_controller = lock_controller

    def set_voice_feedback(self, voice_feedback: Any):
        with self._lock:
            self.voice_feedback = voice_feedback

    def _get_lock_state(self) -> str:
        if self._lock_controller is not None and hasattr(self._lock_controller, "current_state"):
            return str(self._lock_controller.current_state.value)
        return "LOCKED"

    # ==========================================================================
    # Fault Ingestion & Debouncing
    # ==========================================================================

    def report_fault(
        self,
        fault_type: FaultType,
        component: str,
        description: str,
        severity: Optional[FaultSeverity] = None,
        metrics: Optional[Dict[str, Any]] = None,
        force_immediate: bool = False,
    ) -> Optional[FaultEvent]:
        """
        Ingests and processes a detected anomaly.
        Applies rate-limiting/debouncing unless force_immediate is True or severity is CRITICAL.
        """
        now = self._time_func()
        sev = severity or self.config.get_severity(fault_type)

        with self._lock:
            # 1. Check Debouncing / Rate Limiting (CRITICAL faults always bypass)
            if not force_immediate and sev != FaultSeverity.CRITICAL:
                last_time = self._last_reported_times.get(fault_type, 0.0)
                if now - last_time < self.config.fault_dedup_window_s:
                    self._suppressed_counts[fault_type] += 1
                    logger.debug(
                        f"[FAULT DETECTOR] Debounced repeated fault {fault_type.value} "
                        f"(Suppressed count: {self._suppressed_counts[fault_type]})"
                    )
                    return None

            self._last_reported_times[fault_type] = now
            self._fault_counter += 1

            # Determine category
            category = self._infer_category(fault_type)

            merged_metrics = dict(metrics or {})
            if self._suppressed_counts[fault_type] > 0:
                merged_metrics["suppressed_duplicate_count"] = self._suppressed_counts[fault_type]
                self._suppressed_counts[fault_type] = 0

            event = FaultEvent(
                fault_id=self._fault_counter,
                fault_type=fault_type,
                category=category,
                severity=sev,
                component=component,
                description=description,
                timestamp=now,
                iso_time=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                metrics=merged_metrics,
            )

            # 2. Maintain Active & Historical Fault Lists
            # INFO events are recorded in history but don't stay in active faults
            if sev != FaultSeverity.INFO:
                self._active_faults[event.fault_id] = event

            self._fault_history.append(event)
            if len(self._fault_history) > 200:
                self._fault_history.pop(0)

            # 3. Update Component Health
            self._update_component_status_on_fault(component, sev)

            # 4. Recalculate Overall System Health
            self._recalculate_health()

            # 5. Dispatch Audit Log asynchronously
            self._log_audit_fault(event)

            logger.log(
                logging.CRITICAL if sev == FaultSeverity.CRITICAL else
                logging.ERROR if sev == FaultSeverity.ERROR else
                logging.WARNING if sev == FaultSeverity.WARNING else logging.INFO,
                f"[FAULT DETECTOR] [{sev.value}] {component.upper()}: {description} (Fault #{event.fault_id})"
            )

            # 6. Notify Callback / Recovery Handler
            if self.on_fault_detected is not None:
                try:
                    self.on_fault_detected(event)
                except Exception as e:
                    logger.error(f"[FAULT DETECTOR] Error in on_fault_detected callback: {e}")

            if self.voice_feedback is not None and sev != FaultSeverity.INFO:
                try:
                    self.voice_feedback.on_system_fault_detected()
                except Exception as e:
                    logger.debug(f"[FAULT DETECTOR] Voice feedback fault error: {e}")

            return event

    def resolve_fault(self, fault_id: int, resolution_notes: str = "Verified healthy and resolved") -> bool:
        """
        Marks an active fault as resolved and recalculates system health.
        """
        with self._lock:
            fault = self._active_faults.get(fault_id)
            if fault is None:
                return False

            fault.resolved = True
            fault.resolution_timestamp = self._time_func()
            fault.resolution_notes = resolution_notes
            del self._active_faults[fault_id]

            # Re-evaluate component health
            comp_faults = [f for f in self._active_faults.values() if f.component == fault.component]
            if not comp_faults:
                self._component_health[fault.component] = ComponentHealth.OPERATIONAL
            else:
                highest_sev = max((f.severity for f in comp_faults), key=lambda s: ["INFO", "WARNING", "ERROR", "CRITICAL"].index(s.value))
                self._update_component_status_on_fault(fault.component, highest_sev)

            self._recalculate_health()
            logger.info(f"[FAULT DETECTOR] Fault #{fault_id} ({fault.fault_type.value}) marked RESOLVED: {resolution_notes}")
            return True

    def clear_faults_for_component(self, component: str, resolution_notes: str = "Component recovery verified") -> int:
        """
        Resolves all active faults for a specific component upon successful recovery verification.
        """
        with self._lock:
            resolved_count = 0
            to_resolve = [f_id for f_id, f in self._active_faults.items() if f.component == component]
            for f_id in to_resolve:
                if self.resolve_fault(f_id, resolution_notes):
                    resolved_count += 1
            self._component_health[component] = ComponentHealth.OPERATIONAL
            self._recalculate_health()
            return resolved_count

    def set_recovering(self, is_recovering: bool, component: Optional[str] = None):
        """Notifies detector that an active recovery procedure is underway."""
        with self._lock:
            self._is_recovering = is_recovering
            if component and component in self._component_health:
                if is_recovering:
                    self._component_health[component] = ComponentHealth.RECOVERING
            self._recalculate_health()

    # ==========================================================================
    # Periodic Telemetry Evaluation
    # ==========================================================================

    def evaluate_audio_queue(self, occupancy: int, capacity: int, dropped_blocks: int):
        """Evaluates audio queue depth and dropped blocks against configured thresholds."""
        with self._lock:
            if occupancy >= self.config.queue_occupancy_warning_threshold:
                self.report_fault(
                    fault_type=FaultType.EXCESSIVE_QUEUE_OCCUPANCY,
                    component="audio_queue",
                    description=f"Audio queue occupancy high: {occupancy}/{capacity} blocks.",
                    metrics={"occupancy": occupancy, "capacity": capacity},
                )
            if dropped_blocks >= self.config.repeated_drops_threshold:
                self.report_fault(
                    fault_type=FaultType.REPEATED_AUDIO_DROPS,
                    component="audio_queue",
                    description=f"Repeated audio block drops detected: {dropped_blocks} total dropped.",
                    metrics={"dropped_blocks": dropped_blocks},
                )
            elif dropped_blocks >= self.config.queue_overflow_drop_warning:
                self.report_fault(
                    fault_type=FaultType.AUDIO_QUEUE_OVERFLOW,
                    component="audio_queue",
                    description=f"Audio block dropped due to queue saturation ({dropped_blocks} total).",
                    metrics={"dropped_blocks": dropped_blocks},
                )

    def evaluate_scheduler_deadlines(self, consecutive_misses: int, total_misses: int):
        """Evaluates scheduler deadline misses."""
        with self._lock:
            if consecutive_misses >= self.config.max_consecutive_deadline_misses:
                self.report_fault(
                    fault_type=FaultType.REPEATED_DEADLINE_MISSES,
                    component="scheduler",
                    description=f"{consecutive_misses} consecutive real-time deadline misses detected.",
                    metrics={"consecutive_misses": consecutive_misses, "total_misses": total_misses},
                )

    def evaluate_inference_latency(self, latency_ms: float):
        """Evaluates ML model inference execution time."""
        with self._lock:
            if latency_ms >= self.config.max_inference_latency_ms:
                self.report_fault(
                    fault_type=FaultType.EXCESSIVE_INFERENCE_LATENCY,
                    component="inference",
                    description=f"Inference latency ({latency_ms:.1f}ms) exceeded threshold ({self.config.max_inference_latency_ms:.1f}ms).",
                    metrics={"latency_ms": latency_ms, "threshold_ms": self.config.max_inference_latency_ms},
                )

    # ==========================================================================
    # Health Calculation
    # ==========================================================================

    def _update_component_status_on_fault(self, component: str, severity: FaultSeverity):
        if severity == FaultSeverity.CRITICAL or severity == FaultSeverity.ERROR:
            self._component_health[component] = ComponentHealth.FAULT
        elif severity == FaultSeverity.WARNING:
            if self._component_health.get(component) != ComponentHealth.FAULT:
                self._component_health[component] = ComponentHealth.DEGRADED

    def _recalculate_health(self):
        """
        Derives system-level health:
        - CRITICAL active fault or FAULT component -> FAULT
        - ERROR active fault -> FAULT
        - Active recovery in flight -> RECOVERING
        - WARNING active fault or DEGRADED/UNAVAILABLE component -> DEGRADED
        - All clear -> HEALTHY
        """
        if any(f.severity == FaultSeverity.CRITICAL for f in self._active_faults.values()) or any(s == ComponentHealth.FAULT for s in self._component_health.values()):
            self._overall_health = SystemHealthState.FAULT
        elif any(f.severity == FaultSeverity.ERROR for f in self._active_faults.values()):
            self._overall_health = SystemHealthState.FAULT
        elif self._is_recovering or any(s == ComponentHealth.RECOVERING for s in self._component_health.values()):
            self._overall_health = SystemHealthState.RECOVERING
        elif any(f.severity == FaultSeverity.WARNING for f in self._active_faults.values()) or any(s in (ComponentHealth.DEGRADED, ComponentHealth.UNAVAILABLE) for s in self._component_health.values()):
            self._overall_health = SystemHealthState.DEGRADED
        else:
            self._overall_health = SystemHealthState.HEALTHY

    def _infer_category(self, fault_type: FaultType) -> FaultCategory:
        val = fault_type.value
        if val.startswith("MIC_") or "AUDIO" in val:
            return FaultCategory.AUDIO
        elif "INFERENCE" in val or "FEATURE" in val or "DEADLINE" in val or "WORKER" in val:
            return FaultCategory.PROCESSING
        elif "AUTH" in val or "LOCK" in val or "ADMIN" in val or "COMMAND" in val:
            return FaultCategory.SECURITY
        return FaultCategory.APPLICATION

    def _log_audit_fault(self, event: FaultEvent):
        if self.audit_logger is not None:
            try:
                res = AuditResult.WARNING if event.severity in (FaultSeverity.WARNING, FaultSeverity.INFO) else AuditResult.FAILURE
                self.audit_logger.log(
                    event_type=AuditEventType.FAULT_DETECTED,
                    result=res,
                    reason=f"[{event.severity.value}] {event.component.upper()}: {event.description}",
                    lock_state=self._get_lock_state(),
                    source=AuditSource.SYSTEM,
                    metadata=event.to_dict(),
                )
            except Exception as e:
                logger.error(f"[FAULT DETECTOR] Failed to dispatch fault audit log: {e}")

    # ==========================================================================
    # Inspection & Telemetry
    # ==========================================================================

    @property
    def system_health(self) -> SystemHealthState:
        with self._lock:
            return self._overall_health

    def get_component_health(self, component: str) -> ComponentHealth:
        with self._lock:
            return self._component_health.get(component, ComponentHealth.OPERATIONAL)

    def set_component_health(self, component: str, status: ComponentHealth):
        with self._lock:
            self._component_health[component] = status
            self._recalculate_health()

    def get_health_snapshot(self) -> Dict[str, Any]:
        """Returns comprehensive structured telemetry for dashboard and diagnostics."""
        with self._lock:
            return {
                "system_health": self._overall_health.value,
                "components": {k: v.value for k, v in self._component_health.items()},
                "active_faults": [f.to_dict() for f in self._active_faults.values()],
                "active_faults_count": len(self._active_faults),
                "total_faults_detected": self._fault_counter,
                "recent_faults": [f.to_dict() for f in self._fault_history[-10:]],
                "is_recovering": self._is_recovering,
            }
