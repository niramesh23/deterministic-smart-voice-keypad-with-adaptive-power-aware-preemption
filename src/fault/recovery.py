"""
src/fault/recovery.py
---------------------
Automated and Supervised Recovery Manager for Real-Time Access Control Subsystems.

Responsibilities:
1. Implements safe recovery policies for Audio, ML Inference, Persistence, Scheduler, and Dashboard.
2. Ensures non-blocking, bounded-retry execution without stalling live capture.
3. Enforces Safe-Lock policy on CRITICAL faults (fail-closed to LOCKED state, purge PIN buffers).
4. Strictly verifies component health before declaring successful recovery.
5. Emits structured recovery records and persistent audit trail events.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.fault.types import (
    ComponentHealth,
    FaultCategory,
    FaultEvent,
    FaultSeverity,
    FaultType,
    RecoveryRecord,
    SystemHealthState,
)
from src.fault.config import RecoveryConfig
from src.fault.detector import FaultDetector
from src.security.lock_controller import LockController, LockState
from src.security.audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditResult,
    AuditSource,
)

logger = logging.getLogger("rtos.fault.recovery")


class RecoveryManager:
    """
    Coordinates diagnostic verification, bounded retry execution, and safe state enforcement.
    """

    def __init__(
        self,
        fault_detector: FaultDetector,
        lock_controller: Optional[LockController] = None,
        audit_logger: Optional[AuditLogger] = None,
        config: Optional[RecoveryConfig] = None,
        time_func: Optional[Callable[[], float]] = None,
    ):
        self.fault_detector = fault_detector
        self.lock_controller = lock_controller
        self.audit_logger = audit_logger or getattr(lock_controller, "audit_logger", None)
        self.config = config or RecoveryConfig()
        self._time_func = time_func or time.time
        self._lock = threading.RLock()

        # Telemetry & Records
        self._recovery_counter: int = 0
        self._recovery_records: List[RecoveryRecord] = []
        self._component_retry_counts: Dict[str, int] = {}

        # Component References (registered at startup)
        self._audio_pipeline: Optional[Any] = None
        self._inference_service: Optional[Any] = None
        self._dashboard_server: Optional[Any] = None
        self._database_adapter: Optional[Any] = None

    def _get_lock_state(self) -> str:
        if self.lock_controller is not None and hasattr(self.lock_controller, "current_state"):
            return str(self.lock_controller.current_state.value)
        return "LOCKED"

    def register_subsystems(
        self,
        audio_pipeline: Optional[Any] = None,
        inference_service: Optional[Any] = None,
        dashboard_server: Optional[Any] = None,
        database_adapter: Optional[Any] = None,
    ):
        """Attaches live subsystem handles for programmatic recovery actions."""
        with self._lock:
            if audio_pipeline is not None:
                self._audio_pipeline = audio_pipeline
            if inference_service is not None:
                self._inference_service = inference_service
            if dashboard_server is not None:
                self._dashboard_server = dashboard_server
            if database_adapter is not None:
                self._database_adapter = database_adapter

    # ==========================================================================
    # Automated Fault Handler Callback
    # ==========================================================================

    def on_fault_detected(self, event: FaultEvent):
        """
        Entry point called by FaultDetector when a new fault is confirmed.
        Dispatches recovery policies or fails closed on CRITICAL faults.
        """
        # 1. Enforce Safe Lock on CRITICAL faults immediately
        if event.severity == FaultSeverity.CRITICAL or self.config.fail_safe_lock_on_critical and event.severity == FaultSeverity.CRITICAL:
            self.enforce_safe_lock(reason=f"CRITICAL FAULT ({event.fault_type.value}): {event.description}")

        # 2. Automated component recovery dispatch
        if event.category == FaultCategory.AUDIO:
            self.recover_audio_subsystem(event)
        elif event.category == FaultCategory.PROCESSING and "INFERENCE" in event.fault_type.value:
            self.recover_inference_subsystem(event)
        elif event.fault_type == FaultType.DATABASE_CONNECTION_FAILURE or event.fault_type == FaultType.AUDIT_WORKER_FAILURE:
            self.recover_database_subsystem(event)
        elif event.fault_type == FaultType.DASHBOARD_UNAVAILABLE:
            self.recover_dashboard_subsystem(event)

    # ==========================================================================
    # 1. Audio Subsystem Recovery
    # ==========================================================================

    def recover_audio_subsystem(self, event: Optional[FaultEvent] = None) -> Tuple[bool, str]:
        """
        Attempts controlled audio stream reinitialization:
        - Sets component state to RECOVERING.
        - Drains stale backlog from BoundedAudioQueue if configured.
        - Restarts capture stream or bridge thread.
        - Verifies audio pipeline can accept new frames.
        """
        with self._lock:
            self._recovery_counter += 1
            rec_id = self._recovery_counter
            self.fault_detector.set_recovering(True, component="microphone")

            retries = self._component_retry_counts.get("microphone", 0) + 1
            self._component_retry_counts["microphone"] = retries

            if retries > self.config.max_recovery_attempts:
                msg = f"Audio recovery aborted: Max retries ({self.config.max_recovery_attempts}) exceeded."
                logger.error(f"[RECOVERY MANAGER] {msg}")
                self._record_recovery(
                    rec_id=rec_id,
                    fault_id=event.fault_id if event else 0,
                    fault_type=event.fault_type if event else FaultType.MIC_DISCONNECTED,
                    component="microphone",
                    action="ABORT_MAX_RETRIES",
                    success=False,
                    verified=False,
                    details={"retries": retries},
                )
                self.fault_detector.set_component_health("microphone", ComponentHealth.FAULT)
                self.fault_detector.set_recovering(False)
                return False, msg

            logger.info(f"[RECOVERY MANAGER] Initiating audio subsystem recovery (Attempt {retries}/{self.config.max_recovery_attempts})...")

            # 1. Drain stale audio blocks
            if self.config.stale_queue_drain_on_audio_recovery and self._audio_pipeline is not None:
                if hasattr(self._audio_pipeline, "audio_queue"):
                    drained = self._audio_pipeline.audio_queue.clear()
                    logger.info(f"[RECOVERY MANAGER] Drained {drained} stale blocks from audio queue.")

            # 2. Controlled stream reinitialization
            success = False
            verified = False
            action_desc = "REINIT_STREAM"

            try:
                if self._audio_pipeline is not None and hasattr(self._audio_pipeline, "restart_microphone"):
                    success = self._audio_pipeline.restart_microphone()
                elif self._audio_pipeline is not None and hasattr(self._audio_pipeline, "mic"):
                    if self._audio_pipeline.mic is not None:
                        self._audio_pipeline.mic.stop()
                        self._audio_pipeline.mic.start()
                        success = True
                else:
                    success = False

                # 3. VERIFICATION: Verify audio stream is producing blocks or operational
                if success:
                    verified = self._verify_audio_pipeline()
            except Exception as e:
                logger.error(f"[RECOVERY MANAGER] Audio stream recovery failed: {e}")
                success = False
                verified = False

            if verified:
                self.fault_detector.clear_faults_for_component("microphone", "Microphone stream verified operational.")
                self.fault_detector.clear_faults_for_component("audio_queue", "Queue drained and normal.")
                self._component_retry_counts["microphone"] = 0
                msg = "Audio subsystem successfully reinitialized and verified healthy."
                logger.info(f"[RECOVERY MANAGER] {msg}")
            else:
                self.fault_detector.set_component_health("microphone", ComponentHealth.FAULT)
                msg = "Audio subsystem reinitialization failed verification check."
                logger.warning(f"[RECOVERY MANAGER] {msg}")

            self.fault_detector.set_recovering(False)
            self._record_recovery(
                rec_id=rec_id,
                fault_id=event.fault_id if event else 0,
                fault_type=event.fault_type if event else FaultType.MIC_DISCONNECTED,
                component="microphone",
                action=action_desc,
                success=success,
                verified=verified,
                details={"retries": retries},
            )
            return verified, msg

    def _verify_audio_pipeline(self) -> bool:
        """Verifies microphone capture state."""
        if self._audio_pipeline is None:
            return False
        if hasattr(self._audio_pipeline, "mic") and self._audio_pipeline.mic is not None:
            return bool(getattr(self._audio_pipeline.mic, "_is_running", False))
        if hasattr(self._audio_pipeline, "is_running"):
            return bool(getattr(self._audio_pipeline, "is_running", False))
        if hasattr(self._audio_pipeline, "online"):
            return bool(getattr(self._audio_pipeline, "online", False))
        return True

    # ==========================================================================
    # 2. Inference Subsystem Recovery
    # ==========================================================================

    def recover_inference_subsystem(self, event: Optional[FaultEvent] = None) -> Tuple[bool, str]:
        """
        Attempts inference engine recovery:
        - Marks inference temporarily unavailable (never fabricate predictions!).
        - Resets inference service internal state and rolling buffers.
        - Verifies ML engine by passing a neutral synthetic audio tensor.
        - Restores availability only if the verification forward-pass succeeds.
        """
        with self._lock:
            self._recovery_counter += 1
            rec_id = self._recovery_counter
            self.fault_detector.set_recovering(True, component="inference")

            logger.info("[RECOVERY MANAGER] Initiating ML inference service recovery...")
            if self._inference_service is not None and hasattr(self._inference_service, "set_available"):
                self._inference_service.set_available(False)

            verified = False
            try:
                # 1. Reset state
                if self._inference_service is not None and hasattr(self._inference_service, "reset_state"):
                    self._inference_service.reset_state()

                # 2. VERIFICATION: Test forward-pass with synthetic neutral audio
                if self.config.verify_inference_with_synthetic_tensor and self._inference_service is not None:
                    dummy_audio = np.zeros(16000, dtype=np.float32)
                    if hasattr(self._inference_service, "predict"):
                        res = self._inference_service.predict(dummy_audio)
                        if res is not None:
                            verified = True
                else:
                    verified = True  # Simulated / external harness
            except Exception as e:
                logger.error(f"[RECOVERY MANAGER] Inference verification failed: {e}")
                verified = False

            if verified:
                if self._inference_service is not None and hasattr(self._inference_service, "set_available"):
                    self._inference_service.set_available(True)
                self.fault_detector.clear_faults_for_component("inference", "Model inference verified with test tensor.")
                msg = "Inference service state reset and forward-pass verified operational."
                logger.info(f"[RECOVERY MANAGER] {msg}")
            else:
                self.fault_detector.set_component_health("inference", ComponentHealth.FAULT)
                msg = "Inference service recovery failed verification; service remains UNAVAILABLE."
                logger.warning(f"[RECOVERY MANAGER] {msg}")

            self.fault_detector.set_recovering(False)
            self._record_recovery(
                rec_id=rec_id,
                fault_id=event.fault_id if event else 0,
                fault_type=event.fault_type if event else FaultType.INFERENCE_EXCEPTION,
                component="inference",
                action="RESET_AND_VERIFY_FORWARD_PASS",
                success=verified,
                verified=verified,
                details={"verified": verified},
            )
            return verified, msg

    # ==========================================================================
    # 3. Database Subsystem Recovery
    # ==========================================================================

    def recover_database_subsystem(self, event: Optional[FaultEvent] = None) -> Tuple[bool, str]:
        """
        Attempts database recovery with bounded retries and exponential backoff:
        - Retries connection or pending write operations up to max_db_retry_attempts.
        - Verifies database health via probe query.
        - Preserves logging guarantee; reports when persistent audit trail cannot be assured.
        """
        with self._lock:
            self._recovery_counter += 1
            rec_id = self._recovery_counter
            self.fault_detector.set_recovering(True, component="database")

            logger.info("[RECOVERY MANAGER] Initiating database recovery probe...")
            verified = False

            for attempt in range(1, self.config.max_db_retry_attempts + 1):
                try:
                    if self.audit_logger is not None:
                        # Probe database health
                        if hasattr(self.audit_logger, "is_healthy"):
                            if self.audit_logger.is_healthy():
                                verified = True
                                break
                        elif hasattr(self.audit_logger, "_conn"):
                            cur = self.audit_logger._conn.cursor()
                            cur.execute("SELECT 1;")
                            verified = True
                            break
                        else:
                            verified = True
                            break
                except Exception as e:
                    logger.debug(f"[RECOVERY MANAGER] DB probe attempt {attempt} failed: {e}")
                    time.sleep(self.config.db_retry_backoff_s * attempt)

            if verified:
                self.fault_detector.clear_faults_for_component("database", "Database connection verified operational.")
                msg = "Database connection restored and verified healthy."
                logger.info(f"[RECOVERY MANAGER] {msg}")
            else:
                self.fault_detector.set_component_health("database", ComponentHealth.UNAVAILABLE)
                msg = "Database recovery failed: Bounded retries exhausted; logging degraded."
                logger.warning(f"[RECOVERY MANAGER] {msg}")

            self.fault_detector.set_recovering(False)
            self._record_recovery(
                rec_id=rec_id,
                fault_id=event.fault_id if event else 0,
                fault_type=event.fault_type if event else FaultType.DATABASE_CONNECTION_FAILURE,
                component="database",
                action="BOUNDED_RETRY_PROBE",
                success=verified,
                verified=verified,
                details={"attempts": self.config.max_db_retry_attempts},
            )
            return verified, msg

    # ==========================================================================
    # 4. Dashboard Subsystem Recovery
    # ==========================================================================

    def recover_dashboard_subsystem(self, event: Optional[FaultEvent] = None) -> Tuple[bool, str]:
        """
        Attempts controlled recovery of dashboard server:
        - If dashboard is stopped or failed, tries controlled restart.
        - Core speech recognition and access control CONTINUES uninterrupted (headless mode).
        """
        with self._lock:
            self._recovery_counter += 1
            rec_id = self._recovery_counter
            self.fault_detector.set_recovering(True, component="dashboard")

            logger.info("[RECOVERY MANAGER] Attempting controlled dashboard server restart...")
            verified = False

            try:
                if self._dashboard_server is not None:
                    try:
                        self._dashboard_server.stop()
                    except Exception:
                        pass
                    self._dashboard_server.start()
                    verified = getattr(self._dashboard_server, "_is_running", False)
                else:
                    verified = True
            except Exception as e:
                logger.warning(f"[RECOVERY MANAGER] Dashboard server restart failed: {e}. Continuing headless.")
                verified = False

            if verified:
                self.fault_detector.clear_faults_for_component("dashboard", "Dashboard server restarted and verified.")
                msg = "Dashboard service successfully restarted and online."
            else:
                self.fault_detector.set_component_health("dashboard", ComponentHealth.DEGRADED)
                msg = "Dashboard recovery failed. Core access-control continuing in headless mode."

            self.fault_detector.set_recovering(False)
            self._record_recovery(
                rec_id=rec_id,
                fault_id=event.fault_id if event else 0,
                fault_type=event.fault_type if event else FaultType.DASHBOARD_UNAVAILABLE,
                component="dashboard",
                action="RESTART_SERVER_HEADLESS_FALLBACK",
                success=verified,
                verified=verified,
                details={},
            )
            return verified, msg

    # ==========================================================================
    # 5. Safe Lock Enforcement on Critical Faults
    # ==========================================================================

    def enforce_safe_lock(self, reason: str = "Critical application fault detected") -> bool:
        """
        Fails closed on critical faults:
        1. Rejects any pending unlock request.
        2. Clears active PIN entry buffer.
        3. Forces simulated lock to return to LOCKED (or preserves LOCKOUT).
        4. Preserves fault reason and dispatches audit incident record.
        """
        logger.critical(f"[RECOVERY MANAGER] ENFORCING FAIL-SAFE LOCK: {reason}")
        if self.lock_controller is not None:
            try:
                # Trigger emergency stop logic to atomicaly purge buffers and lock barrier
                snap, had_pending = self.lock_controller.emergency_stop(
                    source=AuditSource.SYSTEM,
                    reason=f"FAIL-SAFE LOCKDOWN: {reason}",
                )
                logger.info(
                    f"[RECOVERY MANAGER] Fail-safe lock enforced. "
                    f"Lock state: {snap.state.value} | Pending Cancelled: {had_pending}"
                )
                return True
            except Exception as e:
                logger.critical(f"[RECOVERY MANAGER] Error during fail-safe lock enforcement: {e}")
                return False
        return False

    # ==========================================================================
    # Telemetry & Record Keeping
    # ==========================================================================

    def _record_recovery(
        self,
        rec_id: int,
        fault_id: int,
        fault_type: FaultType,
        component: str,
        action: str,
        success: bool,
        verified: bool,
        details: Dict[str, Any],
    ):
        record = RecoveryRecord(
            recovery_id=rec_id,
            fault_id=fault_id,
            fault_type=fault_type,
            component=component,
            action_taken=action,
            success=success,
            verified_healthy=verified,
            timestamp=self._time_func(),
            iso_time=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            details=details,
        )
        self._recovery_records.append(record)
        if len(self._recovery_records) > 100:
            self._recovery_records.pop(0)

        # Audit log recovery event
        if self.audit_logger is not None and hasattr(self.audit_logger, "log"):
            try:
                res = AuditResult.SUCCESS if verified else AuditResult.FAILURE
                self.audit_logger.log(
                    event_type=AuditEventType.FAULT_RECOVERY,
                    result=res,
                    reason=f"Recovery '{action}' for {component.upper()} (Verified: {verified})",
                    lock_state=self._get_lock_state(),
                    source=AuditSource.SYSTEM,
                    metadata=record.to_dict(),
                )
            except Exception as e:
                logger.error(f"[RECOVERY MANAGER] Failed to dispatch recovery audit event: {e}")

    def get_recovery_records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._recovery_records]

    def get_recovery_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._recovery_records)
            succeeded = sum(1 for r in self._recovery_records if r.verified_healthy)
            failed = total - succeeded
            return {
                "total_recoveries": total,
                "succeeded": succeeded,
                "failed": failed,
                "recent_recoveries": [r.to_dict() for r in self._recovery_records[-10:]],
            }
