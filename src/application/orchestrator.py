"""
src/application/orchestrator.py
-------------------------------
Application Orchestrator for Real-Time Voice Recognition.

Responsibilities:
- Coordinates the complete voice recognition lifecycle:
  MicrophoneCapture -> PingPongBuffer -> AudioQueue -> Framer -> VAD -> Rolling Context ->
  RealTimeScheduler -> AdaptiveController -> CommandFSM -> Dashboard -> StructuredLogger.
- Fault tolerance & integration handling:
  * Safe handling of missing model checkpoint.
  * Microphone device failure and disconnection recovery.
  * Audio queue overflow handling without backpressure on audio callback.
  * Dashboard interruption and graceful server shutdown.
  * Clean shutdown and restart capability.
- Non-blocking guarantee: Audio callbacks and real-time scheduling threads never block on dashboard or logging.
"""

from dataclasses import dataclass, field
import logging
import os
import pathlib
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.application.scheduled_recognizer import ScheduledKeywordRecognizer
from src.application.command_fsm import (
    CommandFiniteStateMachine,
    CommandFSMConfig,
    CommandRecord,
    CommandState,
    RejectionReason,
)
from src.ml.classes import NUMBER_TO_DIGIT, TARGET_COMMANDS
from src.security.config import SecurityConfig
from src.security.lock_controller import LockController, LockState, LockStateSnapshot, AccessResult
from src.security.emergency_handler import EmergencyHandler
from src.security.audit_logger import AuditEventType, AuditResult, AuditSource
from src.dashboard.server import DashboardServer, DashboardStatusProvider
from src.dashboard.snapshot import render_dashboard_snapshot
from src.monitoring.structured_logger import StructuredExperimentLogger
from src.audio.microphone import AudioDeviceError, get_default_input_device
from src.rts.scheduler import SchedulingPolicy, LateJobStrategy
from src.rts.adaptive import AdaptiveControllerConfig
from src.fault.types import (
    FaultSeverity,
    FaultCategory,
    SystemHealthState,
    ComponentHealth,
    FaultType,
    FaultEvent,
    RecoveryRecord,
)
from src.fault.config import FaultDetectorConfig, RecoveryConfig
from src.fault.detector import FaultDetector
from src.fault.recovery import RecoveryManager
from src.analytics import PerformanceAnalyticsCollector, WorkloadRunner

logger = logging.getLogger("rtos.orchestrator")


@dataclass
class OrchestratorConfig:
    """Consolidated configuration for the application orchestrator."""
    policy: SchedulingPolicy = SchedulingPolicy.EDF
    late_job_strategy: LateJobStrategy = LateJobStrategy.EXECUTE_AND_RECORD
    queue_capacity: int = 100
    enable_adaptation: bool = True
    enable_dashboard: bool = True
    dashboard_port: int = 8080
    enable_audio_logging: bool = False
    device_id: Optional[int] = None
    fsm_config: Optional[CommandFSMConfig] = None
    adaptive_config: Optional[AdaptiveControllerConfig] = None
    security_config: Optional[SecurityConfig] = None


class VoiceRecognitionApp:
    """
    Main Application Orchestrator for Real-Time Voice Recognition.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        on_command_callback: Optional[Callable[[CommandRecord], None]] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.user_on_command = on_command_callback

        # 1. Structured Experiment Logger
        self.logger_subsystem = StructuredExperimentLogger(
            output_dir="results/logs",
            enable_audio_recording=self.config.enable_audio_logging,
        )

        # 2. Command Finite-State Machine
        self.fsm = CommandFiniteStateMachine(
            config=self.config.fsm_config,
            on_command_callback=self._handle_fsm_command,
        )

        # 2b. Simulated Smart Lock Controller & Emergency Handler
        self.security_config = self.config.security_config or SecurityConfig()
        self.lock_controller = LockController(config=self.security_config)
        self.emergency_handler = EmergencyHandler(
            lock_controller=self.lock_controller,
            audit_logger=self.lock_controller.audit_logger,
        )

        # 2c. Fault Detection & Safe Recovery Subsystem
        self.fault_detector = FaultDetector(
            audit_logger=self.lock_controller.audit_logger,
        )
        self.fault_detector.set_lock_controller(self.lock_controller)
        self.recovery_manager = RecoveryManager(
            fault_detector=self.fault_detector,
            lock_controller=self.lock_controller,
            audit_logger=self.lock_controller.audit_logger,
        )
        self.fault_detector.on_fault_detected = self.recovery_manager.on_fault_detected

        # 3. Scheduled Pipeline Recognizer
        rec_conf = self.config.fsm_config.confirm_confidence_threshold if self.config.fsm_config else 0.65
        rec_margin = self.config.fsm_config.margin_threshold if self.config.fsm_config else 0.15
        self.recognizer = ScheduledKeywordRecognizer(
            policy=self.config.policy,
            late_job_strategy=self.config.late_job_strategy,
            queue_capacity=self.config.queue_capacity,
            device_id=self.config.device_id,
            enable_adaptation=self.config.enable_adaptation,
            adaptive_config=self.config.adaptive_config,
            confidence_threshold=rec_conf,
            margin_threshold=rec_margin,
        )
        self.recognizer.on_command_callback = self._handle_recognizer_prediction
        self.recognizer.on_vad_state_callback = self._on_vad_state_changed
        self.recognizer.scheduler.on_task_exception = self._on_scheduler_task_exception

        # 3b. Real-Time Performance Analytics Subsystem
        self.analytics_collector = PerformanceAnalyticsCollector()
        self.recognizer.analytics_collector = self.analytics_collector
        self.workload_runner = WorkloadRunner(collector=self.analytics_collector)

        # 3c. Access Analytics & Report Generation Subsystem
        from src.analytics.access_analytics import AccessAnalyticsManager
        self.access_analytics = AccessAnalyticsManager(
            db_path=self.lock_controller.audit_logger.db_path if self.lock_controller.audit_logger else "data/security_audit.db",
            user_manager=self.lock_controller.user_manager,
            audit_logger=self.lock_controller.audit_logger,
        )

        # 3d. Accessibility & Voice Feedback Subsystems
        from src.accessibility import AccessibilitySettingsManager, VoiceFeedback
        self.accessibility_manager = AccessibilitySettingsManager()
        self.voice_feedback = VoiceFeedback(settings=self.accessibility_manager.settings)
        self.lock_controller.set_voice_feedback(self.voice_feedback)
        self.fault_detector.set_voice_feedback(self.voice_feedback)
        self.lock_controller.set_pin_timeout(self.accessibility_manager.settings.pin_entry_timeout_s)
        self.accessibility_manager.add_listener(self._on_accessibility_settings_updated)

        # 4. Monitoring Dashboard
        self.status_provider = DashboardStatusProvider(
            audit_logger=self.lock_controller.audit_logger,
            user_manager=self.lock_controller.user_manager,
            emergency_handler=self.emergency_handler,
            fault_detector=self.fault_detector,
            recovery_manager=self.recovery_manager,
            analytics_collector=self.analytics_collector,
            workload_runner=self.workload_runner,
            access_analytics=self.access_analytics,
            accessibility_manager=self.accessibility_manager,
            voice_feedback=self.voice_feedback,
        )
        self.dashboard_server: Optional[DashboardServer] = None
        if self.config.enable_dashboard:
            self.dashboard_server = DashboardServer(
                port=self.config.dashboard_port,
                provider=self.status_provider,
            )

        # Connect live subsystems to RecoveryManager
        self.recovery_manager.register_subsystems(
            audio_pipeline=self.recognizer,
            inference_service=self.recognizer.inference_service,
            dashboard_server=self.dashboard_server,
        )

        # Internal lifecycle state
        self._is_running = False
        self._lock = threading.Lock()
        self._sync_thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
        self._command_counter: int = 0

    def start(self, use_microphone: bool = True):
        """
        Initializes and starts all application subsystems.
        Handles device/model faults cleanly.
        """
        with self._lock:
            if self._is_running:
                logger.warning("VoiceRecognitionApp is already running.")
                return

            logger.info("Starting VoiceRecognitionApp...")
            self._last_error = None

            # 1. Start Dashboard Server
            if self.dashboard_server is not None:
                try:
                    self.dashboard_server.start()
                except Exception as e:
                    logger.warning(f"Dashboard server failed to start: {e}. Continuing headless.")

            # 2. Start Scheduled Recognizer Engine
            try:
                self.recognizer.start(use_microphone=use_microphone)
            except Exception as e:
                self._last_error = str(e)
                self.logger_subsystem.log_audio_error("ENGINE_START_FAILURE", {"error": str(e)})
                logger.error(f"Failed to start audio recognition engine: {e}")
                self.stop()
                raise e

            self._is_running = True

            # Voice Feedback: Announce system ready
            if hasattr(self, "voice_feedback") and self.voice_feedback is not None:
                try:
                    self.voice_feedback.on_system_ready()
                except Exception as e:
                    logger.debug(f"Voice feedback start error: {e}")

            # Log SYSTEM_STARTUP
            if self.lock_controller.audit_logger is not None:
                self.lock_controller.audit_logger.log(
                    event_type=AuditEventType.SYSTEM_STARTUP,
                    result=AuditResult.INFO,
                    reason="Voice recognition application subsystems started.",
                    lock_state=self.lock_controller.state.value,
                    source=AuditSource.SYSTEM,
                    metadata={"policy": str(self.config.policy), "device_id": self.config.device_id},
                )

            # 3. Start Telemetry Sync Thread (updates dashboard status snapshot at 10 Hz)
            self._sync_thread = threading.Thread(
                target=self._telemetry_sync_loop,
                name="TelemetrySyncThread",
                daemon=True,
            )
            self._sync_thread.start()
            logger.info("VoiceRecognitionApp started successfully.")

    def stop(self):
        """
        Gracefully shuts down all pipeline components, audio callbacks, and servers.
        """
        with self._lock:
            if not self._is_running:
                return

            logger.info("Stopping VoiceRecognitionApp...")
            self._is_running = False

            # Stop pipeline engine
            try:
                self.recognizer.stop()
            except Exception as e:
                logger.warning(f"Error stopping recognizer: {e}")

            # Stop dashboard server
            if self.dashboard_server is not None:
                try:
                    self.dashboard_server.stop()
                except Exception as e:
                    logger.warning(f"Error stopping dashboard: {e}")

            # Close structured logger
            try:
                self.logger_subsystem.close()
            except Exception as e:
                logger.warning(f"Error closing structured logger: {e}")

            # Log SYSTEM_SHUTDOWN and cleanly flush/stop audit logger
            if self.lock_controller.audit_logger is not None:
                self.lock_controller.audit_logger.log(
                    event_type=AuditEventType.SYSTEM_SHUTDOWN,
                    result=AuditResult.INFO,
                    reason="Voice recognition application shutting down cleanly.",
                    lock_state=self.lock_controller.state.value,
                    source=AuditSource.SYSTEM,
                )
                self.lock_controller.stop()

            # Cleanly stop voice feedback background worker
            if hasattr(self, "voice_feedback") and self.voice_feedback is not None:
                try:
                    self.voice_feedback.stop()
                except Exception as e:
                    logger.debug(f"Error stopping voice feedback: {e}")

            logger.info("VoiceRecognitionApp stopped cleanly.")

    def _on_accessibility_settings_updated(self, settings):
        """Invoked when accessibility settings are changed via dashboard or API."""
        if hasattr(self, "voice_feedback") and self.voice_feedback is not None:
            self.voice_feedback.update_settings(settings)
        if hasattr(self, "lock_controller") and self.lock_controller is not None:
            self.lock_controller.set_pin_timeout(settings.pin_entry_timeout_s)

    def restart(self, use_microphone: bool = True):
        """Safely restarts the application from a clean state after error or interruption."""
        logger.info("Restarting VoiceRecognitionApp...")
        self.stop()
        time.sleep(0.1)
        self.emergency_handler.clear_emergency_flag()
        self.fault_detector.clear_faults_for_component("microphone", "Restart initiated")
        self.fault_detector.clear_faults_for_component("inference", "Restart initiated")
        self.fsm.reset()
        self.lock_controller.reset()
        self.start(use_microphone=use_microphone)

    def _on_scheduler_task_exception(self, job: Any, exc: Exception):
        """Callback invoked when a scheduled real-time job encounters an exception."""
        logger.error(f"[ORCHESTRATOR] Task execution exception in {getattr(job, 'job_id', 'job')}: {exc}")
        job_type_str = str(getattr(job, "job_type", "")).upper()
        if "INFERENCE" in job_type_str:
            self.fault_detector.report_fault(
                fault_type=FaultType.INFERENCE_EXCEPTION,
                component="inference",
                description=f"Inference execution exception in {getattr(job, 'job_id', 'unknown')}: {exc}",
                metrics={"job_id": getattr(job, "job_id", None), "error": str(exc)},
            )
        elif "FEATURE" in job_type_str:
            self.fault_detector.report_fault(
                fault_type=FaultType.FEATURE_EXTRACTION_FAILURE,
                component="inference",
                description=f"Feature extraction exception in {getattr(job, 'job_id', 'unknown')}: {exc}",
                metrics={"job_id": getattr(job, "job_id", None), "error": str(exc)},
            )
        else:
            self.fault_detector.report_fault(
                fault_type=FaultType.TASK_EXECUTION_EXCEPTION,
                component="scheduler",
                description=f"Task exception in {getattr(job, 'job_id', 'unknown')}: {exc}",
                metrics={"job_id": getattr(job, "job_id", None), "error": str(exc)},
            )

    def push_audio_hop(self, hop_pcm16: np.ndarray, capture_time: Optional[float] = None):
        """Feeds external audio hop (used for deterministic recorded audio evaluation)."""
        return self.recognizer.push_external_hop(hop_pcm16, capture_time=capture_time)

    def handle_voice_command(
        self,
        command: str,
        confidence: float = 0.95,
        margin: float = 0.40,
    ) -> LockStateSnapshot:
        """
        Public programmatic entrypoint to simulate or dispatch a confirmed voice command
        through the application pipeline (e.g. for deterministic integration testing,
        scripted workflows, or external audio events).
        """
        with self._lock:
            self._command_counter += 1
            cmd_id = self._command_counter
        now = time.perf_counter()
        record = CommandRecord(
            record_id=cmd_id,
            command=command,
            confidence=confidence,
            margin=margin,
            state_at_event=CommandState.COMMAND_CONFIRMED,
            is_confirmed=True,
            rejection_reason=RejectionReason.NONE,
            detected_timestamp=now,
            confirmed_timestamp=now,
            reaction_latency_ms=0.0,
            acoustic_latency_ms=0.0,
        )
        self._handle_fsm_command(record)
        return self.lock_controller.get_snapshot()

    # ==========================================================================
    # Internal Callbacks & Telemetry Coordination
    # ==========================================================================

    def _handle_recognizer_prediction(self, result, telemetry):
        """Invoked by ScheduledKeywordRecognizer when an inference prediction completes."""
        # Update FSM with VAD state and evaluate prediction
        self.fsm.notify_vad_state(self.recognizer.vad.state)
        record = self.fsm.process_prediction(result, telemetry, vad_state=self.recognizer.vad.state)

        # Record Stage 2 E2E: Recognition to FSM confirmation
        if self.analytics_collector is not None and result.decision_timestamp:
            t_conf = record.confirmed_timestamp or record.detected_timestamp or time.perf_counter()
            self.analytics_collector.record_e2e_recognition_to_confirmation(
                result.decision_timestamp,
                t_conf,
            )

        # Log prediction event in structured logger
        self.logger_subsystem.log_prediction_result(record.to_dict())

        # If an unconfirmed or sub-threshold STOP prediction occurred, audit rejection
        if record.command == "stop" and not record.is_confirmed:
            self.emergency_handler.handle_voice_stop(record)

    def _handle_fsm_command(self, record: CommandRecord):
        """Invoked when FSM confirms a valid, non-duplicate command."""
        logger.info(f"[APPLICATION] Executing command action for: '{record.command}'")

        # Track first/last digit timestamps for Stage 3 & 5 E2E
        if record.command in NUMBER_TO_DIGIT and self.analytics_collector is not None:
            curr_len = self.lock_controller.pin_manager.current_length
            t_conf = record.confirmed_timestamp or record.detected_timestamp or time.perf_counter()
            if curr_len == 0:
                self.analytics_collector.record_digit1_capture(t_conf)
            elif curr_len == self.lock_controller.config.pin_length - 1:
                self.analytics_collector.record_digit4_confirmation(time.perf_counter())

        t_conf = time.perf_counter()
        if record.command == "stop":
            # High-priority emergency stop handling
            accepted, lock_snap, decision = self.emergency_handler.handle_voice_stop(record)
            logger.info(
                f"[SMART LOCK] EMERGENCY STOP: {decision} | State: {lock_snap.state.value} "
                f"({lock_snap.last_access_message})"
            )
        else:
            # Deliver confirmed command to Smart Lock Controller
            lock_snap = self.lock_controller.handle_command(record.command, command_id=record.record_id)
            logger.info(
                f"[SMART LOCK] State: {lock_snap.state.value} | Result: {lock_snap.last_access_result.value} "
                f"({lock_snap.last_access_message})"
            )
        t_lock = time.perf_counter()

        # Record Stage 4 (confirmation to lock state transition)
        if self.analytics_collector is not None:
            self.analytics_collector.record_e2e_confirmation_to_lock(t_conf, t_lock)
            # If access decision occurred, record Stage 3 (PIN verification) & Stage 5 (Full E2E)
            if lock_snap.last_access_result in (AccessResult.GRANTED, AccessResult.DENIED_INCORRECT, AccessResult.DENIED_LOCKOUT):
                self.analytics_collector.record_e2e_pin_verification(t_lock)
                self.analytics_collector.record_e2e_command_to_access(t_lock)

        if self.user_on_command is not None:
            try:
                self.user_on_command(record)
            except Exception as e:
                logger.error(f"Error in user command handler: {e}")

    def _on_vad_state_changed(self, vad_state):
        """Immediately syncs acoustic VAD state to the Command FSM."""
        if hasattr(self, "fsm") and self.fsm is not None:
            self.fsm.notify_vad_state(vad_state)

    def _telemetry_sync_loop(self):
        """Background thread updating the DashboardStatusProvider snapshot."""
        while self._is_running:
            try:
                self.lock_controller.tick()
                if self.recognizer and self.recognizer.vad:
                    self.fsm.notify_vad_state(self.recognizer.vad.state)
                self._update_status_provider()
            except Exception as e:
                logger.debug(f"Telemetry sync error: {e}")
            time.sleep(0.10)

    def _update_status_provider(self):
        """Constructs unified status payload and pushes to DashboardStatusProvider."""
        report = self.recognizer.get_analysis_report()
        sched_stats = self.recognizer.scheduler.get_status_summary()
        q_metrics = self.recognizer.audio_queue.get_metrics()
        mic = self.recognizer.mic
        device_name = "Virtual / File Stream"
        device_id = None
        if mic is not None:
            dev_info = getattr(mic, "device_info", None)
            if dev_info is not None:
                device_name = getattr(dev_info, "name", str(dev_info))
                device_id = getattr(dev_info, "device_id", mic.device_id)
            else:
                device_name = getattr(mic, "actual_device_name", f"Device-{mic.device_id}")
                device_id = mic.device_id

        latest_rec = self.fsm.latest_confirmed
        latest_tel = self.recognizer.confirmed_commands_history[-1][1] if self.recognizer.confirmed_commands_history else None

        status_update = {
            "microphone": {
                "device_name": device_name,
                "device_id": device_id,
                "status": "CAPTURING" if (self.recognizer.mic and getattr(self.recognizer.mic, "is_active", self.recognizer.mic.is_running)) else "IDLE",
                "sample_rate": self.recognizer.sample_rate,
                "queue_occupancy": q_metrics.get("current_occupancy", 0),
                "queue_capacity": self.config.queue_capacity,
                "audio_blocks_dropped": self.recognizer.total_audio_blocks_dropped + q_metrics.get("dropped_blocks_count", 0),
            },
            "dsp": {
                "vad_state": self.recognizer.vad.state.name if self.recognizer.vad and hasattr(self.recognizer.vad.state, "name") else (self.recognizer.vad.state.value if self.recognizer.vad else "WAITING"),
                "noise_floor_rms": self.recognizer.vad.noise_floor.noise_floor if self.recognizer.vad else None,
            },
            "command_fsm": {
                "current_state": self.fsm.state.value,
                "latest_command": latest_rec.command if latest_rec else None,
                "latest_digit": NUMBER_TO_DIGIT.get(latest_rec.command) if (latest_rec and latest_rec.command) else None,
                "latest_confidence": latest_rec.confidence if latest_rec else None,
                "total_confirmed": self.fsm.total_confirmed,
                "total_rejected": self.fsm.total_rejected,
            },
            "timing": {
                "processing_reaction_latency_ms": latest_tel.reaction_latency_ms if latest_tel else None,
                "total_acoustic_latency_ms": latest_tel.total_acoustic_latency_ms if latest_tel else None,
                "model_inference_ms": latest_tel.model_inference_ms if latest_tel else None,
                "feature_extraction_ms": latest_tel.feature_extraction_ms if latest_tel else None,
                "queue_delay_ms": latest_tel.queue_delay_ms if latest_tel else None,
            },
            "scheduler": {
                "policy": sched_stats["policy"],
                "ready_jobs_count": sched_stats["ready_jobs_count"],
                "total_executed": sched_stats["total_executed"],
                "total_deadline_misses": sched_stats["total_deadline_misses"],
                "deadline_miss_ratio_pct": report.get("deadline_miss_ratio_pct", 0.0),
            },
            "adaptive_controller": {
                "current_state": self.recognizer.controller.state.value if self.recognizer.controller else "DISABLED",
                "total_state_transitions": self.recognizer.controller.total_state_transitions if self.recognizer.controller else 0,
                "total_stale_jobs_pruned": self.recognizer.controller.total_stale_jobs_pruned if self.recognizer.controller else 0,
                "total_inferences_suppressed": self.recognizer.controller.total_inferences_suppressed if self.recognizer.controller else 0,
                "mean_recent_queue_delay_ms": self.recognizer.controller.mean_recent_queue_delay_ms if self.recognizer.controller else 0.0,
            },
            "lock": self.lock_controller.get_snapshot().to_dict(),
            "emergency": self.emergency_handler.get_emergency_status(),
            "history": [r.to_dict() for r in self.fsm.history[-10:]],
        }

        # Fault Detector Health Evaluations
        if self.recognizer.audio_queue is not None:
            self.fault_detector.evaluate_audio_queue(
                occupancy=q_metrics.get("current_occupancy", 0),
                capacity=self.recognizer.audio_queue.capacity,
                dropped_blocks=self.recognizer.total_audio_blocks_dropped,
            )

        if sched_stats:
            self.fault_detector.evaluate_scheduler_deadlines(
                consecutive_misses=getattr(self.recognizer.scheduler, "consecutive_deadline_misses", 0),
                total_misses=sched_stats.get("total_deadline_misses", 0),
            )

        if latest_tel and latest_tel.model_inference_ms is not None:
            self.fault_detector.evaluate_inference_latency(latest_tel.model_inference_ms)

        # Include health status in payload
        health_snap = self.fault_detector.get_health_snapshot()
        recovery_stats = self.recovery_manager.get_recovery_stats()
        status_update["health"] = {
            **health_snap,
            **recovery_stats,
        }

        # Include real-time performance analytics telemetry
        if self.analytics_collector is not None:
            status_update["analytics"] = self.analytics_collector.get_live_telemetry(
                audio_queue=self.recognizer.audio_queue,
                scheduler=self.recognizer.scheduler,
                adaptive_controller=self.recognizer.controller,
                dropped_audio_blocks=self.recognizer.total_audio_blocks_dropped,
            )

        self.status_provider.update_status(status_update)

    def capture_dashboard_screenshot(self, output_path: str = "results/dashboard_screenshot.png") -> pathlib.Path:
        """Renders visual snapshot of the current dashboard state to PNG."""
        snapshot = self.status_provider.get_snapshot()
        return render_dashboard_snapshot(snapshot, output_path=output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Adaptive Voice Recognition & RTOS Monitor")
    parser.add_argument("--policy", choices=["FIFO", "RM", "EDF"], default="EDF", help="Scheduling policy (default: EDF)")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port (default: 8080)")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard server")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--no-adaptation", action="store_true", help="Disable adaptive processing controller")
    parser.add_argument("--no-mic", action="store_true", help="Do not open live physical microphone")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    policy_map = {
        "FIFO": SchedulingPolicy.FIFO,
        "RM": SchedulingPolicy.RM,
        "EDF": SchedulingPolicy.EDF,
    }

    fsm_cfg = CommandFSMConfig(
        candidate_confidence_threshold=0.28,
        confirm_confidence_threshold=0.35,
        margin_threshold=0.04,
        cooldown_duration_s=0.65,
        digit_cooldown_duration_s=0.30,
        speech_timeout_s=4.0,
    )
    sec_cfg = SecurityConfig(
        pin_timeout_s=60.0,
        unlock_duration_s=15.0,
    )
    adapt_cfg = AdaptiveControllerConfig(
        overload_consecutive_misses=8,
        overload_queue_high_threshold=8,
        overload_queue_delay_ms=35.0,
    )

    config = OrchestratorConfig(
        policy=policy_map[args.policy],
        enable_adaptation=not args.no_adaptation,
        enable_dashboard=not args.no_dashboard,
        dashboard_port=args.port,
        device_id=args.device,
        fsm_config=fsm_cfg,
        security_config=sec_cfg,
        adaptive_config=adapt_cfg,
    )

    def on_command_confirmed(rec: CommandRecord):
        disp = f"'{rec.command.upper()}'"
        if rec.command and rec.command in NUMBER_TO_DIGIT:
            disp += f" ({NUMBER_TO_DIGIT[rec.command]})"
        print(f"\n=======================================================")
        print(f" [COMMAND CONFIRMED] {disp} ")
        print(f" Confidence: {rec.confidence * 100:.1f}% | Margin: {rec.margin:.2f}")
        lat_info = f" Reaction Latency: {rec.reaction_latency_ms:.2f} ms"
        if rec.acoustic_latency_ms is not None:
            lat_info += f" | Acoustic Latency: {rec.acoustic_latency_ms:.1f} ms"
        print(lat_info)

        # Smart Lock Feedback
        lock_snap = app.lock_controller.get_snapshot()
        print(f" -------------------------------------------------------")
        print(f" [SMART LOCK] State: {lock_snap.state.value} | PIN: {lock_snap.pin_progress} (Masked: '{lock_snap.masked_entry}')")
        print(f" Status: {lock_snap.operational_status}")
        print(f" Message: {lock_snap.last_access_message}")
        print(f"=======================================================\n")

    app = VoiceRecognitionApp(config=config, on_command_callback=on_command_confirmed)

    use_mic = not args.no_mic
    print("==================================================================")
    print("  REAL-TIME ADAPTIVE VOICE RECOGNITION SYSTEM")
    print("==================================================================")
    print(f"  Scheduling Policy:     {args.policy}")
    print(f"  Adaptive Controller:   {'Enabled' if not args.no_adaptation else 'Disabled'}")
    print(f"  Target Vocabulary:     10 Core Commands + 9 Numbers (one through nine) [21 classes total]")
    if not args.no_dashboard:
        print(f"  Web Dashboard:         http://127.0.0.1:{args.port}/")
    print(f"  Microphone Ingress:    {'Active' if use_mic else 'Disabled'}")
    print("------------------------------------------------------------------")
    print("  Listening for spoken commands... Press Ctrl+C to terminate.")
    print("==================================================================\n")

    app.start(use_microphone=use_mic)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Terminating VoiceRecognitionApp cleanly...")
    finally:
        app.stop()
        print("[SHUTDOWN] System stopped.")


if __name__ == "__main__":
    main()


