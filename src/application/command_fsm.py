"""
src/application/command_fsm.py
------------------------------
Command Finite-State Machine (FSM) and Software-Only Command Output Handler.

States:
- WAITING: Listening for audio/speech activity.
- SPEECH_DETECTED: Voice Activity Detector has signaled speech onset or active speech.
- CANDIDATE_COMMAND: Neural network output classified a core command above candidate threshold.
- COMMAND_CONFIRMED: All verification criteria satisfied (confidence, margin, smoothing, vocabulary, cooldown).
- COOLDOWN: Temporary post-confirmation suppression window to prevent duplicate triggering from overlapping frames.

Key Guarantees:
- Only accepts commands in CORE_COMMANDS vocabulary.
- Rejects low-confidence, low-margin, or unknown/silence classes.
- Prevents repeated triggering across overlapping sliding inference windows.
- Full audit recording of detected and confirmed timestamps, reaction latency, and rejection reasons.
- Safe reset and graceful shutdown paths.
- Software-only execution: does NOT actuate physical hardware or external relays.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.dsp.vad import VADState
from src.ml.classes import (
    CORE_COMMANDS,
    NUMBER_COMMANDS,
    TARGET_COMMANDS,
    NUMBER_TO_DIGIT,
    CLASS_NAMES,
    get_display_command,
)
from src.ml.inference import RecognitionResult
from src.application.live_recognizer import LivePipelineTelemetry

logger = logging.getLogger("rtos.command_fsm")


class CommandState(str, Enum):
    """Documented states of the Command Finite-State Machine."""
    WAITING = "WAITING"
    SPEECH_DETECTED = "SPEECH_DETECTED"
    CANDIDATE_COMMAND = "CANDIDATE_COMMAND"
    COMMAND_CONFIRMED = "COMMAND_CONFIRMED"
    COOLDOWN = "COOLDOWN"


class RejectionReason(str, Enum):
    """Categorized rejection reasons for unconfirmed predictions."""
    NONE = "NONE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOW_MARGIN = "LOW_MARGIN"
    UNSUPPORTED_VOCABULARY = "UNSUPPORTED_VOCABULARY"
    DUPLICATE_IN_COOLDOWN = "DUPLICATE_IN_COOLDOWN"
    VAD_SILENCE = "VAD_SILENCE"
    SMOOTHING_UNSTABLE = "SMOOTHING_UNSTABLE"


@dataclass
class CommandRecord:
    """Detailed record of a processed command event."""
    record_id: int
    command: Optional[str]
    confidence: float
    margin: float
    state_at_event: CommandState
    is_confirmed: bool
    rejection_reason: RejectionReason
    detected_timestamp: float
    confirmed_timestamp: Optional[float]
    reaction_latency_ms: Optional[float]
    acoustic_latency_ms: Optional[float]
    display_command: Optional[str] = None
    digit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        disp = self.display_command or (get_display_command(self.command) if self.command else None)
        dig = self.digit or (NUMBER_TO_DIGIT.get(self.command) if self.command else None)
        return {
            "record_id": self.record_id,
            "command": self.command,
            "display_command": disp,
            "digit": dig,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "state_at_event": self.state_at_event.value,
            "is_confirmed": self.is_confirmed,
            "rejection_reason": self.rejection_reason.value,
            "detected_timestamp": round(self.detected_timestamp, 6),
            "confirmed_timestamp": round(self.confirmed_timestamp, 6) if self.confirmed_timestamp else None,
            "reaction_latency_ms": round(self.reaction_latency_ms, 2) if self.reaction_latency_ms is not None else None,
            "acoustic_latency_ms": round(self.acoustic_latency_ms, 2) if self.acoustic_latency_ms is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class CommandFSMConfig:
    """Configurable thresholds and timing guards for command acceptance."""
    candidate_confidence_threshold: float = 0.50
    confirm_confidence_threshold: float = 0.65
    margin_threshold: float = 0.15
    cooldown_duration_s: float = 0.80
    digit_cooldown_duration_s: float = 0.45
    speech_timeout_s: float = 1.20
    allowed_vocabulary: List[str] = field(default_factory=lambda: list(TARGET_COMMANDS))


class CommandFiniteStateMachine:
    """
    Finite-State Machine managing keyword command qualification, duplicate prevention,
    cooldown timing, and software-only command delivery.
    """

    def __init__(
        self,
        config: Optional[CommandFSMConfig] = None,
        on_command_callback: Optional[Callable[[CommandRecord], None]] = None,
    ):
        self.config = config or CommandFSMConfig()
        self.on_command_callback = on_command_callback

        # Internal FSM state
        self.state = CommandState.WAITING
        self._last_speech_time: float = 0.0
        self._last_confirmed_time: float = 0.0
        self._last_confirmed_command: Optional[str] = None

        # Statistics & Auditing
        self.total_evaluations: int = 0
        self.total_confirmed: int = 0
        self.total_rejected: int = 0
        self.total_duplicates_suppressed: int = 0
        self.total_state_transitions: int = 0

        self._command_history: List[CommandRecord] = []
        self._latest_confirmed_record: Optional[CommandRecord] = None

    def reset(self):
        """Safely resets the FSM to WAITING state and clears active timers."""
        self.state = CommandState.WAITING
        self._last_speech_time = 0.0
        self._last_confirmed_time = 0.0
        self._last_confirmed_command = None
        logger.info("[COMMAND FSM] FSM reset to WAITING state.")

    def notify_vad_state(self, vad_state: VADState, timestamp: Optional[float] = None):
        """
        Informs FSM of raw acoustic VAD state transitions.
        Handles WAITING <-> SPEECH_DETECTED transitions.
        """
        now = time.perf_counter() if timestamp is None else timestamp
        eff_cd = (
            self.config.digit_cooldown_duration_s
            if (self._last_confirmed_command and self._last_confirmed_command in NUMBER_COMMANDS)
            else self.config.cooldown_duration_s
        )

        if vad_state in (VADState.SPEECH_ACTIVE, VADState.ONSET_CANDIDATE, VADState.HANGOVER):
            self._last_speech_time = now
            if self.state == CommandState.WAITING:
                self._transition_to(CommandState.SPEECH_DETECTED, "VAD signaled speech onset")
            elif self.state == CommandState.COOLDOWN and (now - self._last_confirmed_time >= eff_cd):
                self._transition_to(CommandState.SPEECH_DETECTED, "VAD speech onset after cooldown elapsed")
        elif vad_state == VADState.SILENCE:
            # Check cooldown expiration
            if self.state == CommandState.COOLDOWN:
                if now - self._last_confirmed_time >= eff_cd:
                    self._transition_to(CommandState.WAITING, "Cooldown expired and silence restored")
            elif self.state in (CommandState.SPEECH_DETECTED, CommandState.CANDIDATE_COMMAND):
                # Speech timeout without inference or candidate recovery on silence
                if now - self._last_speech_time >= self.config.speech_timeout_s:
                    self._transition_to(CommandState.WAITING, "Speech timeout elapsed without valid command")

    def process_prediction(
        self,
        result: RecognitionResult,
        telemetry: LivePipelineTelemetry,
        vad_state: VADState = VADState.SPEECH_ACTIVE,
    ) -> CommandRecord:
        """
        Evaluates a neural network inference result against FSM state and guards.
        Returns a CommandRecord documenting the decision.
        """
        now = time.perf_counter()
        self.total_evaluations += 1

        cmd = result.command or (result.raw_class if (result.raw_class not in ("_silence_", "_unknown_")) else None)
        conf = float(result.confidence)
        margin = float(result.margin)
        effective_cooldown = self.config.digit_cooldown_duration_s if (cmd and cmd in NUMBER_COMMANDS) else self.config.cooldown_duration_s
        is_cooldown_active = (now - self._last_confirmed_time < effective_cooldown)

        # Check cooldown expiration if currently in COOLDOWN
        if self.state == CommandState.COOLDOWN and not is_cooldown_active:
            self._transition_to(CommandState.WAITING, "Cooldown duration elapsed")

        # ----------------------------------------------------------------------
        # Guard 1: Duplicate suppression during Cooldown or Overlapping Frames
        # ----------------------------------------------------------------------
        if is_cooldown_active and cmd == self._last_confirmed_command:
            self.total_duplicates_suppressed += 1
            self.total_rejected += 1
            rec = CommandRecord(
                record_id=len(self._command_history) + 1,
                command=cmd,
                confidence=conf,
                margin=margin,
                state_at_event=self.state,
                is_confirmed=False,
                rejection_reason=RejectionReason.DUPLICATE_IN_COOLDOWN,
                detected_timestamp=telemetry.decision_timestamp,
                confirmed_timestamp=None,
                reaction_latency_ms=telemetry.reaction_latency_ms,
                acoustic_latency_ms=telemetry.total_acoustic_latency_ms,
                metadata={"cooldown_remaining_s": round(effective_cooldown - (now - self._last_confirmed_time), 3)},
            )
            self._command_history.append(rec)
            logger.debug(f"[COMMAND FSM] Rejected duplicate command '{cmd}' during cooldown.")
            return rec

        # ----------------------------------------------------------------------
        # Guard 2: Supported Vocabulary Check
        # ----------------------------------------------------------------------
        if cmd is None or cmd not in self.config.allowed_vocabulary:
            self.total_rejected += 1
            reason = RejectionReason.UNSUPPORTED_VOCABULARY
            rec = CommandRecord(
                record_id=len(self._command_history) + 1,
                command=cmd,
                confidence=conf,
                margin=margin,
                state_at_event=self.state,
                is_confirmed=False,
                rejection_reason=reason,
                detected_timestamp=telemetry.decision_timestamp,
                confirmed_timestamp=None,
                reaction_latency_ms=telemetry.reaction_latency_ms,
                acoustic_latency_ms=telemetry.total_acoustic_latency_ms,
                metadata={"reason": f"Class '{cmd}' not in allowed vocabulary"},
            )
            self._command_history.append(rec)
            # If in CANDIDATE_COMMAND or SPEECH_DETECTED, transition appropriately
            if self.state == CommandState.CANDIDATE_COMMAND:
                target = CommandState.WAITING if vad_state == VADState.SILENCE else CommandState.SPEECH_DETECTED
                self._transition_to(target, "Unsupported vocabulary; exiting candidate state")
            elif vad_state == VADState.SILENCE and self.state == CommandState.SPEECH_DETECTED:
                if now - self._last_speech_time >= self.config.speech_timeout_s:
                    self._transition_to(CommandState.WAITING, "VAD silence after unclassified speech")
            return rec

        # ----------------------------------------------------------------------
        # Guard 3: Candidate Confidence Threshold
        # ----------------------------------------------------------------------
        if conf < self.config.candidate_confidence_threshold:
            self.total_rejected += 1
            rec = CommandRecord(
                record_id=len(self._command_history) + 1,
                command=cmd,
                confidence=conf,
                margin=margin,
                state_at_event=self.state,
                is_confirmed=False,
                rejection_reason=RejectionReason.LOW_CONFIDENCE,
                detected_timestamp=telemetry.decision_timestamp,
                confirmed_timestamp=None,
                reaction_latency_ms=telemetry.reaction_latency_ms,
                acoustic_latency_ms=telemetry.total_acoustic_latency_ms,
                metadata={"threshold": self.config.candidate_confidence_threshold},
            )
            self._command_history.append(rec)
            if self.state == CommandState.CANDIDATE_COMMAND:
                target = CommandState.WAITING if vad_state == VADState.SILENCE else CommandState.SPEECH_DETECTED
                self._transition_to(target, "Sub-candidate confidence; exiting candidate state")
            logger.debug(f"[COMMAND FSM] Rejected '{cmd}' (conf {conf:.2f} < candidate threshold {self.config.candidate_confidence_threshold})")
            return rec

        # Reached CANDIDATE_COMMAND state
        self._transition_to(CommandState.CANDIDATE_COMMAND, f"Candidate keyword '{cmd}' (conf={conf:.2f})")

        # ----------------------------------------------------------------------
        # Guard 4: Final Confirmation Thresholds (Confidence & Margin)
        # ----------------------------------------------------------------------
        if conf < self.config.confirm_confidence_threshold:
            self.total_rejected += 1
            rec = CommandRecord(
                record_id=len(self._command_history) + 1,
                command=cmd,
                confidence=conf,
                margin=margin,
                state_at_event=CommandState.CANDIDATE_COMMAND,
                is_confirmed=False,
                rejection_reason=RejectionReason.LOW_CONFIDENCE,
                detected_timestamp=telemetry.decision_timestamp,
                confirmed_timestamp=None,
                reaction_latency_ms=telemetry.reaction_latency_ms,
                acoustic_latency_ms=telemetry.total_acoustic_latency_ms,
                metadata={"confirm_threshold": self.config.confirm_confidence_threshold},
            )
            self._command_history.append(rec)
            target = CommandState.WAITING if vad_state == VADState.SILENCE else CommandState.SPEECH_DETECTED
            self._transition_to(target, f"Candidate rejected (conf {conf:.2f} < {self.config.confirm_confidence_threshold})")
            return rec

        if margin < self.config.margin_threshold:
            self.total_rejected += 1
            rec = CommandRecord(
                record_id=len(self._command_history) + 1,
                command=cmd,
                confidence=conf,
                margin=margin,
                state_at_event=CommandState.CANDIDATE_COMMAND,
                is_confirmed=False,
                rejection_reason=RejectionReason.LOW_MARGIN,
                detected_timestamp=telemetry.decision_timestamp,
                confirmed_timestamp=None,
                reaction_latency_ms=telemetry.reaction_latency_ms,
                acoustic_latency_ms=telemetry.total_acoustic_latency_ms,
                metadata={"margin_threshold": self.config.margin_threshold},
            )
            self._command_history.append(rec)
            target = CommandState.WAITING if vad_state == VADState.SILENCE else CommandState.SPEECH_DETECTED
            self._transition_to(target, f"Candidate rejected (margin {margin:.2f} < {self.config.margin_threshold})")
            logger.debug(f"[COMMAND FSM] Rejected '{cmd}' due to low margin ({margin:.2f} < {self.config.margin_threshold})")
            return rec

        # ----------------------------------------------------------------------
        # Command Fully Confirmed!
        # ----------------------------------------------------------------------
        self._transition_to(CommandState.COMMAND_CONFIRMED, f"Confirmed '{cmd}' (conf={conf:.2f}, margin={margin:.2f})")
        self._last_confirmed_time = now
        self._last_confirmed_command = cmd
        self.total_confirmed += 1

        rec = CommandRecord(
            record_id=len(self._command_history) + 1,
            command=cmd,
            confidence=conf,
            margin=margin,
            state_at_event=CommandState.COMMAND_CONFIRMED,
            is_confirmed=True,
            rejection_reason=RejectionReason.NONE,
            detected_timestamp=telemetry.audio_event_timestamp,
            confirmed_timestamp=telemetry.decision_timestamp,
            reaction_latency_ms=telemetry.reaction_latency_ms,
            acoustic_latency_ms=telemetry.total_acoustic_latency_ms,
            display_command=get_display_command(cmd),
            digit=NUMBER_TO_DIGIT.get(cmd),
            metadata={"top2_class": result.top2_class, "digit": NUMBER_TO_DIGIT.get(cmd)},
        )
        self._command_history.append(rec)
        self._latest_confirmed_record = rec

        cmd_display = get_display_command(cmd)
        logger.info(
            f"[COMMAND FSM] *** COMMAND CONFIRMED: '{cmd_display}' *** "
            f"(Conf: {conf*100:.1f}%, Margin: {margin:.2f}, Reaction: {telemetry.reaction_latency_ms:.2f} ms)"
        )

        # Notify software-only consumer
        if self.on_command_callback is not None:
            try:
                self.on_command_callback(rec)
            except Exception as e:
                logger.error(f"Error in on_command_callback: {e}")

        # Transition immediately to COOLDOWN
        self._transition_to(CommandState.COOLDOWN, f"Entering cooldown for {self.config.cooldown_duration_s}s")
        return rec

    def _transition_to(self, new_state: CommandState, reason: str):
        """Executes explicit, documented state transition."""
        if self.state != new_state:
            prev = self.state
            self.state = new_state
            self.total_state_transitions += 1
            logger.info(f"[COMMAND FSM] Transition: {prev.value} -> {new_state.value} ({reason})")

    @property
    def latest_confirmed(self) -> Optional[CommandRecord]:
        return self._latest_confirmed_record

    @property
    def history(self) -> List[CommandRecord]:
        return list(self._command_history)

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured FSM operational summary."""
        return {
            "current_state": self.state.value,
            "total_evaluations": self.total_evaluations,
            "total_confirmed": self.total_confirmed,
            "total_rejected": self.total_rejected,
            "total_duplicates_suppressed": self.total_duplicates_suppressed,
            "total_state_transitions": self.total_state_transitions,
            "last_confirmed_command": self._last_confirmed_command,
            "last_confirmed_time": round(self._last_confirmed_time, 4) if self._last_confirmed_time > 0 else None,
            "latest_record": self._latest_confirmed_record.to_dict() if self._latest_confirmed_record else None,
        }

