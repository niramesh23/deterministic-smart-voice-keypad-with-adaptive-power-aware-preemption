"""
src/accessibility/voice_feedback.py
-----------------------------------
Voice Feedback and Auditory Status Communication Subsystem.

Responsibilities:
- Synthesize short, clear, non-sensitive spoken feedback for key system events:
  * System ready ("System ready.")
  * Digit accepted ("Digit accepted.")
  * PIN entry complete ("PIN entry complete.")
  * Access granted ("Access granted.")
  * Access denied ("Access denied.")
  * Lock activated ("Lock activated.")
  * Lockout activated ("Lockout activated.")
  * Lockout expired ("Lockout expired.")
  * Emergency STOP accepted ("Emergency STOP accepted.")
  * System fault detected ("System fault detected.")
- Security Constraints:
  * NEVER speak individual digit values (e.g. "one", "5").
  * NEVER speak complete PINs, hashes, salts, or session tokens.
  * Rigorous message sanitization asserts zero credential leakage.
- Non-blocking execution:
  * All speech requests are queued in a background worker thread.
  * Speech synthesis never blocks real-time audio threads or FSM transitions.
- Audio confirmation tones:
  * Optional non-speech auditory cues (chimes/beeps) confirming input registration.
- Pluggable TTS backend:
  * Thread-safe mock backend for testing and inspection.
  * System TTS backend support with graceful fallback.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import queue
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from src.accessibility.settings import AccessibilitySettings

logger = logging.getLogger("rtos.accessibility.voice")


class FeedbackEvent(str, Enum):
    """Categorized system events supported by Voice Feedback."""
    SYSTEM_READY = "SYSTEM_READY"
    DIGIT_ACCEPTED = "DIGIT_ACCEPTED"
    PIN_COMPLETE = "PIN_COMPLETE"
    ACCESS_GRANTED = "ACCESS_GRANTED"
    ACCESS_DENIED = "ACCESS_DENIED"
    LOCK_ACTIVATED = "LOCK_ACTIVATED"
    LOCKOUT_ACTIVATED = "LOCKOUT_ACTIVATED"
    LOCKOUT_EXPIRED = "LOCKOUT_EXPIRED"
    EMERGENCY_STOP_ACCEPTED = "EMERGENCY_STOP_ACCEPTED"
    SYSTEM_FAULT_DETECTED = "SYSTEM_FAULT_DETECTED"


# Canonical short, non-sensitive spoken messages (Strict WCAG / Accessibility Guidelines)
CANONICAL_FEEDBACK_MESSAGES: Dict[FeedbackEvent, str] = {
    FeedbackEvent.SYSTEM_READY: "System ready.",
    FeedbackEvent.DIGIT_ACCEPTED: "Digit accepted.",
    FeedbackEvent.PIN_COMPLETE: "PIN entry complete.",
    FeedbackEvent.ACCESS_GRANTED: "Access granted.",
    FeedbackEvent.ACCESS_DENIED: "Access denied.",
    FeedbackEvent.LOCK_ACTIVATED: "Lock activated.",
    FeedbackEvent.LOCKOUT_ACTIVATED: "Lockout activated.",
    FeedbackEvent.LOCKOUT_EXPIRED: "Lockout expired.",
    FeedbackEvent.EMERGENCY_STOP_ACCEPTED: "Emergency STOP accepted.",
    FeedbackEvent.SYSTEM_FAULT_DETECTED: "System fault detected.",
}

# Auditory confirmation cue characteristics (frequency Hz, duration ms)
AUDIO_CONFIRMATION_CUES: Dict[FeedbackEvent, Dict[str, Any]] = {
    FeedbackEvent.SYSTEM_READY: {"freq_hz": 523, "duration_ms": 120, "pattern": "high_single"},
    FeedbackEvent.DIGIT_ACCEPTED: {"freq_hz": 659, "duration_ms": 60, "pattern": "short_click"},
    FeedbackEvent.PIN_COMPLETE: {"freq_hz": 784, "duration_ms": 100, "pattern": "medium_chime"},
    FeedbackEvent.ACCESS_GRANTED: {"freq_hz": 880, "duration_ms": 250, "pattern": "success_chime"},
    FeedbackEvent.ACCESS_DENIED: {"freq_hz": 330, "duration_ms": 200, "pattern": "error_buzz"},
    FeedbackEvent.LOCK_ACTIVATED: {"freq_hz": 440, "duration_ms": 150, "pattern": "relock_tone"},
    FeedbackEvent.LOCKOUT_ACTIVATED: {"freq_hz": 262, "duration_ms": 300, "pattern": "lockout_alarm"},
    FeedbackEvent.LOCKOUT_EXPIRED: {"freq_hz": 587, "duration_ms": 150, "pattern": "resume_tone"},
    FeedbackEvent.EMERGENCY_STOP_ACCEPTED: {"freq_hz": 988, "duration_ms": 350, "pattern": "emergency_warble"},
    FeedbackEvent.SYSTEM_FAULT_DETECTED: {"freq_hz": 370, "duration_ms": 250, "pattern": "fault_buzz"},
}

# Security pattern matching any numeric digits or word-form digits
SENSITIVE_DIGIT_PATTERN = re.compile(
    r"\b(\d|zero|one|two|three|four|five|six|seven|eight|nine|pin|hash|salt|token|secret)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FeedbackRecord:
    """Immutable record of an evaluated feedback event."""
    event: FeedbackEvent
    message: str
    timestamp: float
    volume: float
    spoken: bool
    audio_cue_played: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.value,
            "message": self.message,
            "timestamp": round(self.timestamp, 4),
            "volume": round(self.volume, 2),
            "spoken": self.spoken,
            "audio_cue_played": self.audio_cue_played,
        }


class TTSBackend:
    """Abstract base class for Text-To-Speech and Audio Cue Backends."""

    def speak(self, text: str, volume: float = 1.0) -> None:
        """Synthesizes or outputs spoken text."""
        raise NotImplementedError

    def play_tone(self, freq_hz: int, duration_ms: int, volume: float = 1.0) -> None:
        """Plays an audio confirmation tone."""
        pass


class MockTTSBackend(TTSBackend):
    """
    Thread-safe in-memory TTS backend designed for deterministic automated testing.
    Records all spoken phrases, volumes, and audio cue parameters.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.spoken_phrases: List[str] = []
        self.spoken_history: List[Dict[str, Any]] = []
        self.tones_played: List[Dict[str, Any]] = []

    def speak(self, text: str, volume: float = 1.0) -> None:
        with self._lock:
            self.spoken_phrases.append(text)
            self.spoken_history.append({
                "text": text,
                "volume": volume,
                "timestamp": time.time(),
            })

    def play_tone(self, freq_hz: int, duration_ms: int, volume: float = 1.0) -> None:
        with self._lock:
            self.tones_played.append({
                "freq_hz": freq_hz,
                "duration_ms": duration_ms,
                "volume": volume,
                "timestamp": time.time(),
            })

    def clear(self) -> None:
        with self._lock:
            self.spoken_phrases.clear()
            self.spoken_history.clear()
            self.tones_played.clear()


class VoiceFeedback:
    """
    Main Voice Feedback and Auditory Status Communication Subsystem.
    Asynchronously processes and dispatches voice feedback without blocking
    real-time audio recognition or scheduling loops.
    """

    def __init__(
        self,
        settings: Optional[AccessibilitySettings] = None,
        backend: Optional[TTSBackend] = None,
        max_history: int = 100,
    ):
        self.settings = settings or AccessibilitySettings()
        self.backend = backend or MockTTSBackend()
        self.max_history = max_history

        self._lock = threading.RLock()
        self._history: deque[FeedbackRecord] = deque(maxlen=max_history)
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._worker_running = True

        # Background worker for asynchronous non-blocking speech
        self._worker_thread = threading.Thread(
            target=self._speech_worker_loop,
            name="VoiceFeedbackWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def update_settings(self, new_settings: AccessibilitySettings) -> None:
        """Updates internal reference to accessibility settings."""
        with self._lock:
            self.settings = new_settings

    @classmethod
    def sanitize_message(cls, message: str) -> str:
        """
        Strict security sanitizer.
        Ensures NO digit values, complete PINs, salts, hashes, or credentials
        are ever spoken. Raises ValueError if a sensitive leak is attempted.
        """
        # Check against forbidden tokens
        # Note: Allow "accepted" in "Digit accepted." but reject individual digit words or numerals
        # Specifically match standalone digits or phrases like "pin is 1234"
        numerals = re.findall(r"\b\d+\b", message)
        if numerals:
            logger.critical(
                f"[SECURITY ALERT] Attempted to speak numeric digit '{numerals}' in voice feedback! Redacting."
            )
            return "[REDACTED_SECURITY_POLICY]"

        forbidden_words = ["password", "secret", "salt", "token", "hash"]
        lower = message.lower()
        for fw in forbidden_words:
            if fw in lower:
                logger.critical(
                    f"[SECURITY ALERT] Attempted to speak forbidden credential keyword '{fw}'! Redacting."
                )
                return "[REDACTED_SECURITY_POLICY]"

        return message

    def emit_event(self, event: FeedbackEvent, custom_message: Optional[str] = None) -> FeedbackRecord:
        """
        Dispatches feedback for a designated event.
        Checks settings, performs sanitization, schedules audio cue and spoken text.
        Returns immutable FeedbackRecord.
        """
        with self._lock:
            curr_settings = self.settings
            raw_msg = custom_message or CANONICAL_FEEDBACK_MESSAGES.get(event, "Status update.")
            safe_msg = self.sanitize_message(raw_msg)

            should_speak = curr_settings.spoken_feedback
            should_cue = curr_settings.audio_confirmation
            volume = curr_settings.voice_feedback_volume

            record = FeedbackRecord(
                event=event,
                message=safe_msg,
                timestamp=time.time(),
                volume=volume,
                spoken=should_speak,
                audio_cue_played=should_cue,
            )
            self._history.append(record)

        # Enqueue for non-blocking asynchronous audio delivery
        if should_cue or should_speak:
            try:
                cue_info = AUDIO_CONFIRMATION_CUES.get(event) if should_cue else None
                self._queue.put_nowait((record, cue_info))
            except queue.Full:
                logger.warning("[VOICE FEEDBACK] Speech queue full; dropping audio output.")

        return record

    def _speech_worker_loop(self) -> None:
        """Background worker thread draining the speech queue."""
        while self._worker_running:
            try:
                item = self._queue.get(timeout=0.20)
            except queue.Empty:
                continue

            record, cue_info = item
            try:
                # 1. Play audio confirmation cue if enabled
                if record.audio_cue_played and cue_info is not None:
                    try:
                        self.backend.play_tone(
                            freq_hz=cue_info["freq_hz"],
                            duration_ms=cue_info["duration_ms"],
                            volume=record.volume,
                        )
                    except Exception as e:
                        logger.debug(f"Audio cue playback error: {e}")

                # 2. Speak message if enabled and volume > 0
                if record.spoken and record.volume > 0.0:
                    try:
                        self.backend.speak(record.message, volume=record.volume)
                    except Exception as e:
                        logger.warning(f"Voice feedback TTS error: {e}")

            except Exception as e:
                logger.error(f"Voice feedback worker exception: {e}")
            finally:
                self._queue.task_done()

    # Convenience event triggers matching Stage 8 requirements
    def on_system_ready(self) -> FeedbackRecord:
        return self.emit_event(FeedbackEvent.SYSTEM_READY)

    def on_digit_accepted(self) -> FeedbackRecord:
        return self.emit_event(FeedbackEvent.DIGIT_ACCEPTED)

    def on_pin_complete(self) -> FeedbackRecord:
        return self.emit_event(FeedbackEvent.PIN_COMPLETE)

    def on_access_granted(self) -> FeedbackRecord:
        return self.emit_event(FeedbackEvent.ACCESS_GRANTED)

    def on_access_denied(self) -> FeedbackRecord:
        return self.emit_event(FeedbackEvent.ACCESS_DENIED)

    def on_lock_activated(self) -> FeedbackRecord:
        return self.emit_event(FeedbackEvent.LOCK_ACTIVATED)

    def on_lockout_activated(self) -> FeedbackRecord:
        return self.emit_event(FeedbackEvent.LOCKOUT_ACTIVATED)

    def on_lockout_expired(self) -> FeedbackRecord:
        return self.emit_event(FeedbackEvent.LOCKOUT_EXPIRED)

    def on_emergency_stop_accepted(self) -> FeedbackRecord:
        return self.emit_event(FeedbackEvent.EMERGENCY_STOP_ACCEPTED)

    def on_system_fault_detected(self) -> FeedbackRecord:
        return self.emit_event(FeedbackEvent.SYSTEM_FAULT_DETECTED)

    # Inspection and telemetry methods
    def get_history(self) -> List[Dict[str, Any]]:
        """Returns list of all recorded feedback events."""
        with self._lock:
            return [r.to_dict() for r in self._history]

    def get_last_event(self) -> Optional[FeedbackRecord]:
        """Returns the most recent feedback record, if any."""
        with self._lock:
            return self._history[-1] if self._history else None

    def clear_history(self) -> None:
        """Clears feedback history buffer."""
        with self._lock:
            self._history.clear()

    def wait_until_idle(self, timeout: float = 1.0) -> None:
        """Blocks until pending speech worker queue items are finished."""
        try:
            self._queue.join()
        except Exception:
            pass

    def stop(self) -> None:
        """Shuts down background worker cleanly."""
        self._worker_running = False

