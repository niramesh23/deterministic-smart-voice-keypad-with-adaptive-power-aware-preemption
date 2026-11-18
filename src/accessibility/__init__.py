"""
src/accessibility/__init__.py
-----------------------------
Accessibility package exports for RTOS Voice Smart-Lock application.
"""

from src.accessibility.settings import (
    AccessibilitySettings,
    AccessibilitySettingsManager,
    MIN_PIN_TIMEOUT_S,
    MAX_PIN_TIMEOUT_S,
    DEFAULT_PIN_TIMEOUT_S,
    MIN_VOLUME,
    MAX_VOLUME,
    DEFAULT_VOLUME,
)
from src.accessibility.voice_feedback import (
    VoiceFeedback,
    FeedbackEvent,
    FeedbackRecord,
    TTSBackend,
    MockTTSBackend,
    CANONICAL_FEEDBACK_MESSAGES,
    AUDIO_CONFIRMATION_CUES,
)

__all__ = [
    "AccessibilitySettings",
    "AccessibilitySettingsManager",
    "MIN_PIN_TIMEOUT_S",
    "MAX_PIN_TIMEOUT_S",
    "DEFAULT_PIN_TIMEOUT_S",
    "MIN_VOLUME",
    "MAX_VOLUME",
    "DEFAULT_VOLUME",
    "VoiceFeedback",
    "FeedbackEvent",
    "FeedbackRecord",
    "TTSBackend",
    "MockTTSBackend",
    "CANONICAL_FEEDBACK_MESSAGES",
    "AUDIO_CONFIRMATION_CUES",
]

