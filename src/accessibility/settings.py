"""
src/accessibility/settings.py
-----------------------------
Accessibility Settings and Preference Persistence Module.

Responsibilities:
- Manage user accessibility configurations:
  * Large text mode (increases font scaling and UI sizing).
  * High-contrast mode (WCAG AAA compliant color palette).
  * Clear status indicators (textual badges + symbolic redundant indicators).
  * Spoken voice feedback (audible status and event messages).
  * Audio confirmation (short acoustic tones/chimes on user interactions).
  * Adjustable PIN entry timeout (bounded within safe limits [5.0s, 60.0s]).
  * Adjustable voice feedback volume (normalized [0.0, 1.0]).
  * Reduced visual complexity (eliminates animations, transitions, and decorative effects).
- Safe, atomic file persistence (JSON with .tmp write and atomic rename).
- Resilient error handling (graceful fallback to safe defaults on missing or corrupt files).
- Enforces strict security constraints:
  * Accessibility settings NEVER bypass PIN verification.
  * Accessibility settings NEVER bypass lockout or reduce security thresholds.
  * Accessibility settings NEVER change user permissions or roles.
  * PIN entry timeout cannot be set to 0, negative, or excessive/infinite values.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import os
import pathlib
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("rtos.accessibility.settings")

# Security limits for adjustable parameters
MIN_PIN_TIMEOUT_S = 5.0
MAX_PIN_TIMEOUT_S = 60.0
DEFAULT_PIN_TIMEOUT_S = 10.0

MIN_VOLUME = 0.0
MAX_VOLUME = 1.0
DEFAULT_VOLUME = 1.0


@dataclass
class AccessibilitySettings:
    """Configurable accessibility and usability parameters."""
    large_text: bool = False
    high_contrast: bool = False
    clear_status_indicators: bool = True
    spoken_feedback: bool = True
    audio_confirmation: bool = True
    pin_entry_timeout_s: float = DEFAULT_PIN_TIMEOUT_S
    voice_feedback_volume: float = DEFAULT_VOLUME
    reduced_visual_complexity: bool = False
    persistence_path: str = "configs/accessibility_settings.json"

    def validate_and_clamp(self) -> None:
        """
        Enforces security invariants and bounds on all fields.
        Prevents weakening authentication security.
        """
        # Ensure pin_entry_timeout_s is strictly bounded
        try:
            val = float(self.pin_entry_timeout_s)
            if val < MIN_PIN_TIMEOUT_S:
                logger.warning(
                    f"Requested pin_entry_timeout_s ({val}s) below minimum ({MIN_PIN_TIMEOUT_S}s); clamping."
                )
                self.pin_entry_timeout_s = MIN_PIN_TIMEOUT_S
            elif val > MAX_PIN_TIMEOUT_S:
                logger.warning(
                    f"Requested pin_entry_timeout_s ({val}s) exceeds maximum ({MAX_PIN_TIMEOUT_S}s); clamping."
                )
                self.pin_entry_timeout_s = MAX_PIN_TIMEOUT_S
            else:
                self.pin_entry_timeout_s = val
        except (ValueError, TypeError):
            self.pin_entry_timeout_s = DEFAULT_PIN_TIMEOUT_S

        # Ensure voice_feedback_volume is clamped to [0.0, 1.0]
        try:
            vol = float(self.voice_feedback_volume)
            self.voice_feedback_volume = max(MIN_VOLUME, min(MAX_VOLUME, vol))
        except (ValueError, TypeError):
            self.voice_feedback_volume = DEFAULT_VOLUME

        # Enforce boolean types
        self.large_text = bool(self.large_text)
        self.high_contrast = bool(self.high_contrast)
        self.clear_status_indicators = bool(self.clear_status_indicators)
        self.spoken_feedback = bool(self.spoken_feedback)
        self.audio_confirmation = bool(self.audio_confirmation)
        self.reduced_visual_complexity = bool(self.reduced_visual_complexity)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes settings to a dictionary for API/JSON export."""
        return {
            "large_text": self.large_text,
            "high_contrast": self.high_contrast,
            "clear_status_indicators": self.clear_status_indicators,
            "spoken_feedback": self.spoken_feedback,
            "audio_confirmation": self.audio_confirmation,
            "pin_entry_timeout_s": round(self.pin_entry_timeout_s, 1),
            "voice_feedback_volume": round(self.voice_feedback_volume, 2),
            "reduced_visual_complexity": self.reduced_visual_complexity,
            "persistence_path": self.persistence_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], persistence_path: Optional[str] = None) -> "AccessibilitySettings":
        """Instantiates settings from dictionary with strict sanitization and bounds checking."""
        path = persistence_path or data.get("persistence_path", "configs/accessibility_settings.json")
        settings = cls(
            large_text=bool(data.get("large_text", False)),
            high_contrast=bool(data.get("high_contrast", False)),
            clear_status_indicators=bool(data.get("clear_status_indicators", True)),
            spoken_feedback=bool(data.get("spoken_feedback", True)),
            audio_confirmation=bool(data.get("audio_confirmation", True)),
            pin_entry_timeout_s=float(data.get("pin_entry_timeout_s", DEFAULT_PIN_TIMEOUT_S)),
            voice_feedback_volume=float(data.get("voice_feedback_volume", DEFAULT_VOLUME)),
            reduced_visual_complexity=bool(data.get("reduced_visual_complexity", False)),
            persistence_path=str(path),
        )
        settings.validate_and_clamp()
        return settings


class AccessibilitySettingsManager:
    """
    Thread-safe manager for loading, updating, persisting, and notifying
    changes to AccessibilitySettings.
    """

    def __init__(
        self,
        config_path: str = "configs/accessibility_settings.json",
        auto_load: bool = True,
    ):
        self.config_path = config_path
        self._lock = threading.RLock()
        self._listeners: List[Callable[[AccessibilitySettings], None]] = []
        self._settings = AccessibilitySettings(persistence_path=self.config_path)

        if auto_load:
            self.load()

    @property
    def settings(self) -> AccessibilitySettings:
        """Returns a snapshot of the current accessibility settings."""
        with self._lock:
            # Return a copy to avoid external concurrent mutation
            return AccessibilitySettings.from_dict(self._settings.to_dict(), self.config_path)

    def add_listener(self, listener: Callable[[AccessibilitySettings], None]) -> None:
        """Registers a callback function invoked whenever settings are updated."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[AccessibilitySettings], None]) -> None:
        """Unregisters a callback function."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def load(self) -> AccessibilitySettings:
        """
        Loads settings from disk JSON.
        Falls back to safe defaults if file is missing, empty, or corrupted.
        """
        with self._lock:
            path = pathlib.Path(self.config_path)
            if not path.exists():
                logger.info(f"Accessibility config file not found at {path}; using safe defaults.")
                self._settings = AccessibilitySettings(persistence_path=self.config_path)
                return self.settings

            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        logger.warning(f"Accessibility config file {path} is empty; using defaults.")
                        self._settings = AccessibilitySettings(persistence_path=self.config_path)
                        return self.settings
                    data = json.loads(content)
                    if not isinstance(data, dict):
                        logger.warning(f"Invalid accessibility config format in {path}; using defaults.")
                        self._settings = AccessibilitySettings(persistence_path=self.config_path)
                        return self.settings

                    self._settings = AccessibilitySettings.from_dict(data, persistence_path=self.config_path)
                    logger.info(f"Loaded accessibility settings from {path}.")
            except Exception as e:
                logger.warning(
                    f"Failed to parse accessibility settings from {path} ({e}); falling back to safe defaults."
                )
                self._settings = AccessibilitySettings(persistence_path=self.config_path)

            return self.settings

    def save(self) -> bool:
        """
        Safely persists current settings to disk using atomic temporary file replacement.
        """
        with self._lock:
            self._settings.validate_and_clamp()
            target_path = pathlib.Path(self.config_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target_path.with_suffix(".tmp")

            try:
                data = self._settings.to_dict()
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic replace
                tmp_path.replace(target_path)
                logger.info(f"Persisted accessibility settings to {target_path}.")
                return True
            except Exception as e:
                logger.error(f"Failed to persist accessibility settings to {target_path}: {e}")
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                return False

    def update(self, updates: Dict[str, Any], persist: bool = True) -> AccessibilitySettings:
        """
        Applies updates to accessibility settings, validates bounds, optionally persists,
        and notifies registered listeners.
        """
        with self._lock:
            curr = self._settings.to_dict()
            for key, val in updates.items():
                if key in curr and key != "persistence_path":
                    curr[key] = val

            self._settings = AccessibilitySettings.from_dict(curr, persistence_path=self.config_path)
            if persist:
                self.save()

            snapshot = self.settings

        # Notify listeners outside lock
        for listener in list(self._listeners):
            try:
                listener(snapshot)
            except Exception as e:
                logger.error(f"Error in accessibility settings listener: {e}")

        return snapshot

    def reset_defaults(self, persist: bool = True) -> AccessibilitySettings:
        """Resets all accessibility settings to safe factory defaults."""
        with self._lock:
            self._settings = AccessibilitySettings(persistence_path=self.config_path)
            if persist:
                self.save()
            snapshot = self.settings

        for listener in list(self._listeners):
            try:
                listener(snapshot)
            except Exception as e:
                logger.error(f"Error in accessibility settings listener: {e}")

        return snapshot

