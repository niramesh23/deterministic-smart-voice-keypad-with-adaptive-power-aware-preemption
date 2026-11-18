"""
src/dsp/vad.py
--------------
Voice Activity Detector (VAD) with adaptive noise tracking, onset confirmation,
hangover bridging, and pre-roll integration.

LIMITATION NOTICE:
    RMS-based energy detection functions as an acoustic energy gate, NOT a phoneme
    or semantic speech classifier. It identifies segments where signal energy
    significantly exceeds the background acoustic noise floor. Non-speech acoustic
    transients (such as door slams, desk taps, microphone handling noise, or sharp coughs)
    can exceed the energy threshold. Downstream rejection of non-command transients
    is handled by neural network unknown-word and silence classification in Stage 5.
"""

from dataclasses import dataclass
from enum import Enum, auto
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.dsp.energy import calculate_rms
from src.dsp.noise import AdaptiveNoiseFloor
from src.dsp.preroll import PreRollBuffer


class VADState(Enum):
    """Voice Activity Detection state machine."""
    SILENCE = 0             # Below threshold, background noise adaptation active
    ONSET_CANDIDATE = 1     # Single hop above threshold, awaiting onset confirmation
    SPEECH_ACTIVE = 2       # Confirmed speech, accumulating utterance samples
    HANGOVER = 3            # Energy dropped below threshold, holding gate open for gaps
    UTTERANCE_COMPLETE = 4  # Utterance terminated; ready for feature extraction/inference


@dataclass
class UtteranceSegment:
    """
    Complete isolated speech command segment ready for feature extraction and inference.

    Attributes:
        audio: 1D array of 16-bit PCM or float32 samples (includes prepended pre-roll).
        sample_rate: Audio sampling frequency in Hz (16,000 Hz).
        duration_ms: Total duration of segment in milliseconds.
        start_timestamp: Perf counter timestamp of utterance onset.
        peak_rms: Maximum RMS energy recorded during the utterance.
        pre_roll_samples: Number of pre-roll samples prepended before trigger point.
    """
    audio: np.ndarray
    sample_rate: int
    duration_ms: float
    start_timestamp: float
    peak_rms: float
    pre_roll_samples: int


class VoiceActivityDetector:
    """
    Real-time energy-based Voice Activity Detector with hangover and pre-roll.

    Parameters:
        sample_rate: Target sampling rate (default: 16,000 Hz).
        hop_size: Hop duration in samples (default: 320 samples = 20.0 ms).
        threshold_factor: Multiplier above adaptive noise floor for trigger (default: 2.5).
        min_speech_duration_ms: Minimum duration to accept as valid utterance (default: 250.0 ms).
        max_speech_duration_ms: Maximum duration before forcing utterance completion (default: 1000.0 ms).
        hangover_duration_ms: Duration to maintain active state across inter-syllable gaps (default: 160.0 ms).
        pre_roll_duration_ms: Duration of audio preceding trigger to prepend (default: 200.0 ms).
        onset_confirmation_hops: Consecutive above-threshold hops required to confirm onset (default: 2).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_size: int = 320,
        threshold_factor: float = 2.5,
        min_speech_duration_ms: float = 180.0,
        max_speech_duration_ms: float = 1000.0,
        hangover_duration_ms: float = 160.0,
        pre_roll_duration_ms: float = 200.0,
        onset_confirmation_hops: int = 2,
        initial_noise_floor: float = 0.005,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_ms = max_speech_duration_ms
        self.onset_confirmation_hops = max(1, onset_confirmation_hops)

        # Derived timing in terms of hops
        hop_ms = (hop_size / sample_rate) * 1000.0
        self.hangover_max_hops = int(round(hangover_duration_ms / hop_ms))
        self.min_speech_hops = int(round(min_speech_duration_ms / hop_ms))
        self.max_speech_hops = int(round(max_speech_duration_ms / hop_ms))

        # Core subsystems
        self.noise_floor = AdaptiveNoiseFloor(
            initial_noise_floor=initial_noise_floor,
            threshold_factor=threshold_factor,
        )
        self.preroll_buffer = PreRollBuffer(
            sample_rate=sample_rate,
            duration_ms=pre_roll_duration_ms,
            hop_size=hop_size,
        )

        # State tracking
        self.state = VADState.SILENCE
        self._consecutive_above_thresh = 0
        self._hangover_counter = 0
        self._current_utterance_hops: List[np.ndarray] = []
        self._candidate_hops: List[np.ndarray] = []
        self._utterance_start_time = 0.0
        self._peak_rms = 0.0
        self._captured_preroll: Optional[np.ndarray] = None

        # Telemetry
        self.total_hops_processed = 0
        self.utterances_completed = 0
        self.transients_rejected = 0
        self.state_history: List[Tuple[float, VADState]] = []

    def process_hop(self, hop_data: np.ndarray) -> Optional[UtteranceSegment]:
        """
        Processes a single audio hop (320 samples) and updates the VAD state machine.

        Parameters:
            hop_data: 1D array of audio samples (length = hop_size).

        Returns:
            UtteranceSegment if an utterance just completed, otherwise None.
        """
        now = time.perf_counter()
        self.total_hops_processed += 1
        hop_arr = np.asarray(hop_data)

        # 1. Compute frame RMS energy
        rms = calculate_rms(hop_arr)
        current_threshold = self.noise_floor.threshold
        is_above_threshold = (rms >= current_threshold)

        completed_segment: Optional[UtteranceSegment] = None

        # 2. State Machine Transitions
        if self.state == VADState.SILENCE:
            # Feed pre-roll circular history buffer
            self.preroll_buffer.push_hop(hop_arr)

            if is_above_threshold:
                self._consecutive_above_thresh = 1
                self._candidate_hops = [hop_arr.copy()]
                if self.onset_confirmation_hops <= 1:
                    # Immediate activation
                    self._transition_to_active(now, rms)
                else:
                    self._set_state(VADState.ONSET_CANDIDATE, now)
            else:
                # Update background noise floor during quiet periods
                self.noise_floor.update(rms, is_speech_active=False)

        elif self.state == VADState.ONSET_CANDIDATE:
            if is_above_threshold:
                self._consecutive_above_thresh += 1
                self._candidate_hops.append(hop_arr.copy())
                if self._consecutive_above_thresh >= self.onset_confirmation_hops:
                    self._transition_to_active(now, rms)
            else:
                # False alarm transient; return to silence without triggering speech
                self.transients_rejected += 1
                for c_hop in self._candidate_hops:
                    self.preroll_buffer.push_hop(c_hop)
                self.preroll_buffer.push_hop(hop_arr)
                self._candidate_hops.clear()
                self._consecutive_above_thresh = 0
                self._set_state(VADState.SILENCE, now)
                self.noise_floor.update(rms, is_speech_active=False)

        elif self.state == VADState.SPEECH_ACTIVE:
            self._current_utterance_hops.append(hop_arr.copy())
            self._peak_rms = max(self._peak_rms, rms)

            if not is_above_threshold:
                # Transition to hangover state to bridge potential silence gaps
                self._hangover_counter = 1
                self._set_state(VADState.HANGOVER, now)
            elif len(self._current_utterance_hops) >= self.max_speech_hops:
                # Forced termination at max speech duration (e.g. 1.0s limit)
                completed_segment = self._finalize_utterance(now)

        elif self.state == VADState.HANGOVER:
            self._current_utterance_hops.append(hop_arr.copy())
            self._peak_rms = max(self._peak_rms, rms)

            if is_above_threshold:
                # Speech resumed before hangover expired! Revert to SPEECH_ACTIVE
                self._hangover_counter = 0
                self._set_state(VADState.SPEECH_ACTIVE, now)
            else:
                self._hangover_counter += 1
                if self._hangover_counter >= self.hangover_max_hops:
                    # Hangover expired: speech has cleanly ended
                    completed_segment = self._finalize_utterance(now)
                elif len(self._current_utterance_hops) >= self.max_speech_hops:
                    completed_segment = self._finalize_utterance(now)

        return completed_segment

    def _transition_to_active(self, now: float, rms: float):
        """Transitions into confirmed SPEECH_ACTIVE and snapshots pre-roll audio."""
        self._set_state(VADState.SPEECH_ACTIVE, now)
        self._utterance_start_time = now
        self._peak_rms = rms

        # Capture preceding 200 ms from pre-roll buffer
        self._captured_preroll = self.preroll_buffer.get_preroll_samples()

        # Add candidate onset hops into active speech buffer
        self._current_utterance_hops = list(self._candidate_hops)
        self._candidate_hops.clear()

    def _finalize_utterance(self, now: float) -> Optional[UtteranceSegment]:
        """Finalizes accumulated speech hops into an UtteranceSegment."""
        total_hops = len(self._current_utterance_hops)

        # Validate minimum duration constraint
        if total_hops < self.min_speech_hops:
            self.transients_rejected += 1
            # Push remaining hops to pre-roll to resume tracking
            for h in self._current_utterance_hops:
                self.preroll_buffer.push_hop(h)
            self._current_utterance_hops.clear()
            self._set_state(VADState.SILENCE, now)
            return None

        # Build full audio array: [Pre-roll (3200 samples) + Active Speech]
        speech_audio = np.concatenate(self._current_utterance_hops)
        preroll_audio = self._captured_preroll if self._captured_preroll is not None else np.array([], dtype=speech_audio.dtype)
        full_audio = np.concatenate([preroll_audio, speech_audio])

        duration_ms = (len(full_audio) / self.sample_rate) * 1000.0
        segment = UtteranceSegment(
            audio=full_audio,
            sample_rate=self.sample_rate,
            duration_ms=duration_ms,
            start_timestamp=self._utterance_start_time,
            peak_rms=self._peak_rms,
            pre_roll_samples=len(preroll_audio),
        )

        self.utterances_completed += 1
        self._current_utterance_hops.clear()
        self._captured_preroll = None
        self.preroll_buffer.reset()
        self._set_state(VADState.SILENCE, now)

        return segment

    def _set_state(self, new_state: VADState, now: float):
        if self.state != new_state:
            self.state = new_state
            if len(self.state_history) < 1000:
                self.state_history.append((now, new_state))

    def flush(self) -> Optional[UtteranceSegment]:
        """Flushes and finalizes any active speech utterance currently in progress."""
        if self.state in (VADState.SPEECH_ACTIVE, VADState.HANGOVER) and self._current_utterance_hops:
            return self._finalize_utterance(time.perf_counter())
        return None

    def reset(self):
        """Resets detector state and buffers."""
        self.state = VADState.SILENCE
        self._consecutive_above_thresh = 0
        self._hangover_counter = 0
        self._current_utterance_hops.clear()
        self._candidate_hops.clear()
        self.preroll_buffer.reset()
        self.noise_floor.reset()

    def get_diagnostics(self) -> Dict[str, any]:
        return {
            "state": self.state.name,
            "noise_floor": self.noise_floor.noise_floor,
            "threshold": self.noise_floor.threshold,
            "utterances_completed": self.utterances_completed,
            "transients_rejected": self.transients_rejected,
            "total_hops_processed": self.total_hops_processed,
        }
