"""
src/rts/adaptive.py
-------------------
Adaptive Workload Controller for Real-Time Voice Recognition Pipeline.

Architecture & State Machine:
- States:
  * IDLE: Sustained silence detected by VAD. Avoids unneeded model inference, preserves buffers,
    continues audio acquisition, maintains pre-roll, and keeps monitoring active to catch speech onset.
  * NORMAL: Confirmed speech activity or recent utterance. Full pipeline active with standard
    sliding-window inference (every 5 hops = 100ms) and utterance boundary classification.
  * OVERLOAD: Computational pressure detected (queue occupancy >= threshold, consecutive deadline
    misses, or high queue delay). Enforces workload shedding: suppresses intermediate sliding-window
    inferences, prunes stale queued inference jobs, and defers non-critical monitoring tasks.
  * RECOVERY: Queue backlog drained (<= low threshold) and zero deadline misses for a stable window.
    Gradually restores normal processing cadence with hysteresis to prevent state flapping.

Guarantees:
- Audio acquisition and ping-pong buffering are NEVER stopped or blocked.
- Model input shape is NEVER altered; feature extraction parameters remain identical.
- Every state transition and adaptation decision is explicitly recorded with audit telemetry.
- All adaptation actions are deterministic, bounded, and fully reversible.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.dsp.vad import VADState

logger = logging.getLogger("rtos.rts.adaptive")


class ControllerState(str, Enum):
    """Discrete operational states of the adaptive workload controller."""
    IDLE = "IDLE"          # Sustained silence; zero unprompted inference, continuous audio ingest
    NORMAL = "NORMAL"      # Active speech; standard feature extraction and inference cadence
    OVERLOAD = "OVERLOAD"  # High queue occupancy or deadline misses; sheds redundant inference
    RECOVERY = "RECOVERY"  # Pressure cleared; gradual return to normal cadence with hysteresis


class AdaptationAction(str, Enum):
    """Actions taken by the adaptive controller to modulate pipeline workload."""
    NO_ACTION = "NO_ACTION"
    ENTER_IDLE = "ENTER_IDLE"
    EXIT_IDLE = "EXIT_IDLE"
    ENTER_OVERLOAD = "ENTER_OVERLOAD"
    THROTTLE_INFERENCE = "THROTTLE_INFERENCE"
    PRUNE_STALE_JOBS = "PRUNE_STALE_JOBS"
    ENTER_RECOVERY = "ENTER_RECOVERY"
    RESTORE_NORMAL = "RESTORE_NORMAL"


@dataclass
class AdaptationDecision:
    """Audit record capturing an individual adaptation state change or action."""
    decision_index: int
    timestamp: float
    previous_state: ControllerState
    new_state: ControllerState
    action: AdaptationAction
    trigger_reason: str
    queue_occupancy: int
    consecutive_deadline_misses: int
    recent_queue_delay_ms: float
    dropped_blocks: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "timestamp": round(self.timestamp, 6),
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "action": self.action.value,
            "trigger_reason": self.trigger_reason,
            "queue_occupancy": self.queue_occupancy,
            "consecutive_deadline_misses": self.consecutive_deadline_misses,
            "recent_queue_delay_ms": round(self.recent_queue_delay_ms, 3),
            "dropped_blocks": self.dropped_blocks,
            "metadata": self.metadata,
        }


@dataclass
class AdaptiveControllerConfig:
    """Configurable thresholds and parameters for adaptive state transitions."""
    # Silence & Idle Detection
    idle_silence_hops_threshold: int = 10     # 10 hops (200 ms) of continuous silence -> IDLE

    # Overload Detection
    overload_queue_high_threshold: int = 8    # Queue occupancy >= 8 blocks -> OVERLOAD
    overload_consecutive_misses: int = 2      # >= 2 consecutive deadline misses -> OVERLOAD
    overload_queue_delay_ms: float = 15.0     # Mean queue delay >= 15 ms -> OVERLOAD

    # Recovery & Hysteresis
    recovery_queue_low_threshold: int = 2     # Queue occupancy <= 2 blocks -> enter RECOVERY
    recovery_stable_hops: int = 15            # 15 hops (300 ms) of stable execution -> NORMAL

    # Inference Cadence Modulations
    normal_inference_stride: int = 5          # Normal: evaluate sliding context every 5 hops (100 ms)
    recovery_inference_stride: int = 10       # Recovery: evaluate every 10 hops (200 ms)


class AdaptiveProcessingController:
    """
    Closed-loop adaptive controller modulating pipeline workload based on real-time feedback.
    """

    def __init__(self, config: Optional[AdaptiveControllerConfig] = None):
        self.config = config or AdaptiveControllerConfig()
        self.state = ControllerState.NORMAL

        # State tracking counters
        self.silence_hops_count = 0
        self.consecutive_deadline_misses = 0
        self.recovery_stable_hops_count = 0
        self.total_state_transitions = 0
        self.total_overload_events = 0
        self.total_inferences_suppressed = 0
        self.total_stale_jobs_pruned = 0

        # Moving average queue delay tracker
        self._recent_delays: List[float] = []

        # Audit decision history
        self._decision_history: List[AdaptationDecision] = []

    def reset(self):
        """Resets controller state and internal counters."""
        self.state = ControllerState.NORMAL
        self.silence_hops_count = 0
        self.consecutive_deadline_misses = 0
        self.recovery_stable_hops_count = 0
        self._recent_delays.clear()
        self._decision_history.clear()
        self.total_state_transitions = 0
        self.total_overload_events = 0
        self.total_inferences_suppressed = 0
        self.total_stale_jobs_pruned = 0

    def record_job_completion(self, deadline_missed: bool, queue_delay_ms: float):
        """Updates deadline miss counters and queue delay moving average."""
        if deadline_missed:
            self.consecutive_deadline_misses += 1
            self.recovery_stable_hops_count = 0
        else:
            self.consecutive_deadline_misses = 0

        self._recent_delays.append(queue_delay_ms)
        if len(self._recent_delays) > 20:
            self._recent_delays.pop(0)

    @property
    def mean_recent_queue_delay_ms(self) -> float:
        if not self._recent_delays:
            return 0.0
        return sum(self._recent_delays) / len(self._recent_delays)

    @property
    def decision_history(self) -> List[AdaptationDecision]:
        return list(self._decision_history)

    def evaluate_state(
        self,
        vad_state: VADState,
        queue_occupancy: int,
        dropped_blocks: int = 0,
    ) -> ControllerState:
        """
        Evaluates system observables and computes next controller state with hysteresis.
        Returns the updated state.
        """
        prev_state = self.state
        now = time.perf_counter()
        avg_delay = self.mean_recent_queue_delay_ms

        # ----------------------------------------------------------------------
        # 1. Check for OVERLOAD condition (Highest precedence)
        # ----------------------------------------------------------------------
        is_overloaded = (
            queue_occupancy >= self.config.overload_queue_high_threshold
            or self.consecutive_deadline_misses >= self.config.overload_consecutive_misses
            or (self.state == ControllerState.RECOVERY and self.consecutive_deadline_misses >= 1)
            or avg_delay >= self.config.overload_queue_delay_ms
            or dropped_blocks > 0
        )

        if is_overloaded:
            if self.state != ControllerState.OVERLOAD:
                self.state = ControllerState.OVERLOAD
                self.total_overload_events += 1
                self.total_state_transitions += 1
                self.recovery_stable_hops_count = 0

                trigger = []
                if queue_occupancy >= self.config.overload_queue_high_threshold:
                    trigger.append(f"Queue occupancy {queue_occupancy} >= {self.config.overload_queue_high_threshold}")
                if self.consecutive_deadline_misses >= self.config.overload_consecutive_misses:
                    trigger.append(f"Consecutive misses {self.consecutive_deadline_misses} >= {self.config.overload_consecutive_misses}")
                elif prev_state == ControllerState.RECOVERY and self.consecutive_deadline_misses >= 1:
                    trigger.append(f"Deadline miss during RECOVERY (misses={self.consecutive_deadline_misses})")
                if avg_delay >= self.config.overload_queue_delay_ms:
                    trigger.append(f"Mean delay {avg_delay:.1f}ms >= {self.config.overload_queue_delay_ms}ms")
                if dropped_blocks > 0:
                    trigger.append(f"Dropped blocks: {dropped_blocks}")

                self._record_decision(
                    prev_state=prev_state,
                    new_state=self.state,
                    action=AdaptationAction.ENTER_OVERLOAD,
                    reason="; ".join(trigger),
                    queue_occ=queue_occupancy,
                    delay=avg_delay,
                    drops=dropped_blocks,
                )
                logger.warning(f"[ADAPTIVE CONTROLLER] Entered OVERLOAD: {'; '.join(trigger)}")
            return self.state

        # ----------------------------------------------------------------------
        # 2. In OVERLOAD: Check for transition to RECOVERY
        # ----------------------------------------------------------------------
        if self.state == ControllerState.OVERLOAD:
            # Must satisfy lower hysteresis threshold
            if (
                queue_occupancy <= self.config.recovery_queue_low_threshold
                and self.consecutive_deadline_misses == 0
                and avg_delay < self.config.overload_queue_delay_ms / 2.0
            ):
                self.state = ControllerState.RECOVERY
                self.total_state_transitions += 1
                self.recovery_stable_hops_count = 1
                self._record_decision(
                    prev_state=prev_state,
                    new_state=self.state,
                    action=AdaptationAction.ENTER_RECOVERY,
                    reason=f"Queue backlog cleared (occ={queue_occupancy} <= {self.config.recovery_queue_low_threshold}) and 0 misses",
                    queue_occ=queue_occupancy,
                    delay=avg_delay,
                    drops=dropped_blocks,
                )
                logger.info(f"[ADAPTIVE CONTROLLER] Overload resolved. Entering RECOVERY (occ={queue_occupancy})")
            return self.state

        # ----------------------------------------------------------------------
        # 3. In RECOVERY: Verify sustained stability before returning to NORMAL
        # ----------------------------------------------------------------------
        if self.state == ControllerState.RECOVERY:
            self.recovery_stable_hops_count += 1
            if self.recovery_stable_hops_count >= self.config.recovery_stable_hops:
                self.state = ControllerState.NORMAL
                self.total_state_transitions += 1
                self._record_decision(
                    prev_state=prev_state,
                    new_state=self.state,
                    action=AdaptationAction.RESTORE_NORMAL,
                    reason=f"Sustained stability confirmed for {self.recovery_stable_hops_count} hops",
                    queue_occ=queue_occupancy,
                    delay=avg_delay,
                    drops=dropped_blocks,
                )
                logger.info("[ADAPTIVE CONTROLLER] Recovery complete. Returned to NORMAL.")
            return self.state

        # ----------------------------------------------------------------------
        # 4. NORMAL <-> IDLE Transitions (Silence Awareness)
        # ----------------------------------------------------------------------
        is_silent = (vad_state == VADState.SILENCE)
        if is_silent:
            self.silence_hops_count += 1
            if self.silence_hops_count >= self.config.idle_silence_hops_threshold and self.state == ControllerState.NORMAL:
                self.state = ControllerState.IDLE
                self.total_state_transitions += 1
                self._record_decision(
                    prev_state=prev_state,
                    new_state=self.state,
                    action=AdaptationAction.ENTER_IDLE,
                    reason=f"Sustained silence for {self.silence_hops_count} hops ({self.silence_hops_count * 20}ms)",
                    queue_occ=queue_occupancy,
                    delay=avg_delay,
                    drops=dropped_blocks,
                )
                logger.debug(f"[ADAPTIVE CONTROLLER] Entering IDLE state after {self.silence_hops_count} silent hops.")
        else:
            # Speech detected (ONSET_CANDIDATE, SPEECH_ACTIVE, HANGOVER, UTTERANCE_COMPLETE)
            self.silence_hops_count = 0
            if self.state == ControllerState.IDLE:
                self.state = ControllerState.NORMAL
                self.total_state_transitions += 1
                self._record_decision(
                    prev_state=prev_state,
                    new_state=self.state,
                    action=AdaptationAction.EXIT_IDLE,
                    reason=f"Speech onset detected by VAD (State: {vad_state.name})",
                    queue_occ=queue_occupancy,
                    delay=avg_delay,
                    drops=dropped_blocks,
                )
                logger.info(f"[ADAPTIVE CONTROLLER] Speech onset detected ({vad_state.name}). Returning to NORMAL.")

        return self.state

    def should_infer_sliding(self, hop_index: int) -> bool:
        """
        Determines whether intermediate sliding-window inference should execute for this hop.
        Governed by active controller state.
        """
        if self.state == ControllerState.IDLE:
            # In IDLE: NEVER run unprompted sliding inference on silence
            self.total_inferences_suppressed += 1
            return False

        elif self.state == ControllerState.OVERLOAD:
            # In OVERLOAD: Suppress intermediate sliding inference entirely to shed CPU load!
            # Inference will only execute on confirmed utterance completion.
            self.total_inferences_suppressed += 1
            return False

        elif self.state == ControllerState.RECOVERY:
            # In RECOVERY: Evaluate at relaxed cadence (every 10 hops = 200 ms)
            if hop_index % self.config.recovery_inference_stride == 0:
                return True
            self.total_inferences_suppressed += 1
            return False

        else:  # ControllerState.NORMAL
            # In NORMAL: Evaluate at standard cadence (every 5 hops = 100 ms)
            return (hop_index % self.config.normal_inference_stride == 0)

    def should_infer_completed_utterance(self) -> bool:
        """
        Utterance completion inference is ALWAYS permitted across all states to guarantee
        that no completed speech command is ever dropped or lost.
        """
        return True

    def get_monitoring_interval_s(self) -> float:
        """Modulates telemetry monitoring frequency based on controller state."""
        if self.state == ControllerState.IDLE:
            return 0.500   # 500 ms in IDLE (conserve background cycles)
        elif self.state == ControllerState.OVERLOAD:
            return 1.000   # 1000 ms in OVERLOAD (defer telemetry work)
        elif self.state == ControllerState.RECOVERY:
            return 0.300   # 300 ms in RECOVERY
        else:
            return 0.200   # 200 ms in NORMAL

    def _record_decision(
        self,
        prev_state: ControllerState,
        new_state: ControllerState,
        action: AdaptationAction,
        reason: str,
        queue_occ: int,
        delay: float,
        drops: int,
    ):
        dec = AdaptationDecision(
            decision_index=len(self._decision_history) + 1,
            timestamp=time.perf_counter(),
            previous_state=prev_state,
            new_state=new_state,
            action=action,
            trigger_reason=reason,
            queue_occupancy=queue_occ,
            consecutive_deadline_misses=self.consecutive_deadline_misses,
            recent_queue_delay_ms=delay,
            dropped_blocks=drops,
        )
        self._decision_history.append(dec)

    def get_history(self) -> List[AdaptationDecision]:
        return list(self._decision_history)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns comprehensive status dictionary for reporting."""
        return {
            "current_state": self.state.value,
            "silence_hops_count": self.silence_hops_count,
            "consecutive_deadline_misses": self.consecutive_deadline_misses,
            "recovery_stable_hops_count": self.recovery_stable_hops_count,
            "total_state_transitions": self.total_state_transitions,
            "total_overload_events": self.total_overload_events,
            "total_inferences_suppressed": self.total_inferences_suppressed,
            "total_stale_jobs_pruned": self.total_stale_jobs_pruned,
            "mean_recent_queue_delay_ms": round(self.mean_recent_queue_delay_ms, 3),
            "decisions_recorded": len(self._decision_history),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Alias for get_diagnostics for consistent telemetry interface."""
        return self.get_diagnostics()
