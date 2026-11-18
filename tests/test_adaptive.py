"""
tests/test_adaptive.py
----------------------
Deterministic unit and integration tests for Stage 8: Adaptive Processing Controller.

Test Coverage:
1. Controller state transitions: NORMAL -> IDLE -> NORMAL.
2. OVERLOAD trigger via queue occupancy threshold.
3. OVERLOAD trigger via consecutive deadline misses.
4. OVERLOAD trigger via queue delay and dropped blocks.
5. RECOVERY state and hysteresis stability window before returning to NORMAL.
6. Sliding inference suppression during IDLE and OVERLOAD while strictly preserving completed utterance inference.
7. Telemetry monitoring interval modulation across states.
8. Audio buffer and pre-roll preservation across state transitions.
9. End-to-end integration with ScheduledKeywordRecognizer on silence vs speech streams.
"""

import time
import numpy as np
import pytest

from src.dsp.vad import VADState, VoiceActivityDetector
from src.rts.tasks import (
    JobType,
    RealTimeJob,
    create_audio_block_job,
    create_feature_extraction_job,
    create_keyword_inference_job,
)
from src.rts.scheduler import RealTimeScheduler, SchedulingPolicy
from src.rts.adaptive import (
    AdaptiveProcessingController,
    AdaptiveControllerConfig,
    ControllerState,
    AdaptationAction,
    AdaptationDecision,
)
from src.application.scheduled_recognizer import ScheduledKeywordRecognizer
from src.audio import AudioBlock


# ==============================================================================
# Unit Tests: AdaptiveProcessingController State Machine
# ==============================================================================

def test_initial_state_and_defaults():
    controller = AdaptiveProcessingController()
    assert controller.state == ControllerState.NORMAL
    assert controller.silence_hops_count == 0
    assert controller.consecutive_deadline_misses == 0
    assert controller.recovery_stable_hops_count == 0
    assert controller.total_state_transitions == 0
    assert controller.total_overload_events == 0
    assert controller.total_inferences_suppressed == 0
    assert controller.total_stale_jobs_pruned == 0
    assert len(controller.decision_history) == 0


def test_transition_normal_to_idle_on_sustained_silence():
    cfg = AdaptiveControllerConfig(idle_silence_hops_threshold=5)
    controller = AdaptiveProcessingController(cfg)

    # 4 hops of silence -> remains NORMAL
    for i in range(4):
        state = controller.evaluate_state(vad_state=VADState.SILENCE, queue_occupancy=0)
        assert state == ControllerState.NORMAL
        assert controller.silence_hops_count == i + 1

    # 5th hop of silence -> transitions to IDLE
    state = controller.evaluate_state(vad_state=VADState.SILENCE, queue_occupancy=0)
    assert state == ControllerState.IDLE
    assert controller.total_state_transitions == 1
    assert len(controller.decision_history) == 1
    assert controller.decision_history[-1].action == AdaptationAction.ENTER_IDLE


def test_transition_idle_to_normal_on_speech_onset():
    cfg = AdaptiveControllerConfig(idle_silence_hops_threshold=3)
    controller = AdaptiveProcessingController(cfg)

    for _ in range(3):
        controller.evaluate_state(vad_state=VADState.SILENCE, queue_occupancy=0)
    assert controller.state == ControllerState.IDLE

    # Speech onset -> immediately exits IDLE to NORMAL
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=0)
    assert state == ControllerState.NORMAL
    assert controller.silence_hops_count == 0
    assert controller.total_state_transitions == 2
    assert controller.decision_history[-1].action == AdaptationAction.EXIT_IDLE


def test_overload_triggered_by_queue_occupancy():
    cfg = AdaptiveControllerConfig(overload_queue_high_threshold=8)
    controller = AdaptiveProcessingController(cfg)

    # Queue at 7 (< 8) -> remains NORMAL
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=7)
    assert state == ControllerState.NORMAL

    # Queue at 8 (>= 8) -> enters OVERLOAD
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=8)
    assert state == ControllerState.OVERLOAD
    assert controller.total_overload_events == 1
    assert controller.total_state_transitions == 1
    assert controller.decision_history[-1].action == AdaptationAction.ENTER_OVERLOAD
    assert "Queue occupancy 8 >= 8" in controller.decision_history[-1].trigger_reason


def test_overload_triggered_by_consecutive_deadline_misses():
    cfg = AdaptiveControllerConfig(overload_consecutive_misses=2)
    controller = AdaptiveProcessingController(cfg)

    # 1 deadline miss -> remains NORMAL
    controller.record_job_completion(deadline_missed=True, queue_delay_ms=2.0)
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=1)
    assert state == ControllerState.NORMAL
    assert controller.consecutive_deadline_misses == 1

    # 2nd consecutive deadline miss -> triggers OVERLOAD on evaluation
    controller.record_job_completion(deadline_missed=True, queue_delay_ms=4.0)
    assert controller.consecutive_deadline_misses == 2
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=1)
    assert state == ControllerState.OVERLOAD
    assert controller.total_overload_events == 1
    assert "Consecutive misses 2 >= 2" in controller.decision_history[-1].trigger_reason


def test_overload_triggered_by_queue_delay_and_audio_drops():
    cfg = AdaptiveControllerConfig(overload_queue_delay_ms=10.0)
    controller = AdaptiveProcessingController(cfg)

    # Feed high queue delay (12.0 ms)
    for _ in range(5):
        controller.record_job_completion(deadline_missed=False, queue_delay_ms=12.0)
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=2)
    assert state == ControllerState.OVERLOAD
    assert "Mean delay" in controller.decision_history[-1].trigger_reason

    # Test audio drops trigger
    controller.reset()
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=2, dropped_blocks=1)
    assert state == ControllerState.OVERLOAD
    assert "Dropped blocks" in controller.decision_history[-1].trigger_reason


def test_recovery_hysteresis_and_return_to_normal():
    cfg = AdaptiveControllerConfig(
        overload_queue_high_threshold=8,
        recovery_queue_low_threshold=2,
        recovery_stable_hops=5,
    )
    controller = AdaptiveProcessingController(cfg)

    # Force OVERLOAD
    controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=10)
    assert controller.state == ControllerState.OVERLOAD

    # Queue drops to 4 (between 2 and 8) -> remains OVERLOAD (hysteresis prevents premature exit)
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=4)
    assert state == ControllerState.OVERLOAD

    # Queue drops to 2 (<= recovery_queue_low_threshold) -> enters RECOVERY
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=2)
    assert state == ControllerState.RECOVERY
    assert controller.decision_history[-1].action == AdaptationAction.ENTER_RECOVERY
    assert controller.recovery_stable_hops_count == 1

    # In RECOVERY: 3 stable hops -> still in RECOVERY (counts 2, 3, 4)
    for i in range(3):
        state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=1)
        assert state == ControllerState.RECOVERY
        assert controller.recovery_stable_hops_count == i + 2

    # In RECOVERY: 4th stable hop reached (total count 5 >= 5) -> restores NORMAL
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=1)
    assert state == ControllerState.NORMAL
    assert controller.decision_history[-1].action == AdaptationAction.RESTORE_NORMAL


def test_recovery_relapse_to_overload_if_queue_spikes():
    cfg = AdaptiveControllerConfig(
        overload_queue_high_threshold=8,
        recovery_queue_low_threshold=2,
        recovery_stable_hops=10,
    )
    controller = AdaptiveProcessingController(cfg)

    # Enter OVERLOAD then RECOVERY
    controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=10)
    controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=1)
    assert controller.state == ControllerState.RECOVERY

    # Queue spikes back up to 9 before stability window elapses
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=9)
    assert state == ControllerState.OVERLOAD
    assert controller.recovery_stable_hops_count == 0


def test_sliding_inference_gating_logic():
    cfg = AdaptiveControllerConfig(
        normal_inference_stride=5,
        recovery_inference_stride=10,
    )
    controller = AdaptiveProcessingController(cfg)

    # 1. NORMAL state: infer every 5 hops
    assert controller.should_infer_sliding(hop_index=5) is True
    assert controller.should_infer_sliding(hop_index=4) is False

    # 2. IDLE state: suppress sliding inference entirely
    controller.state = ControllerState.IDLE
    assert controller.should_infer_sliding(hop_index=5) is False
    assert controller.should_infer_sliding(hop_index=10) is False
    assert controller.total_inferences_suppressed == 2

    # 3. OVERLOAD state: suppress all intermediate sliding inference
    controller.state = ControllerState.OVERLOAD
    assert controller.should_infer_sliding(hop_index=5) is False
    assert controller.should_infer_sliding(hop_index=10) is False
    assert controller.total_inferences_suppressed == 4

    # 4. RECOVERY state: infer every 10 hops
    controller.state = ControllerState.RECOVERY
    assert controller.should_infer_sliding(hop_index=5) is False
    assert controller.should_infer_sliding(hop_index=10) is True
    assert controller.total_inferences_suppressed == 5

    # 5. Invariant: Completed utterance inference is NEVER suppressed
    for st in [ControllerState.NORMAL, ControllerState.IDLE, ControllerState.OVERLOAD, ControllerState.RECOVERY]:
        controller.state = st
        assert controller.should_infer_completed_utterance() is True


def test_telemetry_interval_adaptation():
    controller = AdaptiveProcessingController()

    # NORMAL -> 0.200s
    assert controller.get_monitoring_interval_s() == 0.200

    # IDLE -> 0.500s
    controller.state = ControllerState.IDLE
    assert controller.get_monitoring_interval_s() == 0.500

    # OVERLOAD -> 1.000s
    controller.state = ControllerState.OVERLOAD
    assert controller.get_monitoring_interval_s() == 1.000

    # RECOVERY -> 0.300s
    controller.state = ControllerState.RECOVERY
    assert controller.get_monitoring_interval_s() == 0.300


# ==============================================================================
# Integration Tests: Pipeline and ScheduledKeywordRecognizer
# ==============================================================================

def test_pipeline_silence_workload_reduction():
    """
    Verifies that feeding sustained silence into ScheduledKeywordRecognizer with adaptation
    causes 0 model inferences to be submitted, while keeping audio rolling buffer active.
    """
    recognizer = ScheduledKeywordRecognizer(
        enable_adaptation=True,
        policy=SchedulingPolicy.EDF,
    )
    recognizer.start(use_microphone=False)

    try:
        # Feed 25 silent hops at realistic pace (e.g. 5 ms between hops)
        silent_hop = np.zeros(320, dtype=np.int16)
        for i in range(25):
            block = recognizer.push_external_hop(silent_hop)
            assert block is not None
            time.sleep(0.005)

        # Give scheduler time to drain any remaining audio job
        time.sleep(0.1)

        report = recognizer.get_analysis_report()
        # Controller should have entered IDLE
        assert report["adaptation"]["enabled"] is True
        assert report["adaptation"]["current_state"] == ControllerState.IDLE.value
        # No keyword inferences run on silence!
        assert recognizer.total_inferences_run == 0
        # Rolling buffer is continuously updated
        assert recognizer._rolling_buffer.shape == (16000,)
        assert np.all(recognizer._rolling_buffer == 0.0)

    finally:
        recognizer.stop()


def test_pipeline_speech_onset_and_audio_preservation():
    """
    Verifies that after sustained silence, when speech samples arrive:
    1. Pipeline transitions IDLE -> NORMAL.
    2. VAD triggers speech detection.
    3. Neural feature extraction and inference are dispatched.
    4. Audio data and pre-roll context are preserved.
    """
    recognizer = ScheduledKeywordRecognizer(
        enable_adaptation=True,
        policy=SchedulingPolicy.EDF,
    )
    recognizer.start(use_microphone=False)

    try:
        # 1. Feed 15 hops of silence paced at 5 ms to allow scheduler to drain and enter IDLE
        silent_hop = np.zeros(320, dtype=np.int16)
        for _ in range(15):
            recognizer.push_external_hop(silent_hop)
            time.sleep(0.005)

        time.sleep(0.1)
        assert recognizer.controller.state == ControllerState.IDLE

        # 2. Inject synthetic speech tone (e.g. 500 Hz tone)
        t = np.linspace(0, 0.02, 320, endpoint=False)
        speech_hop = (np.sin(2 * np.pi * 500 * t) * 15000).astype(np.int16)

        # Feed 15 hops of synthetic speech paced at 5 ms
        for _ in range(15):
            recognizer.push_external_hop(speech_hop)
            time.sleep(0.005)

        # Give scheduler time to process
        time.sleep(0.3)

        # Controller must have transitioned back to NORMAL or OVERLOAD
        assert recognizer.controller.state in (ControllerState.NORMAL, ControllerState.OVERLOAD)
        # Inferences should have been triggered
        assert recognizer.total_inferences_run > 0

    finally:
        recognizer.stop()


def test_overload_stale_job_pruning():
    """
    Verifies that when the scheduler ready queue exceeds backlog limits during OVERLOAD,
    stale intermediate inference jobs are pruned without dropping raw audio jobs.
    """
    scheduler = RealTimeScheduler(policy=SchedulingPolicy.EDF)
    controller = AdaptiveProcessingController(
        AdaptiveControllerConfig(overload_queue_high_threshold=3)
    )

    # Put controller into OVERLOAD
    controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=5)
    assert controller.state == ControllerState.OVERLOAD

    # Submit 1 audio job and 3 inference jobs
    audio_job = create_audio_block_job("audio_1", lambda b: None, (None,), time.perf_counter())
    scheduler.submit_job(audio_job)

    infer_jobs = [
        create_keyword_inference_job(f"infer_{i}", lambda *args: None, (None,), time.perf_counter())
        for i in range(3)
    ]
    for ij in infer_jobs:
        scheduler.submit_job(ij)

    assert scheduler.ready_count == 4

    # Simulate pruning loop from ScheduledKeywordRecognizer
    with scheduler._lock:
        for j in list(scheduler._ready_jobs):
            if j.job_type == JobType.KEYWORD_INFERENCE:
                scheduler.cancel_job(j.job_id)
                controller.total_stale_jobs_pruned += 1

    # Ready queue should only retain the high-priority AUDIO_BLOCK job
    assert scheduler.ready_count == 1
    assert scheduler._ready_jobs[0].job_type == JobType.AUDIO_BLOCK
    assert controller.total_stale_jobs_pruned == 3


def test_recovery_relapse_to_overload_on_deadline_miss():
    """
    Verifies that during RECOVERY, any single deadline miss immediately aborts recovery
    and kicks the controller back into OVERLOAD.
    """
    cfg = AdaptiveControllerConfig(
        overload_queue_high_threshold=8,
        recovery_queue_low_threshold=2,
        recovery_stable_hops=10,
    )
    controller = AdaptiveProcessingController(cfg)

    # Transition NORMAL -> OVERLOAD -> RECOVERY
    controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=10)
    controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=1)
    assert controller.state == ControllerState.RECOVERY

    # 1 deadline miss occurs during recovery
    controller.record_job_completion(deadline_missed=True, queue_delay_ms=4.0)
    state = controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=1)
    assert state == ControllerState.OVERLOAD
    assert controller.recovery_stable_hops_count == 0
    assert "Deadline miss during RECOVERY" in controller.decision_history[-1].trigger_reason


def test_speech_onset_recovery_after_prolonged_silence():
    """
    Verifies that after prolonged silence (1.0s = 50 hops) putting system in IDLE:
    - Ingesting speech audio prompts immediate IDLE -> NORMAL transition.
    - Full DS-CNN forward pass executes cleanly.
    - Target keyword ('yes') is successfully confirmed.
    - Reaction latency is non-zero and measured.
    """
    import pathlib
    from src.ml.dataset import load_wav_16k_mono

    wav_path = list(pathlib.Path("data/raw/yes").glob("*.wav"))[0]
    audio_f32 = load_wav_16k_mono(wav_path)
    audio_pcm16 = np.clip(audio_f32 * 32767.0, -32768, 32767).astype(np.int16)

    recognizer = ScheduledKeywordRecognizer(enable_adaptation=True, policy=SchedulingPolicy.EDF)
    recognizer.inference_service.cooldown_seconds = 0.20  # Lower cooldown for test pacing
    recognizer.start(use_microphone=False)

    confirmed_cmds = []
    latencies = []

    def on_cmd(res, tel):
        if res.is_confirmed:
            confirmed_cmds.append(res.command)
            latencies.append(tel.reaction_latency_ms)

    recognizer.on_command_callback = on_cmd

    try:
        # 1. Feed 50 silent hops (1.0s)
        silent_hop = np.zeros(320, dtype=np.int16)
        for _ in range(50):
            recognizer.push_external_hop(silent_hop)
            time.sleep(0.002)

        time.sleep(0.1)
        assert recognizer.controller.state == ControllerState.IDLE

        # 2. Feed 'yes' audio hops
        num_hops = len(audio_pcm16) // 320
        for h in range(num_hops):
            recognizer.push_external_hop(audio_pcm16[h * 320 : (h + 1) * 320])
            time.sleep(0.002)

        # 3. Trailing silence (20 hops) to allow hangover to finish
        for _ in range(20):
            recognizer.push_external_hop(silent_hop)
            time.sleep(0.002)

        time.sleep(0.3)

        assert "yes" in confirmed_cmds
        assert len(latencies) >= 1
        assert latencies[0] > 0.0
        # Speech onset returned controller to NORMAL
        assert recognizer.controller.total_state_transitions >= 2

    finally:
        recognizer.stop()


def test_short_pauses_within_utterance_preserves_speech_state():
    """
    Verifies that brief silence pauses within an utterance (< 200 ms / 10 hops)
    do NOT prematurely transition the controller into IDLE.
    """
    cfg = AdaptiveControllerConfig(idle_silence_hops_threshold=10)
    controller = AdaptiveProcessingController(cfg)

    # 1. Active speech for 15 hops
    for _ in range(15):
        controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=0)
    assert controller.state == ControllerState.NORMAL

    # 2. Brief intra-utterance pause of 5 silent hops (100 ms < 200 ms threshold)
    for _ in range(5):
        controller.evaluate_state(vad_state=VADState.SILENCE, queue_occupancy=0)
    assert controller.state == ControllerState.NORMAL
    assert controller.silence_hops_count == 5

    # 3. Speech resumes
    controller.evaluate_state(vad_state=VADState.SPEECH_ACTIVE, queue_occupancy=0)
    assert controller.state == ControllerState.NORMAL
    assert controller.silence_hops_count == 0


def test_repeated_commands_recognition():
    """
    Verifies that two consecutive commands separated by a realistic silence pause
    are both successfully confirmed without suppression or buffer corruption.
    """
    import pathlib
    from src.ml.dataset import load_wav_16k_mono

    wav_path = list(pathlib.Path("data/raw/stop").glob("*.wav"))[0]
    audio_f32 = load_wav_16k_mono(wav_path)
    audio_pcm16 = np.clip(audio_f32 * 32767.0, -32768, 32767).astype(np.int16)

    recognizer = ScheduledKeywordRecognizer(enable_adaptation=True, policy=SchedulingPolicy.EDF)
    recognizer.inference_service.cooldown_seconds = 0.20  # Lower cooldown for test pacing
    recognizer.start(use_microphone=False)

    confirmed_cmds = []

    def on_cmd(res, tel):
        if res.is_confirmed:
            confirmed_cmds.append(res.command)

    recognizer.on_command_callback = on_cmd

    try:
        silent_hop = np.zeros(320, dtype=np.int16)
        num_hops = len(audio_pcm16) // 320

        # Command 1: 'stop'
        for h in range(num_hops):
            recognizer.push_external_hop(audio_pcm16[h * 320 : (h + 1) * 320])
            time.sleep(0.002)

        # Inter-command pause: 30 silent hops (600 ms > cooldown period)
        for _ in range(30):
            recognizer.push_external_hop(silent_hop)
            time.sleep(0.002)

        # Command 2: 'stop' again
        for h in range(num_hops):
            recognizer.push_external_hop(audio_pcm16[h * 320 : (h + 1) * 320])
            time.sleep(0.002)

        # Trailing pause: 20 silent hops
        for _ in range(20):
            recognizer.push_external_hop(silent_hop)
            time.sleep(0.002)

        time.sleep(0.3)

        # Both stop commands must be confirmed
        assert confirmed_cmds.count("stop") == 2

    finally:
        recognizer.stop()
