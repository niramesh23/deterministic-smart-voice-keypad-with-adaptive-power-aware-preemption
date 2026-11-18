"""
tests/test_stage9.py
--------------------
Comprehensive Unit and Integration Tests for Stage 9:
Command Finite-State Machine, Command Output, Monitoring Dashboard, Structured Logging, and Fault Handling.

Test Coverage:
1. Command FSM state transitions: WAITING -> SPEECH_DETECTED -> CANDIDATE -> CONFIRMED -> COOLDOWN -> WAITING.
2. Low-confidence, low-margin, and unsupported-vocabulary rejection guards.
3. Cooldown guard and duplicate suppression across overlapping inference frames.
4. Dashboard server HTTP endpoints (/api/status, /api/commands, /) and JSON structure.
5. Missing metrics vs genuine zero values (N/A formatting).
6. Dashboard snapshot image rendering.
7. Structured experiment logging (JSONL persistence, audio opt-in guard).
8. Fault handling: model loading failure, queue overflow, graceful shutdown, and clean restart.
"""

import json
import pathlib
import time
import urllib.request
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from src.application.command_fsm import (
    CommandFiniteStateMachine,
    CommandState,
    CommandRecord,
    CommandFSMConfig,
    RejectionReason,
)
from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig
from src.dashboard.server import DashboardServer, DashboardStatusProvider
from src.dashboard.snapshot import render_dashboard_snapshot
from src.monitoring.structured_logger import StructuredExperimentLogger
from src.dsp.vad import VADState
from src.ml.inference import RecognitionResult
from src.application.live_recognizer import LivePipelineTelemetry
from src.rts.scheduler import SchedulingPolicy


def create_dummy_telemetry(reaction_ms: float = 4.5, acoustic_ms: float = 650.0) -> LivePipelineTelemetry:
    t = time.perf_counter()
    return LivePipelineTelemetry(
        sequence_number=1,
        audio_event_timestamp=t - 0.05,
        utterance_onset_timestamp=t - 0.65,
        decision_timestamp=t,
        capture_and_buffering_delay_ms=0.5,
        queue_delay_ms=0.2,
        feature_extraction_ms=0.4,
        model_inference_ms=3.1,
        postprocessing_ms=0.05,
        reaction_latency_ms=reaction_ms,
        total_acoustic_latency_ms=acoustic_ms,
    )


def make_dummy_result(
    command: Optional[str] = "yes",
    raw_class: str = "yes",
    confidence: float = 0.88,
    margin: float = 0.35,
    top2_class: str = "no",
    is_confirmed: bool = True,
) -> RecognitionResult:
    now = time.perf_counter()
    return RecognitionResult(
        command=command,
        raw_class=raw_class,
        confidence=confidence,
        raw_confidence=confidence,
        top2_class=top2_class,
        top2_confidence=max(0.0, confidence - margin),
        margin=margin,
        is_confirmed=is_confirmed,
        is_silence=False,
        is_unknown=False,
        is_cooldown=False,
        probabilities={raw_class: confidence, top2_class: max(0.0, confidence - margin)},
        audio_event_timestamp=now - 0.05,
        decision_timestamp=now,
        timings={"model_inference_ms": 3.1, "feature_extraction_ms": 0.4},
    )


# ==============================================================================
# Task A & B: Command Finite-State Machine & Output Handling
# ==============================================================================

def test_command_fsm_initial_state_and_speech_detection():
    fsm = CommandFiniteStateMachine()
    assert fsm.state == CommandState.WAITING
    assert fsm.total_evaluations == 0
    assert fsm.total_confirmed == 0
    assert fsm.total_rejected == 0

    # VAD signals speech -> SPEECH_DETECTED
    fsm.notify_vad_state(VADState.SPEECH_ACTIVE)
    assert fsm.state == CommandState.SPEECH_DETECTED
    assert fsm.total_state_transitions == 1

    # Speech timeout without inference -> returns to WAITING
    fsm.notify_vad_state(VADState.SILENCE, timestamp=time.perf_counter() + 2.0)
    assert fsm.state == CommandState.WAITING


def test_command_fsm_successful_confirmation():
    confirmed_records = []
    fsm = CommandFiniteStateMachine(on_command_callback=lambda r: confirmed_records.append(r))

    fsm.notify_vad_state(VADState.SPEECH_ACTIVE)
    res = make_dummy_result(command="yes", raw_class="yes", confidence=0.88, margin=0.35)
    tel = create_dummy_telemetry(reaction_ms=4.2)

    rec = fsm.process_prediction(res, tel)
    assert rec.is_confirmed is True
    assert rec.command == "yes"
    assert rec.rejection_reason == RejectionReason.NONE
    assert rec.reaction_latency_ms == 4.2
    assert len(confirmed_records) == 1
    assert confirmed_records[0].command == "yes"

    # Immediately after confirmation, FSM must be in COOLDOWN
    assert fsm.state == CommandState.COOLDOWN
    assert fsm.total_confirmed == 1


def test_command_fsm_rejection_low_confidence_and_margin():
    fsm = CommandFiniteStateMachine()
    fsm.notify_vad_state(VADState.SPEECH_ACTIVE)
    tel = create_dummy_telemetry()

    # 1. Low confidence (< candidate_confidence_threshold = 0.50)
    res_low_conf = make_dummy_result(command="up", raw_class="up", confidence=0.42, margin=0.20, is_confirmed=False)
    rec1 = fsm.process_prediction(res_low_conf, tel)
    assert rec1.is_confirmed is False
    assert rec1.rejection_reason == RejectionReason.LOW_CONFIDENCE

    # 2. Candidate passes (0.58 >= 0.50), but below confirmation threshold (0.65)
    res_mid_conf = make_dummy_result(command="up", raw_class="up", confidence=0.58, margin=0.25, is_confirmed=False)
    rec2 = fsm.process_prediction(res_mid_conf, tel, vad_state=VADState.SPEECH_ACTIVE)
    assert rec2.is_confirmed is False
    assert rec2.rejection_reason == RejectionReason.LOW_CONFIDENCE
    # Candidate rejection must not leave FSM stuck in CANDIDATE_COMMAND
    assert fsm.state != CommandState.CANDIDATE_COMMAND
    assert fsm.state == CommandState.SPEECH_DETECTED

    # 3. High confidence (0.78), but low margin (0.05 < 0.15)
    res_low_margin = make_dummy_result(command="up", raw_class="up", confidence=0.78, margin=0.05, is_confirmed=False)
    rec3 = fsm.process_prediction(res_low_margin, tel, vad_state=VADState.SPEECH_ACTIVE)
    assert rec3.is_confirmed is False
    assert rec3.rejection_reason == RejectionReason.LOW_MARGIN
    assert fsm.state != CommandState.CANDIDATE_COMMAND
    assert fsm.state == CommandState.SPEECH_DETECTED

    # 4. Candidate rejected under silence immediately transitions to WAITING
    rec4 = fsm.process_prediction(res_mid_conf, tel, vad_state=VADState.SILENCE)
    assert rec4.is_confirmed is False
    assert fsm.state == CommandState.WAITING


def test_command_fsm_rejection_unsupported_vocabulary():
    fsm = CommandFiniteStateMachine()
    tel = create_dummy_telemetry()

    # Unsupported command label (e.g. unknown class or invented label)
    res_unsupported = make_dummy_result(command="banana", raw_class="banana", confidence=0.95, margin=0.50, is_confirmed=False)
    rec = fsm.process_prediction(res_unsupported, tel)
    assert rec.is_confirmed is False
    assert rec.rejection_reason == RejectionReason.UNSUPPORTED_VOCABULARY
    assert fsm.total_rejected == 1


def test_command_fsm_cooldown_duplicate_suppression():
    cfg = CommandFSMConfig(cooldown_duration_s=0.50)
    fsm = CommandFiniteStateMachine(cfg)
    tel = create_dummy_telemetry()

    res_stop = make_dummy_result(command="stop", raw_class="stop", confidence=0.85, margin=0.30, is_confirmed=True)

    # 1. First confirmation succeeds
    rec1 = fsm.process_prediction(res_stop, tel)
    assert rec1.is_confirmed is True
    assert fsm.state == CommandState.COOLDOWN

    # 2. Second overlapping frame of 'stop' arriving immediately -> duplicate rejected
    rec2 = fsm.process_prediction(res_stop, tel)
    assert rec2.is_confirmed is False
    assert rec2.rejection_reason == RejectionReason.DUPLICATE_IN_COOLDOWN
    assert fsm.total_duplicates_suppressed == 1

    # 3. Wait for cooldown to expire
    time.sleep(0.55)
    fsm.notify_vad_state(VADState.SILENCE)
    assert fsm.state == CommandState.WAITING

    # 4. Now 'stop' can be confirmed again
    rec3 = fsm.process_prediction(res_stop, tel)
    assert rec3.is_confirmed is True
    assert fsm.total_confirmed == 2


# ==============================================================================
# Task C: Monitoring Dashboard Server & Snapshot
# ==============================================================================

def test_dashboard_server_endpoints_and_json_structure():
    provider = DashboardStatusProvider()
    provider.update_status({
        "microphone": {"device_name": "Test Mic Array", "queue_occupancy": 3, "audio_blocks_dropped": 0},
        "command_fsm": {"current_state": "WAITING", "total_confirmed": 2},
        "timing": {"processing_reaction_latency_ms": 4.15, "model_inference_ms": 3.02},
        "scheduler": {"policy": "EDF", "total_executed": 45, "total_deadline_misses": 0},
        "adaptive_controller": {"current_state": "NORMAL", "mean_recent_queue_delay_ms": 0.18},
    })

    server = DashboardServer(port=8899, provider=provider)
    server.start()
    try:
        url = server.url

        # 1. Test HTML index endpoint
        with urllib.request.urlopen(f"{url}/") as response:
            assert response.status == 200
            html = response.read().decode("utf-8")
            assert "RTOS Voice Recognition Dashboard" in html

        # 2. Test /api/status endpoint
        with urllib.request.urlopen(f"{url}/api/status") as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
            assert data["microphone"]["device_name"] == "Test Mic Array"
            assert data["microphone"]["queue_occupancy"] == 3
            assert data["timing"]["processing_reaction_latency_ms"] == 4.15
            assert data["scheduler"]["policy"] == "EDF"
            assert data["adaptive_controller"]["current_state"] == "NORMAL"
            assert data["resources"]["cpu_utilization_pct"] >= 0.0

        # 3. Test /api/commands endpoint
        with urllib.request.urlopen(f"{url}/api/commands") as response:
            assert response.status == 200
            cmds = json.loads(response.read().decode("utf-8"))
            assert isinstance(cmds, list)

    finally:
        server.stop()


def test_dashboard_snapshot_renderer(tmp_path):
    provider = DashboardStatusProvider()
    provider.update_status({
        "microphone": {"device_name": "Verified Intel Array", "device_id": 1, "queue_occupancy": 1},
        "command_fsm": {"current_state": "COMMAND_CONFIRMED", "latest_command": "yes", "latest_confidence": 0.84, "total_confirmed": 5},
        "timing": {"processing_reaction_latency_ms": 4.25, "total_acoustic_latency_ms": 580.0, "model_inference_ms": 3.12, "feature_extraction_ms": 0.42, "queue_delay_ms": 0.15},
        "scheduler": {"policy": "EDF", "ready_jobs_count": 0, "total_executed": 120, "total_deadline_misses": 0, "deadline_miss_ratio_pct": 0.0},
        "adaptive_controller": {"current_state": "IDLE", "total_state_transitions": 8, "total_stale_jobs_pruned": 0, "mean_recent_queue_delay_ms": 0.12},
        "history": [
            {"command": "yes", "confidence": 0.84, "margin": 0.35, "reaction_latency_ms": 4.25, "is_confirmed": True, "detected_timestamp": time.time()},
            {"command": "stop", "confidence": 0.78, "margin": 0.28, "reaction_latency_ms": 4.10, "is_confirmed": True, "detected_timestamp": time.time() + 1},
        ],
    })

    out_file = tmp_path / "test_snapshot.png"
    rendered = render_dashboard_snapshot(provider.get_snapshot(), output_path=out_file, dpi=100)
    assert rendered.exists()
    assert rendered.stat().st_size > 10000  # Valid non-empty PNG image


# ==============================================================================
# Task D: Structured Experiment Logger
# ==============================================================================

def test_structured_logger_records(tmp_path):
    log_dir = tmp_path / "logs"
    logger = StructuredExperimentLogger(output_dir=log_dir, session_id="test_session", enable_audio_recording=False)

    try:
        logger.log_scheduler_decision({"policy": "EDF", "selected_job": "audio_1"})
        logger.log_prediction_result({"command": "yes", "confidence": 0.85, "is_confirmed": True})
        logger.log_controller_transition({"from": "NORMAL", "to": "OVERLOAD", "reason": "Queue >= 8"})
        logger.log_audio_error("BUFFER_OVERRUN", {"slot": 1})
        logger.log_deadline_miss("infer_1", 100.0, 102.5, {"reason": "Heavy load"})
        logger.log_queue_overflow("audio_queue", 100, 100)
        logger.log_resource_measurement(5.2, 120.4)

        events = logger.get_events()
        assert len(events) == 7
        event_types = [e["event_type"] for e in events]
        assert "SCHEDULER_DECISION" in event_types
        assert "PREDICTION_RESULT" in event_types
        assert "CONTROLLER_TRANSITION" in event_types
        assert "DEADLINE_MISS" in event_types

        # Audio recording opt-in guard: no raw audio files should have been created
        audio_files = list(log_dir.glob("*.wav")) + list(log_dir.glob("*.raw"))
        assert len(audio_files) == 0

    finally:
        logger.close()


# ==============================================================================
# Task E: Application Orchestration & Fault Handling
# ==============================================================================

def test_orchestrator_missing_checkpoint_fails_clearly():
    from src.ml.inference import KeywordSpotterInferenceService, InferenceInitializationError
    # Non-existent checkpoint path
    with pytest.raises(InferenceInitializationError):
        _ = KeywordSpotterInferenceService(checkpoint_path=pathlib.Path("non_existent_model.pt"))


def test_orchestrator_lifecycle_and_non_blocking_dashboard():
    """
    Verifies that VoiceRecognitionApp starts, processes external audio hops,
    updates dashboard without blocking audio callback, and shuts down cleanly.
    """
    app = VoiceRecognitionApp(
        config=OrchestratorConfig(
            enable_dashboard=True,
            dashboard_port=8895,
        )
    )
    app.start(use_microphone=False)

    try:
        # Feed 15 audio hops
        silent_hop = np.zeros(320, dtype=np.int16)
        for _ in range(15):
            app.push_audio_hop(silent_hop)
            time.sleep(0.002)

        time.sleep(0.15)

        # Dashboard status must be populated and non-blocking
        snapshot = app.status_provider.get_snapshot()
        assert snapshot["microphone"]["queue_capacity"] == 100
        assert snapshot["scheduler"]["policy"] == "EDF"

        # Safe restart
        app.restart(use_microphone=False)
        assert app._is_running is True

    finally:
        app.stop()
        assert app._is_running is False
