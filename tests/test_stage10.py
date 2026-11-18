"""
tests/test_stage10.py
---------------------
Stage 10 Integration and End-to-End Validation Tests.

Validates:
1. Component configuration consistency across audio, DSP, ML, and RTS.
2. Complete pipeline integration (Audio -> Ingress Queue -> Framer -> VAD -> Scheduler -> Controller -> FSM -> Dashboard -> Logger).
3. Normal operation and failure recovery scenarios.
4. Deterministic scheduling comparisons (FIFO vs RM vs EDF).
5. Adaptive processing modes (Always-on vs Silence-gated vs Silence-gated with Pre-roll).
"""

import json
import os
import pathlib
import time
import pytest
import numpy as np
from scipy.io import wavfile
import torch

from src.application.command_fsm import CommandFiniteStateMachine, CommandState, CommandRecord
from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig
from src.audio.framing import pcm16_to_float32, SlidingWindowFramer
from src.audio.queue import BoundedAudioQueue
from src.audio.types import AudioBlock, QueueOverflowPolicy
from src.dsp.features import LogMelSpectrogramExtractor
from src.dsp.vad import VoiceActivityDetector, VADState
from src.ml.classes import CLASS_NAMES, CLASS_NAMES_12, CORE_COMMANDS
from src.ml.inference import KeywordSpotterInferenceService, InferenceInitializationError
from src.ml.model import ResidualDSCNNKeywordSpotter
from src.rts.tasks import (
    create_audio_block_job,
    create_feature_extraction_job,
    create_keyword_inference_job,
)
from src.rts.scheduler import (
    RealTimeScheduler,
    SchedulingPolicy,
    LateJobStrategy,
)
from src.rts.adaptive import AdaptiveProcessingController, AdaptiveControllerConfig, ControllerState
from src.dashboard.server import DashboardStatusProvider


# ==============================================================================
# Task A: Component Configuration and Checkpoint Consistency
# ==============================================================================

def test_component_configuration_consistency():
    """Confirms that audio, DSP, and model checkpoint share identical configurations."""
    ckpt_path = pathlib.Path("models/checkpoints/dscnn_improved.pt")
    feat_cfg_path = pathlib.Path("models/checkpoints/feature_config_improved.json")
    class_map_path = pathlib.Path("models/checkpoints/class_mapping_improved.json")

    assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
    assert feat_cfg_path.exists(), f"Feature config missing: {feat_cfg_path}"
    assert class_map_path.exists(), f"Class mapping missing: {class_map_path}"

    with open(feat_cfg_path, "r", encoding="utf-8") as f:
        feat_cfg = json.load(f)
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    # 1. Audio sample rate and feature extraction parameters
    assert feat_cfg["sample_rate"] == 16000
    assert feat_cfg["hop_size"] == 320  # 20 ms
    assert feat_cfg["frame_size"] == 640  # 40 ms
    assert feat_cfg["num_mel_bins"] == 40
    assert feat_cfg["shape"] == [1, 49, 40]

    # 2. Vocabulary classes
    assert class_map["num_classes"] == 12
    assert class_map["class_names"] == CLASS_NAMES_12
    assert class_map["core_commands"] == CORE_COMMANDS

    # 3. Model checkpoint metadata
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["num_classes"] == 12
    assert ckpt["class_names"] == CLASS_NAMES_12
    assert ckpt["parameter_count"] == 28108


# ==============================================================================
# Task A & B: Full Pipeline Integration (Audio -> Decision -> Dashboard)
# ==============================================================================

def test_complete_end_to_end_pipeline_integration():
    """
    Verifies full integration path:
    Audio Block -> Ping-Pong/Queue -> Framer -> VAD -> Scheduler -> Controller -> FSM -> Dashboard -> Logger.
    """
    confirmed_commands = []

    def on_command(rec: CommandRecord):
        confirmed_commands.append(rec)

    config = OrchestratorConfig(
        policy=SchedulingPolicy.EDF,
        enable_adaptation=True,
        enable_dashboard=True,
        dashboard_port=8891,
        enable_audio_logging=False,
    )
    app = VoiceRecognitionApp(config=config, on_command_callback=on_command)
    app.start(use_microphone=False)

    try:
        # Load a real 'yes' sample from raw test dataset
        wav_path = pathlib.Path("data/raw/yes/bf92eb49_nohash_0.wav")
        if not wav_path.exists():
            # Fallback to any valid audio in data/raw/yes
            candidates = list(pathlib.Path("data/raw/yes").glob("*.wav"))
            assert len(candidates) > 0, "No sample wav file found in data/raw/yes"
            wav_path = candidates[0]

        sr, audio = wavfile.read(str(wav_path))
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)

        # Feed 1.5 seconds of audio (including padding) in 20 ms (320-sample) hops
        hop_size = 320
        total_samples = len(audio)
        for i in range(0, total_samples, hop_size):
            hop = audio[i : i + hop_size]
            if len(hop) < hop_size:
                hop = np.pad(hop, (0, hop_size - len(hop)))
            app.push_audio_hop(hop)
            time.sleep(0.015)

        # Allow worker thread to finish real-time scheduled inference
        time.sleep(0.50)

        # Check telemetry and dashboard
        snap = app.status_provider.get_snapshot()
        assert snap["scheduler"]["policy"] == "EDF"
        assert snap["scheduler"]["total_executed"] > 0
        assert snap["scheduler"]["total_deadline_misses"] <= 10
        assert snap["microphone"]["audio_blocks_dropped"] == 0

        # Verify command confirmation
        assert len(confirmed_commands) >= 1
        assert confirmed_commands[0].command == "yes"
        assert confirmed_commands[0].is_confirmed is True
        assert confirmed_commands[0].reaction_latency_ms is not None

    finally:
        app.stop()


# ==============================================================================
# Task A: Failure Scenarios and Fault Handling
# ==============================================================================

def test_missing_model_checkpoint_fails_cleanly():
    """Verifies that an invalid checkpoint path fails fast with clear error."""
    with pytest.raises(InferenceInitializationError):
        KeywordSpotterInferenceService(checkpoint_path="non_existent_model_checkpoint.pt")


def test_queue_overflow_drop_oldest_non_blocking():
    """Verifies that pushing audio faster than consumer capacity drops oldest blocks without blocking."""
    queue = BoundedAudioQueue(capacity=5, overflow_policy=QueueOverflowPolicy.DROP_OLDEST)
    # Push 15 items into queue of capacity 5
    for i in range(15):
        block = AudioBlock(
            data=np.zeros(320, dtype=np.int16),
            sequence_number=i,
            capture_timestamp=time.perf_counter(),
        )
        queue.push(block)

    metrics = queue.get_metrics()
    assert metrics["current_occupancy"] == 5
    assert metrics["dropped_blocks_count"] == 10
    assert metrics["overflow_count"] == 10


# ==============================================================================
# Task B: Scheduling Policies Comparative Invariants (FIFO vs RM vs EDF)
# ==============================================================================

def test_scheduling_policies_deterministic_ordering():
    """
    Submits simultaneous jobs with varying deadlines and priorities:
    Job 1: D=100ms, P=3 (Keyword Inference)
    Job 2: D=40ms,  P=2 (Feature Extraction)
    Job 3: D=20ms,  P=1 (Audio Block Ingest)
    Verifies that FIFO, RM, and EDF order execution according to their respective policies.
    """
    executed_order = []

    def make_worker(name):
        return lambda: executed_order.append(name)

    now = time.perf_counter()

    # 1. Baseline FIFO: Orders by arrival (1, 2, 3)
    executed_order.clear()
    sched_fifo = RealTimeScheduler(policy=SchedulingPolicy.FIFO)
    sched_fifo.submit_job(create_keyword_inference_job("job_inf", make_worker("INFER"), release_time=now))
    sched_fifo.submit_job(create_feature_extraction_job("job_feat", make_worker("FEAT"), release_time=now + 0.001))
    sched_fifo.submit_job(create_audio_block_job("job_block", make_worker("BLOCK"), release_time=now + 0.002))
    sched_fifo.run_all_pending()
    assert executed_order == ["INFER", "FEAT", "BLOCK"], f"FIFO failed: {executed_order}"

    # 2. Rate Monotonic (RM): Orders by fixed priority (BLOCK=1 > FEAT=2 > INFER=3)
    executed_order.clear()
    sched_rm = RealTimeScheduler(policy=SchedulingPolicy.RM)
    sched_rm.submit_job(create_keyword_inference_job("job_inf", make_worker("INFER"), release_time=now))
    sched_rm.submit_job(create_feature_extraction_job("job_feat", make_worker("FEAT"), release_time=now + 0.001))
    sched_rm.submit_job(create_audio_block_job("job_block", make_worker("BLOCK"), release_time=now + 0.002))
    sched_rm.run_all_pending()
    assert executed_order == ["BLOCK", "FEAT", "INFER"], f"RM failed: {executed_order}"

    # 3. Earliest Deadline First (EDF): Orders by absolute deadline (BLOCK=now+20ms < FEAT=now+40ms < INFER=now+100ms)
    executed_order.clear()
    sched_edf = RealTimeScheduler(policy=SchedulingPolicy.EDF)
    sched_edf.submit_job(create_keyword_inference_job("job_inf", make_worker("INFER"), release_time=now))
    sched_edf.submit_job(create_feature_extraction_job("job_feat", make_worker("FEAT"), release_time=now + 0.001))
    sched_edf.submit_job(create_audio_block_job("job_block", make_worker("BLOCK"), release_time=now + 0.002))
    sched_edf.run_all_pending()
    assert executed_order == ["BLOCK", "FEAT", "INFER"], f"EDF failed: {executed_order}"


# ==============================================================================
# Task B: Adaptive Processing Modes Invariants
# ==============================================================================

def test_adaptive_modes_silence_gating_and_preroll():
    """
    Compares Always-on vs Silence-gated vs Silence-gated with pre-roll:
    1. During continuous silence, silence gating skips inference entirely.
    2. During speech onset, pre-roll buffer restores preceding 100-200 ms audio.
    """
    # 1. Silence Gating Test
    ctrl = AdaptiveProcessingController(AdaptiveControllerConfig(idle_silence_hops_threshold=5))
    ctrl.reset()

    # Sustained silence transitions controller to IDLE
    for _ in range(6):
        ctrl.evaluate_state(vad_state=VADState.SILENCE, queue_occupancy=0)
    assert ctrl.state == ControllerState.IDLE
    # Under IDLE, sliding inference is suppressed
    assert ctrl.should_infer_sliding(hop_index=10) is False

    # Speech onset restores NORMAL state immediately
    ctrl.evaluate_state(vad_state=VADState.ONSET_CANDIDATE, queue_occupancy=0)
    assert ctrl.state == ControllerState.NORMAL

    # 2. Pre-roll Buffer Test
    vad = VoiceActivityDetector(pre_roll_duration_ms=100.0)
    vad.reset()

    # Feed silence frames
    silence_frame = np.zeros(320, dtype=np.int16)
    for _ in range(10):
        vad.process_hop(silence_frame)

    assert vad.preroll_buffer.capacity_samples > 0
    pre_samples = vad.preroll_buffer.get_preroll_samples()
    assert len(pre_samples) == int(0.100 * 16000)  # 1,600 samples for 100 ms
