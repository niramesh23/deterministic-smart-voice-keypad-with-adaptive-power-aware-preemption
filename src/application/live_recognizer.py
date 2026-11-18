"""
src/application/live_recognizer.py
----------------------------------
Live keyword recognition pipeline connecting:
Microphone / Audio Source -> Ping-Pong Buffer -> Audio Queue -> Framer -> VAD -> Rolling Context -> Feature Extraction -> DS-CNN Inference.

Key Guarantees:
- Model inference runs in a dedicated processing thread, NEVER inside the audio callback.
- Standard 16,000 Hz, mono, 320 hop (20 ms), 640 frame (40 ms), 40 Mel filterbanks.
- Real-time latency tracking across all pipeline stages:
  * Capture and buffering delay
  * Queue latency
  * DSP feature extraction duration
  * Neural network forward pass duration
  * End-to-end reaction and acoustic latency
- Configurable confidence threshold, margin check, probability smoothing, and cooldown.
- Rejection of silence, unknown words, and transient noise.
"""

from dataclasses import dataclass, field
import json
import logging
import os
import pathlib
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.buffers import SoftwarePingPongBuffer
from src.audio.framing import SlidingWindowFramer, pcm16_to_float32
from src.audio.microphone import MicrophoneCapture, get_default_input_device
from src.audio.queue import BoundedAudioQueue
from src.audio.types import AudioBlock, QueueOverflowPolicy
from src.dsp.vad import VADState, UtteranceSegment, VoiceActivityDetector
from src.ml.inference import KeywordSpotterInferenceService, RecognitionResult

logger = logging.getLogger("rtos.live_recognizer")


@dataclass
class LivePipelineTelemetry:
    """
    Comprehensive latency breakdown for a live keyword detection event.
    """
    sequence_number: int
    audio_event_timestamp: float
    utterance_onset_timestamp: Optional[float]
    decision_timestamp: float

    # Subsystem delays in milliseconds
    capture_and_buffering_delay_ms: float
    queue_delay_ms: float
    feature_extraction_ms: float
    model_inference_ms: float
    postprocessing_ms: float

    # End-to-End Latency:
    # 1. Computational reaction latency: time from buffer completion to emitted decision
    reaction_latency_ms: float
    # 2. Total acoustic latency: time from initial speech onset to emitted decision
    total_acoustic_latency_ms: Optional[float]


class LiveKeywordRecognizer:
    """
    Complete real-time keyword recognition engine.

    Connects:
        Microphone / Stream Source -> Software Ping-Pong Buffer ->
        Bounded Queue -> Sliding Window Framer -> Adaptive VAD ->
        Context Window -> Log-Mel Feature Extractor -> Residual DS-CNN.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/dscnn_numbers_v1.pt",
        confidence_threshold: float = 0.65,
        margin_threshold: float = 0.15,
        smoothing_alpha: float = 0.70,
        cooldown_seconds: float = 0.80,
        sample_rate: int = 16000,
        hop_size: int = 320,
        frame_size: int = 640,
        queue_capacity: int = 100,
        device_id: Optional[int] = None,
        on_command_callback: Optional[Callable[[RecognitionResult, LivePipelineTelemetry], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_size = frame_size
        self.on_command_callback = on_command_callback

        # 1. Initialize Inference Service (Task A & C)
        logger.info("Initializing KeywordSpotterInferenceService...")
        self.inference_service = KeywordSpotterInferenceService(
            checkpoint_path=checkpoint_path,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            smoothing_alpha=smoothing_alpha,
            cooldown_seconds=cooldown_seconds,
        )

        # 2. Audio Buffering Subsystem
        self.ping_pong = SoftwarePingPongBuffer(buffer_size=hop_size, dtype=np.int16)
        self.audio_queue = BoundedAudioQueue(
            capacity=queue_capacity,
            overflow_policy=QueueOverflowPolicy.DROP_OLDEST,
        )

        # 3. DSP Framing and Voice Activity Detection
        self.framer = SlidingWindowFramer(frame_size=frame_size, hop_size=hop_size)
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            hop_size=hop_size,
            threshold_factor=2.5,
            min_speech_duration_ms=250.0,
            max_speech_duration_ms=1000.0,
            hangover_duration_ms=160.0,
            pre_roll_duration_ms=200.0,
        )

        # 4. Context Rolling Audio Buffer (16,000 samples = 1.0s @ 16 kHz)
        self.target_length = 16000
        self._rolling_buffer = np.zeros(self.target_length, dtype=np.float32)

        # 5. Microphone Hardware Setup (Optional: initialized on start if device_id provided)
        self.device_id = device_id
        self.mic: Optional[MicrophoneCapture] = None

        # 6. Worker Thread Control
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Telemetry & Diagnostics
        self.total_hops_processed = 0
        self.total_vad_triggers = 0
        self.total_inferences_run = 0
        self.confirmed_commands_history: List[Tuple[RecognitionResult, LivePipelineTelemetry]] = []

    def start(self, use_microphone: bool = True):
        """Starts the audio acquisition and processing engine."""
        with self._lock:
            if self._is_running:
                return

            self._is_running = True
            self.total_hops_processed = 0
            self.total_vad_triggers = 0
            self.total_inferences_run = 0
            self.confirmed_commands_history.clear()
            self.inference_service.reset_state()
            self.framer.reset()

            # Start worker thread
            self._worker_thread = threading.Thread(
                target=self._processing_worker_loop,
                name="AudioProcessingWorker",
                daemon=True,
            )
            self._worker_thread.start()

            # Start physical microphone if requested
            if use_microphone:
                dev_id = self.device_id
                if dev_id is None:
                    dev_id = get_default_input_device().device_id

                self.mic = MicrophoneCapture(
                    device_id=dev_id,
                    sample_rate=self.sample_rate,
                    channels=1,
                    hop_size=self.hop_size,
                    dtype="int16",
                    queue=self.audio_queue,
                    ping_pong=self.ping_pong,
                )
                self.mic.start()
                logger.info(f"Microphone capture started on device ID {dev_id}.")

    def stop(self):
        """Gracefully stops the live recognition pipeline."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        if self.mic is not None:
            try:
                self.mic.stop()
            except Exception as e:
                logger.warning(f"Error stopping microphone: {e}")
            self.mic = None

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        logger.info("Live recognition engine stopped cleanly.")

    def push_external_hop(self, hop_pcm16: np.ndarray, capture_time: Optional[float] = None) -> AudioBlock:
        """
        Allows feeding external recorded audio hops for reproducible testing.
        Pushes through the exact same ping-pong buffer and queue as live mic.
        """
        now = time.perf_counter() if capture_time is None else capture_time
        seq = self.ping_pong.total_written
        slot_idx, was_overrun = self.ping_pong.write_block(hop_pcm16, sequence_number=seq)
        block = AudioBlock(
            data=hop_pcm16.copy(),
            sequence_number=seq,
            capture_timestamp=now,
            sample_rate=self.sample_rate,
            channels=1,
            buffer_id=slot_idx,
            enqueue_timestamp=time.perf_counter(),
        )
        self.audio_queue.push(block)
        return block

    def _processing_worker_loop(self):
        """
        Main audio processing loop running in a dedicated thread.
        Consumes blocks from the audio queue, processes DSP, and invokes ML inference.
        """
        logger.info("Processing worker thread running.")

        while self._is_running:
            block = self.audio_queue.pop(timeout=0.05)
            if block is None:
                continue

            t_dequeue = time.perf_counter()
            block.dequeue_timestamp = t_dequeue

            # Read and release corresponding slot in ping-pong buffer
            _ = self.ping_pong.read_ready_buffer(timeout=0)

            self.total_hops_processed += 1
            hop_data = block.data
            hop_f32 = pcm16_to_float32(hop_data)

            # 1. Update Continuous Rolling Context Buffer (16,000 samples)
            # Shift buffer left by hop_size (320) and insert new hop at the end
            self._rolling_buffer[:-self.hop_size] = self._rolling_buffer[self.hop_size:]
            self._rolling_buffer[-self.hop_size:] = hop_f32

            # 2. Update Sliding Window Framer (320 -> 640 samples)
            frame_pcm16 = self.framer.process_hop(hop_data)

            # 3. Voice Activity Detection (VAD)
            completed_utterance = self.vad.process_hop(hop_data)

            should_infer = False
            utterance_onset_time: Optional[float] = None
            audio_for_inference: Optional[np.ndarray] = None

            # Condition A: Utterance completed by VAD hangover termination
            if completed_utterance is not None:
                self.total_vad_triggers += 1
                should_infer = True
                utterance_onset_time = completed_utterance.start_timestamp
                # Use the isolated utterance audio (with pre-roll)
                audio_for_inference = pcm16_to_float32(completed_utterance.audio)

            # Condition B: Continuous active speech window (every 5 hops = 100 ms during active speech)
            elif self.vad.state in (VADState.SPEECH_ACTIVE, VADState.HANGOVER):
                if self.total_hops_processed % 5 == 0:
                    should_infer = True
                    utterance_onset_time = self.vad._utterance_start_time
                    audio_for_inference = self._rolling_buffer.copy()

            if should_infer and audio_for_inference is not None:
                self.total_inferences_run += 1
                t_event_completion = t_dequeue

                # 4. Invoke Machine Learning Inference Service
                result = self.inference_service.predict(
                    audio_samples=audio_for_inference,
                    audio_event_timestamp=t_event_completion,
                )

                # 5. Measure End-to-End Latencies
                t_decision = result.decision_timestamp
                capture_delay_ms = (t_dequeue - block.capture_timestamp) * 1000.0
                queue_delay_ms = block.queue_latency_ms or 0.0
                reaction_latency_ms = (t_decision - t_event_completion) * 1000.0

                acoustic_latency_ms: Optional[float] = None
                if utterance_onset_time and utterance_onset_time > 0:
                    acoustic_latency_ms = (t_decision - utterance_onset_time) * 1000.0

                telemetry = LivePipelineTelemetry(
                    sequence_number=block.sequence_number,
                    audio_event_timestamp=t_event_completion,
                    utterance_onset_timestamp=utterance_onset_time,
                    decision_timestamp=t_decision,
                    capture_and_buffering_delay_ms=round(capture_delay_ms, 3),
                    queue_delay_ms=round(queue_delay_ms, 3),
                    feature_extraction_ms=result.timings.get("feature_extraction_ms", 0.0),
                    model_inference_ms=result.timings.get("model_inference_ms", 0.0),
                    postprocessing_ms=result.timings.get("postprocessing_ms", 0.0),
                    reaction_latency_ms=round(reaction_latency_ms, 3),
                    total_acoustic_latency_ms=round(acoustic_latency_ms, 3) if acoustic_latency_ms else None,
                )

                if result.is_confirmed:
                    self.confirmed_commands_history.append((result, telemetry))
                    logger.info(
                        f"[COMMAND CONFIRMED] '{result.command}' (Conf: {result.confidence*100:.1f}%, "
                        f"Reaction: {telemetry.reaction_latency_ms:.2f} ms)"
                    )

                if self.on_command_callback is not None:
                    try:
                        self.on_command_callback(result, telemetry)
                    except Exception as e:
                        logger.error(f"Error in on_command_callback: {e}")

        logger.info("Processing worker thread terminated.")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic and operational metrics."""
        return {
            "is_running": self._is_running,
            "total_hops_processed": self.total_hops_processed,
            "total_vad_triggers": self.total_vad_triggers,
            "total_inferences_run": self.total_inferences_run,
            "total_confirmed_commands": len(self.confirmed_commands_history),
            "ping_pong_status": self.ping_pong.get_status(),
            "queue_diagnostics": self.audio_queue.get_metrics(),
            "vad_diagnostics": self.vad.get_diagnostics(),
            "inference_service_stats": self.inference_service.get_stats(),
        }
