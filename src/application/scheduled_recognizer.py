"""
src/application/scheduled_recognizer.py
---------------------------------------
Scheduled live keyword recognition pipeline integrated with application-level
Rate Monotonic (RM) and Earliest Deadline First (EDF) real-time scheduling.

Architecture:
1. Audio Acquisition (Microphone Callback / push_external_hop):
   - Strictly non-blocking (< 0.1 ms).
   - Writes 320 PCM16 samples to SoftwarePingPongBuffer.
   - Pushes AudioBlock to BoundedAudioQueue.
   - Submits AUDIO_BLOCK real-time job (T=20ms, D=20ms, P=1) to RealTimeScheduler.
2. Real-Time Scheduler Dispatcher:
   - Evaluates all pending eligible jobs in the ready queue.
   - Selects the next job based on active policy:
     * FIFO: Baseline arrival ordering.
     * RM: Fixed period priority (1=AUDIO_BLOCK, 2=FEATURE_EXTRACTION, 3=KEYWORD_INFERENCE, 4=MONITORING).
     * EDF: Dynamic earliest absolute deadline ordering (d_i = r_i + D_i).
   - Records every SchedulingDecision and measures scheduler dispatch overhead.
3. Feature Extraction Job (D=40ms, P=2):
   - Dispatched when VAD detects speech or speech completion.
   - Computes 40-bin Log-Mel filterbank spectrogram from 1.0s context buffer.
   - Submits KEYWORD_INFERENCE job to RealTimeScheduler.
4. Keyword Inference Job (D=100ms, P=3):
   - Dispatched by scheduler.
   - Executes PyTorch forward pass, probability smoothing, margin check, and cooldown.
5. Periodic Monitoring Job (T=200ms, D=200ms, P=4):
   - Background telemetry, CPU tracking, and queue health monitoring.
"""

from dataclasses import dataclass, field
import logging
import pathlib
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

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
from src.application.live_recognizer import LivePipelineTelemetry
from src.rts.tasks import (
    RealTimeJob,
    JobType,
    JobStatus,
    create_audio_block_job,
    create_feature_extraction_job,
    create_keyword_inference_job,
    create_monitoring_job,
)
from src.rts.scheduler import (
    RealTimeScheduler,
    SchedulingPolicy,
    LateJobStrategy,
    SchedulingDecision,
)
from src.rts.monitor import RealTimeMonitor
from src.rts.adaptive import (
    ControllerState,
    AdaptationAction,
    AdaptationDecision,
    AdaptiveControllerConfig,
    AdaptiveProcessingController,
)

logger = logging.getLogger("rtos.scheduled_recognizer")


class ScheduledKeywordRecognizer:
    """
    Live Keyword Recognition Engine with Integrated Real-Time Scheduling.
    """

    def __init__(
        self,
        policy: SchedulingPolicy = SchedulingPolicy.EDF,
        late_job_strategy: LateJobStrategy = LateJobStrategy.EXECUTE_AND_RECORD,
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
        on_vad_state_callback: Optional[Callable[[VADState], None]] = None,
        enable_adaptation: bool = True,
        adaptive_config: Optional[AdaptiveControllerConfig] = None,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_size = frame_size
        self.on_command_callback = on_command_callback
        self.on_vad_state_callback = on_vad_state_callback
        self.device_id = device_id
        self.enable_adaptation = enable_adaptation
        self.controller = AdaptiveProcessingController(config=adaptive_config)

        # 1. Inference Service
        self.inference_service = KeywordSpotterInferenceService(
            checkpoint_path=checkpoint_path,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            smoothing_alpha=smoothing_alpha,
            cooldown_seconds=cooldown_seconds,
        )

        # 2. Audio Buffering
        self.ping_pong = SoftwarePingPongBuffer(buffer_size=hop_size, dtype=np.int16)
        self.audio_queue = BoundedAudioQueue(
            capacity=queue_capacity,
            overflow_policy=QueueOverflowPolicy.DROP_OLDEST,
        )

        # 3. DSP & VAD
        self.framer = SlidingWindowFramer(frame_size=frame_size, hop_size=hop_size)
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            hop_size=hop_size,
            threshold_factor=2.5,
            min_speech_duration_ms=180.0,
            max_speech_duration_ms=1000.0,
            hangover_duration_ms=160.0,
            pre_roll_duration_ms=200.0,
        )

        # 4. Rolling Context Buffer (16,000 samples = 1.0s)
        self.target_length = 16000
        self._rolling_buffer = np.zeros(self.target_length, dtype=np.float32)

        # 5. Real-Time Scheduler & Monitor
        self.scheduler = RealTimeScheduler(
            policy=policy,
            late_job_strategy=late_job_strategy,
            max_queue_capacity=queue_capacity * 2,
            on_job_completed=self._on_scheduler_job_completed,
        )
        self.monitor = RealTimeMonitor()

        # 6. Operational State
        self.mic: Optional[MicrophoneCapture] = None
        self._mic_ingest_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._lock = threading.Lock()
        self.analytics_collector: Optional[Any] = None
        self._calibrating_mic_hops = 0
        self._calibration_rms_list: List[float] = []

        # Telemetry & History
        self.total_audio_blocks_ingested = 0
        self.total_audio_blocks_dropped = 0
        self.total_hops_processed = 0
        self.total_vad_triggers = 0
        self.total_inferences_run = 0
        self._last_evaluated_drops = 0
        self.confirmed_commands_history: List[Tuple[RecognitionResult, LivePipelineTelemetry]] = []

        # Periodic Monitoring Tracking
        self._last_monitoring_time = 0.0
        self._monitoring_interval_s = 0.200  # 200 ms

    @property
    def policy(self) -> SchedulingPolicy:
        return self.scheduler.policy

    def set_policy(self, policy: SchedulingPolicy):
        """Switches scheduling policy (FIFO, RM, or EDF) at runtime."""
        self.scheduler.set_policy(policy)

    def start(self, use_microphone: bool = True):
        """Starts real-time scheduled recognition engine."""
        with self._lock:
            if self._is_running:
                return

            self._is_running = True
            self.total_audio_blocks_ingested = 0
            self.total_audio_blocks_dropped = 0
            self._last_evaluated_drops = 0
            self.total_hops_processed = 0
            self.total_vad_triggers = 0
            self.total_inferences_run = 0
            self.confirmed_commands_history.clear()
            self.inference_service.reset_state()
            self.framer.reset()
            self.scheduler.clear()
            self.controller.reset()
            self.monitor.reset_cpu_timer()
            self._last_monitoring_time = time.perf_counter()

            # Start asynchronous scheduler worker thread
            self.scheduler.start_async_worker()

            # Start physical microphone if requested
            if use_microphone:
                self._calibrating_mic_hops = 25  # ~500 ms ambient calibration window
                self._calibration_rms_list.clear()
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

                # Start ingest bridge thread to pull from audio_queue and submit to scheduler
                self._mic_ingest_thread = threading.Thread(
                    target=self._mic_ingest_loop,
                    name="MicIngestBridge",
                    daemon=True,
                )
                self._mic_ingest_thread.start()

    def stop(self):
        """Gracefully stops scheduled engine."""
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

        if self._mic_ingest_thread is not None and self._mic_ingest_thread.is_alive():
            self._mic_ingest_thread.join(timeout=1.0)
            self._mic_ingest_thread = None

        self.scheduler.stop_async_worker()
        logger.info("Scheduled recognition engine stopped cleanly.")

    def restart_microphone(self) -> bool:
        """
        Attempts controlled audio capture reinitialization.
        Used by RecoveryManager during audio stream faults.
        """
        with self._lock:
            if not self._is_running:
                return False

            # Stop existing microphone stream if present
            if self.mic is not None:
                try:
                    self.mic.stop()
                except Exception as e:
                    logger.debug(f"[RECOGNIZER] Error stopping previous microphone: {e}")
                self.mic = None

            # Drain queue backlog
            if self.audio_queue is not None:
                self.audio_queue.clear()

            # Attempt reinitialization
            try:
                from src.audio.microphone import get_default_input_device, MicrophoneCapture
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
                logger.info(f"[RECOGNIZER] Microphone capture restarted on device ID {dev_id}.")
                return True
            except Exception as e:
                logger.error(f"[RECOGNIZER] Microphone restart failed: {e}")
                return False

    def _mic_ingest_loop(self):
        """
        Pulls audio blocks from audio_queue and submits them as AUDIO_BLOCK jobs to scheduler.
        Includes watchdog health checks and loop error isolation.
        """
        consecutive_empty_pops = 0
        while self._is_running:
            try:
                block = self.audio_queue.pop(timeout=0.02)
                if block is None:
                    consecutive_empty_pops += 1
                    # After ~1 second of zero blocks (50 * 20ms), check if hardware stream stalled
                    if consecutive_empty_pops >= 50:
                        if self.mic is not None and hasattr(self.mic, "is_active") and not self.mic.is_active:
                            logger.warning("[MIC INGEST] PortAudio stream is inactive! Auto-restarting microphone...")
                            self.restart_microphone()
                        consecutive_empty_pops = 0
                    continue

                consecutive_empty_pops = 0

                # Live microphone ambient noise floor calibration
                if self._calibrating_mic_hops > 0:
                    block_rms = float(np.sqrt(np.mean((block.data.astype(np.float32) / 32768.0) ** 2)))
                    self._calibration_rms_list.append(block_rms)
                    self._calibrating_mic_hops -= 1
                    if self._calibrating_mic_hops == 0 and self._calibration_rms_list:
                        ambient_floor = float(np.median(self._calibration_rms_list))
                        clamped_floor = max(0.0008, ambient_floor)
                        self.vad.noise_floor.set_floor(clamped_floor)
                        self.vad.threshold_factor = 1.8
                        self.vad.noise_floor.threshold_factor = 1.8
                        logger.info(
                            f"[MIC CALIBRATION] Ambient noise floor calibrated to {clamped_floor:.6f} RMS "
                            f"(VAD Threshold: {self.vad.noise_floor.threshold:.6f}, Factor: 1.8x)"
                        )
                        self._calibration_rms_list.clear()

                self.total_audio_blocks_ingested += 1
                now = time.perf_counter()
                job = create_audio_block_job(
                    job_id=f"audio_{block.sequence_number}",
                    work_fn=self._job_audio_block,
                    args=(block,),
                    release_time=now,
                    expected_release_time=now,
                )
                accepted = self.scheduler.submit_job(job)
                if not accepted:
                    self.total_audio_blocks_dropped += 1

                # Check periodic monitoring release
                curr_interval = self.controller.get_monitoring_interval_s() if self.enable_adaptation else self._monitoring_interval_s
                if now - self._last_monitoring_time >= curr_interval:
                    mon_job = create_monitoring_job(
                        job_id=f"mon_{int(now * 1000)}",
                        work_fn=self._job_monitoring,
                        release_time=now,
                        expected_release_time=self._last_monitoring_time + curr_interval,
                    )
                    self.scheduler.submit_job(mon_job)
                    self._last_monitoring_time = now
            except Exception as loop_err:
                logger.error(f"[MIC INGEST] Error in ingest loop: {loop_err}", exc_info=True)
                time.sleep(0.01)

    def push_external_hop(self, hop_pcm16: np.ndarray, capture_time: Optional[float] = None) -> AudioBlock:
        """
        Feeds an external audio hop through Ping-Pong buffer, queue, and scheduler.
        """
        now = time.perf_counter() if capture_time is None else capture_time
        seq = self.ping_pong.total_written
        slot_idx, was_overrun = self.ping_pong.write_block(hop_pcm16, sequence_number=seq)
        if self.analytics_collector is not None:
            self.analytics_collector.record_buffer_switch(now)
        block = AudioBlock(
            data=hop_pcm16.copy(),
            sequence_number=seq,
            capture_timestamp=now,
            sample_rate=self.sample_rate,
            channels=1,
            buffer_id=slot_idx,
            enqueue_timestamp=time.perf_counter(),
        )
        enqueued = self.audio_queue.push(block)
        self.total_audio_blocks_ingested += 1
        if not enqueued:
            self.total_audio_blocks_dropped += 1

        # Submit AUDIO_BLOCK real-time job to the scheduler
        expected_rel = now
        job = create_audio_block_job(
            job_id=f"audio_{seq}",
            work_fn=self._job_audio_block,
            args=(block,),
            release_time=now,
            expected_release_time=expected_rel,
        )
        accepted = self.scheduler.submit_job(job)
        if not accepted:
            self.total_audio_blocks_dropped += 1

        # Check periodic monitoring job release
        curr_interval = self.controller.get_monitoring_interval_s() if self.enable_adaptation else self._monitoring_interval_s
        if now - self._last_monitoring_time >= curr_interval:
            mon_job = create_monitoring_job(
                job_id=f"mon_{int(now * 1000)}",
                work_fn=self._job_monitoring,
                release_time=now,
                expected_release_time=self._last_monitoring_time + curr_interval,
            )
            self.scheduler.submit_job(mon_job)
            self._last_monitoring_time = now

        return block

    # ==========================================================================
    # Schedulable Job Work Functions
    # ==========================================================================

    def _job_audio_block(self, block: AudioBlock):
        """
        Job 1: Audio Block Processing (T=20ms, D=20ms, Priority 1).
        Performs ping-pong buffer sync, framing, RMS energy, and VAD update.
        """
        t_dequeue = time.perf_counter()
        block.dequeue_timestamp = t_dequeue

        # Release corresponding slot in ping-pong buffer
        _ = self.ping_pong.read_ready_buffer(timeout=0)
        if self._mic_ingest_thread is None:
            self.audio_queue.pop(timeout=0)

        self.total_hops_processed += 1
        hop_data = block.data
        hop_f32 = pcm16_to_float32(hop_data)

        # Update Continuous Rolling Context Buffer (16,000 samples)
        self._rolling_buffer[:-self.hop_size] = self._rolling_buffer[self.hop_size:]
        self._rolling_buffer[-self.hop_size:] = hop_f32

        # Framing and VAD
        prev_vad_state = self.vad.state
        frame_pcm16 = self.framer.process_hop(hop_data)
        completed_utterance = self.vad.process_hop(hop_data)
        t_proc_done = time.perf_counter()

        # Acoustic VAD state transition and periodic clock notification to FSM
        if self.on_vad_state_callback is not None:
            try:
                self.on_vad_state_callback(self.vad.state)
            except Exception as e:
                logger.debug(f"[RECOGNIZER] Error in on_vad_state_callback: {e}")

        if self.analytics_collector is not None:
            self.analytics_collector.record_audio_block_timing(
                capture_timestamp=block.capture_timestamp,
                enqueue_timestamp=block.enqueue_timestamp,
                dequeue_timestamp=block.dequeue_timestamp,
                framer_processed_timestamp=t_proc_done,
            )

        should_infer = False
        is_completed_utterance = False
        utterance_onset_time: Optional[float] = None
        audio_for_inference: Optional[np.ndarray] = None

        if self.enable_adaptation:
            # Evaluate adaptive controller state with windowed drop delta (prevents permanent OVERLOAD trap)
            audio_occ = self.audio_queue.get_metrics().get("current_occupancy", 0)
            queue_occ = max(audio_occ, self.scheduler.ready_count)
            q_drops = self.audio_queue.get_metrics().get("dropped_blocks_count", 0) if self.audio_queue else 0
            total_drops_now = self.total_audio_blocks_dropped + q_drops + self.scheduler.total_dropped
            recent_drops = max(0, total_drops_now - self._last_evaluated_drops)
            self._last_evaluated_drops = total_drops_now

            ctrl_state = self.controller.evaluate_state(
                vad_state=self.vad.state,
                queue_occupancy=queue_occ,
                dropped_blocks=recent_drops,
            )

            # In OVERLOAD: Prune stale queued inference jobs if backlog is accumulating
            if ctrl_state == ControllerState.OVERLOAD and self.scheduler.ready_count > 4:
                with self.scheduler._lock:
                    for j in list(self.scheduler._ready_jobs):
                        if j.job_type in (JobType.KEYWORD_INFERENCE, JobType.FEATURE_EXTRACTION):
                            self.scheduler.cancel_job(j.job_id)
                            self.controller.total_stale_jobs_pruned += 1

            if completed_utterance is not None and self.controller.should_infer_completed_utterance():
                self.total_vad_triggers += 1
                should_infer = True
                is_completed_utterance = True
                utterance_onset_time = completed_utterance.start_timestamp
                utt_f32 = pcm16_to_float32(completed_utterance.audio)
                if len(utt_f32) < 16000:
                    pad_len = 16000 - len(utt_f32)
                    audio_for_inference = np.pad(utt_f32, (0, pad_len), mode="constant")
                else:
                    audio_for_inference = utt_f32[:16000]

            elif self.vad.state in (VADState.SPEECH_ACTIVE, VADState.HANGOVER):
                if self.controller.should_infer_sliding(self.total_hops_processed):
                    should_infer = True
                    is_completed_utterance = False
                    utterance_onset_time = self.vad._utterance_start_time
                    audio_for_inference = self._rolling_buffer.copy()
                elif self.controller.state == ControllerState.OVERLOAD and self.total_hops_processed % 10 == 0:
                    # In OVERLOAD fallback: do not mute speech completely; throttle to 10-hop (200ms) cadence
                    should_infer = True
                    is_completed_utterance = False
                    utterance_onset_time = self.vad._utterance_start_time
                    audio_for_inference = self._rolling_buffer.copy()

        else:
            # Baseline unadapted mode: Evaluates continuous sliding window inference every 5 hops (100 ms)
            # without silence-aware gating or overload throttling.
            if completed_utterance is not None:
                self.total_vad_triggers += 1
                should_infer = True
                is_completed_utterance = True
                utterance_onset_time = completed_utterance.start_timestamp
                utt_f32 = pcm16_to_float32(completed_utterance.audio)
                if len(utt_f32) < 16000:
                    pad_len = 16000 - len(utt_f32)
                    audio_for_inference = np.pad(utt_f32, (0, pad_len), mode="constant")
                else:
                    audio_for_inference = utt_f32[:16000]

            elif self.total_hops_processed % 5 == 0:
                should_infer = True
                is_completed_utterance = False
                utterance_onset_time = self.vad._utterance_start_time
                audio_for_inference = self._rolling_buffer.copy()

        # If speech event triggered, submit FEATURE_EXTRACTION real-time job
        if should_infer and audio_for_inference is not None:
            t_event_completion = t_dequeue
            feat_job = create_feature_extraction_job(
                job_id=f"feat_{block.sequence_number}",
                work_fn=self._job_feature_extraction,
                args=(audio_for_inference, block, utterance_onset_time, t_event_completion, is_completed_utterance),
                release_time=time.perf_counter(),
            )
            self.scheduler.submit_job(feat_job)

    def _job_feature_extraction(
        self,
        audio_samples: np.ndarray,
        block: AudioBlock,
        utterance_onset_time: Optional[float],
        t_event_completion: float,
        is_completed_utterance: bool = False,
    ):
        """
        Job 2: DSP Feature Extraction (D=40ms, Priority 2).
        Computes 40-bin Log-Mel filterbank spectrogram and submits Keyword Inference job.
        """
        t_dsp_0 = time.perf_counter()
        log_mel = self.inference_service.extract_features(audio_samples)
        t_dsp_1 = time.perf_counter()
        dsp_time_ms = (t_dsp_1 - t_dsp_0) * 1000.0

        if self.analytics_collector is not None:
            self.analytics_collector.record_feature_extraction(dsp_time_ms)

        # Submit KEYWORD_INFERENCE real-time job
        infer_job = create_keyword_inference_job(
            job_id=f"infer_{block.sequence_number}",
            work_fn=self._job_keyword_inference,
            args=(log_mel, block, utterance_onset_time, t_event_completion, dsp_time_ms, is_completed_utterance),
            release_time=time.perf_counter(),
        )
        self.scheduler.submit_job(infer_job)

    def _job_keyword_inference(
        self,
        log_mel: np.ndarray,
        block: AudioBlock,
        utterance_onset_time: Optional[float],
        t_event_completion: float,
        dsp_time_ms: float,
        is_completed_utterance: bool = False,
    ):
        """
        Job 3: Keyword Inference (D=100ms, Priority 3).
        Evaluates DS-CNN forward pass, smoothing, margin check, and command confirmation.
        """
        self.total_inferences_run += 1

        result = self.inference_service.predict_from_features(
            log_mel=log_mel,
            audio_event_timestamp=t_event_completion,
            dsp_time_ms=dsp_time_ms,
            reset_smoothing=is_completed_utterance,
        )

        t_decision = result.decision_timestamp
        capture_delay_ms = (block.dequeue_timestamp - block.capture_timestamp) * 1000.0 if block.dequeue_timestamp else 0.0
        queue_delay_ms = block.queue_latency_ms or 0.0
        reaction_latency_ms = (t_decision - t_event_completion) * 1000.0

        if self.analytics_collector is not None:
            infer_ms = result.timings.get("model_inference_ms", 0.0)
            self.analytics_collector.record_inference_execution(infer_ms)
            self.analytics_collector.record_e2e_audio_to_inference(block.capture_timestamp, t_decision)

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
                f"[SCHEDULER {self.scheduler.policy.value}] Command Confirmed: '{result.command}' "
                f"(Conf: {result.confidence*100:.1f}%, Reaction: {telemetry.reaction_latency_ms:.2f} ms)"
            )

        if self.on_command_callback is not None:
            try:
                self.on_command_callback(result, telemetry)
            except Exception as e:
                logger.error(f"Error in on_command_callback: {e}")

    def _job_monitoring(self):
        """
        Job 4: Periodic Monitoring (T=200ms, D=200ms, Priority 4).
        Collects system telemetry, queue occupancy, and CPU utilization.
        """
        cpu_pct = self.monitor.get_cpu_utilization()
        q_metrics = self.audio_queue.get_metrics()
        noise_floor = self.vad.noise_floor.noise_floor
        # Low-overhead log at debug level
        logger.debug(
            f"[MONITOR] CPU: {cpu_pct:.1f}%, Queue: {q_metrics.get('current_occupancy', 0)}, "
            f"Noise Floor RMS: {noise_floor:.6f}"
        )

    def _on_scheduler_job_completed(self, job: RealTimeJob):
        """Scheduler callback to feed deadline miss and queue delay metrics to adaptive controller and analytics."""
        if self.enable_adaptation and self.controller is not None:
            self.controller.record_job_completion(
                deadline_missed=job.deadline_missed,
                queue_delay_ms=job.queue_delay_ms or 0.0,
            )
        if self.analytics_collector is not None:
            self.analytics_collector.record_scheduler_job_execution(
                execution_time_ms=job.measured_execution_time_ms or 0.0,
                response_time_ms=job.response_time_ms or 0.0,
                waiting_time_ms=job.queue_delay_ms or 0.0,
            )

    def get_analysis_report(self) -> Dict[str, Any]:
        """Generates comprehensive timing, scheduling, and adaptive control report."""
        report = self.monitor.analyze_scheduler_run(
            scheduler=self.scheduler,
            total_audio_blocks_ingested=self.total_audio_blocks_ingested,
            total_audio_blocks_dropped=self.total_audio_blocks_dropped,
        )
        report["total_inferences_run"] = self.total_inferences_run
        if self.enable_adaptation and self.controller is not None:
            report["adaptation"] = {
                "enabled": True,
                "current_state": self.controller.state.value,
                "total_state_transitions": self.controller.total_state_transitions,
                "total_overload_events": self.controller.total_overload_events,
                "total_inferences_suppressed": self.controller.total_inferences_suppressed,
                "total_stale_jobs_pruned": self.controller.total_stale_jobs_pruned,
                "mean_recent_queue_delay_ms": round(self.controller.mean_recent_queue_delay_ms, 3),
                "decisions_recorded": len(self.controller.decision_history),
            }
        else:
            report["adaptation"] = {"enabled": False}

        return report
