"""
src/audio/microphone.py
-----------------------
Physical microphone acquisition module utilizing PortAudio via sounddevice.

Features:
- Enumerate and query host audio input devices.
- Hardware capability checking and target format negotiation (16,000 Hz, mono, 16-bit PCM).
- Fast, non-blocking real-time audio callback.
- Direct integration with SoftwarePingPongBuffer and BoundedAudioQueue.
- Graceful startup, shutdown, and robust error handling for permissions and device state.
- Resampling/conversion support outside the callback if hardware requires alternate rates.
"""

from collections import deque
import logging
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import sounddevice as sd

logger = logging.getLogger("rtos.audio.microphone")

from src.audio.buffers import SoftwarePingPongBuffer
from src.audio.queue import BoundedAudioQueue
from src.audio.types import AudioBlock, AudioDeviceInfo


class AudioDeviceError(RuntimeError):
    """Raised when an audio input device error occurs."""
    pass


class AudioPermissionError(AudioDeviceError):
    """Raised when the application lacks permission to access the microphone."""
    pass


class AudioFormatError(AudioDeviceError):
    """Raised when the requested format cannot be negotiated with the hardware."""
    pass


def list_audio_input_devices() -> List[AudioDeviceInfo]:
    """
    Queries and returns all available audio input devices on the host system.
    """
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        hostapis = sd.query_hostapis()
    except Exception as e:
        raise AudioDeviceError(f"Failed to query system audio devices: {e}") from e

    input_devices = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            api_idx = dev.get("hostapi", 0)
            api_name = hostapis[api_idx]["name"] if api_idx < len(hostapis) else f"API-{api_idx}"

            info = AudioDeviceInfo(
                device_id=idx,
                name=dev.get("name", "Unknown"),
                hostapi_name=api_name,
                max_input_channels=dev.get("max_input_channels", 0),
                max_output_channels=dev.get("max_output_channels", 0),
                default_sample_rate=float(dev.get("default_samplerate", 0.0)),
                is_default_input=(idx == default_in),
            )
            input_devices.append(info)

    return input_devices


def get_default_input_device() -> AudioDeviceInfo:
    """Returns the default system input device information."""
    devices = list_audio_input_devices()
    for d in devices:
        if d.is_default_input:
            return d
    if devices:
        return devices[0]
    raise AudioDeviceError("No audio input devices found on this system.")


def verify_device_support(
    device_id: int,
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",
) -> Tuple[bool, Optional[str]]:
    """
    Checks whether a specific device natively supports the target audio parameters.

    Returns:
        Tuple of (is_supported: bool, error_message: Optional[str])
    """
    try:
        sd.check_input_settings(
            device=device_id,
            channels=channels,
            dtype=dtype,
            samplerate=sample_rate,
        )
        return True, None
    except Exception as e:
        return False, str(e)


def resample_linear(
    audio_data: np.ndarray, orig_sr: int, target_sr: int = 16000
) -> np.ndarray:
    """
    Resamples 1D audio array outside the real-time callback using linear interpolation.
    Used as a fallback if the host hardware only supports 44.1 kHz or 48 kHz.
    """
    if orig_sr == target_sr:
        return audio_data.copy()

    duration = len(audio_data) / orig_sr
    num_target_samples = int(round(duration * target_sr))
    orig_times = np.linspace(0, duration, len(audio_data), endpoint=False)
    target_times = np.linspace(0, duration, num_target_samples, endpoint=False)

    resampled = np.interp(target_times, orig_times, audio_data)
    if audio_data.dtype == np.int16:
        return np.round(resampled).astype(np.int16)
    return resampled.astype(audio_data.dtype)


class MicrophoneCapture:
    """
    Manages live microphone audio acquisition using sounddevice.InputStream.

    Guarantees:
    - Audio callback is strictly non-blocking (O(1) memory copy and queue push).
    - No disk I/O, heavy mathematics, or ML inference in the audio callback.
    - Synchronized with SoftwarePingPongBuffer and BoundedAudioQueue.
    """

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        hop_size: int = 320,
        dtype: str = "int16",
        queue: Optional[BoundedAudioQueue] = None,
        ping_pong: Optional[SoftwarePingPongBuffer] = None,
    ):
        """
        Initializes microphone capture instance.

        Parameters:
            device_id: Device index from list_audio_input_devices(). If None, uses default.
            sample_rate: Target sample rate in Hz (default: 16,000 Hz).
            channels: Target channels (default: 1 = mono).
            hop_size: Block size in samples per callback (default: 320 = 20 ms @ 16 kHz).
            dtype: Sample data format ("int16" or "float32").
            queue: Bounded audio queue to push captured blocks to.
            ping_pong: Software ping-pong buffer to alternate writes into.
        """
        self.device_id = device_id if device_id is not None else get_default_input_device().device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.hop_size = hop_size
        self.dtype = dtype

        self.queue = queue
        self.ping_pong = ping_pong

        self._stream: Optional[sd.InputStream] = None
        self._is_running = False
        self._lock = threading.Lock()

        # Telemetry & timing tracking (O(1) bounded deques)
        self._sequence_number = 0
        self._total_frames_captured = 0
        self._callback_count = 0
        self._callback_durations: deque[float] = deque(maxlen=1000)
        self._last_callback_time: Optional[float] = None
        self._callback_intervals: deque[float] = deque(maxlen=1000)
        self._stream_status_flags: deque[str] = deque(maxlen=200)

        # Actual verified stream parameters
        self.actual_device_name = ""
        self.actual_sample_rate = 0.0
        self.actual_channels = 0
        self.actual_latency = 0.0

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    @property
    def is_active(self) -> bool:
        """Checks if the underlying hardware/PortAudio stream is currently active and capturing."""
        with self._lock:
            if not self._is_running or self._stream is None:
                return False
            try:
                return bool(self._stream.active)
            except Exception:
                return False

    @property
    def device_info(self) -> Optional[AudioDeviceInfo]:
        """Returns structured metadata of the active input device."""
        try:
            return AudioDeviceInfo(
                device_id=self.device_id,
                name=self.actual_device_name or f"Device-{self.device_id}",
                max_input_channels=self.actual_channels or self.channels,
                default_sample_rate=self.actual_sample_rate or float(self.sample_rate),
                host_api="WASAPI",
                is_default=False,
            )
        except Exception:
            return None

    def start(self):
        """Validates format and opens the audio input stream."""
        with self._lock:
            if self._is_running:
                return

            # Verify device capability
            is_ok, err_msg = verify_device_support(
                device_id=self.device_id,
                sample_rate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
            )
            if not is_ok:
                raise AudioFormatError(
                    f"Device ID {self.device_id} cannot satisfy format "
                    f"({self.sample_rate} Hz, {self.channels} ch, {self.dtype}): {err_msg}"
                )

            # Query device details for logging
            dev_info = sd.query_devices(self.device_id)
            self.actual_device_name = dev_info.get("name", f"Device-{self.device_id}")

            try:
                self._stream = sd.InputStream(
                    device=self.device_id,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    dtype=self.dtype,
                    blocksize=self.hop_size,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._is_running = True
                self.actual_sample_rate = float(self._stream.samplerate)
                self.actual_channels = int(self._stream.channels)
                self.actual_latency = float(self._stream.latency)
            except sd.PortAudioError as e:
                err_text = str(e).lower()
                if "permission" in err_text or "access denied" in err_text:
                    raise AudioPermissionError(f"Microphone access permission denied: {e}") from e
                raise AudioDeviceError(f"Failed to start audio input stream: {e}") from e

    def stop(self):
        """Stops the audio stream and releases hardware resources cleanly."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ):
        """
        High-priority real-time audio callback.
        Must execute under strict deadline (~1 ms typical budget; never block!).
        Fully isolated with top-level exception handling so PortAudio never aborts.
        """
        try:
            t0 = time.perf_counter()

            if status:
                self._stream_status_flags.append(str(status))

            # Interval between callbacks
            if self._last_callback_time is not None:
                interval_ms = (t0 - self._last_callback_time) * 1000.0
                self._callback_intervals.append(interval_ms)
            self._last_callback_time = t0

            self._callback_count += 1
            self._sequence_number += 1
            self._total_frames_captured += frames

            # Extract mono 1D array
            if indata.ndim > 1:
                raw_samples = indata[:, 0].copy()
            else:
                raw_samples = indata.copy()

            # Step 1: Write to software ping-pong buffer (if configured)
            buffer_id = -1
            if self.ping_pong is not None:
                try:
                    buffer_id, _ = self.ping_pong.write_block(raw_samples, self._sequence_number)
                except Exception:
                    pass

            # Step 2: Package into AudioBlock
            t_dict = None
            if time_info is not None:
                t_dict = {
                    "input_adc_time": getattr(time_info, "inputBufferAdcTime", 0.0),
                    "current_time": getattr(time_info, "currentTime", 0.0),
                }

            block = AudioBlock(
                data=raw_samples,
                sequence_number=self._sequence_number,
                capture_timestamp=t0,
                sample_rate=self.sample_rate,
                channels=self.channels,
                buffer_id=buffer_id,
                status_flags=str(status) if status else None,
                time_info=t_dict,
            )

            # Step 3: Enqueue to bounded queue (Non-blocking O(1))
            if self.queue is not None:
                self.queue.push(block)

            # Timing telemetry (O(1) deque eviction)
            dur_ms = (time.perf_counter() - t0) * 1000.0
            self._callback_durations.append(dur_ms)
        except Exception as cb_err:
            logger.error(f"[MIC CALLBACK] Unhandled exception in PortAudio callback: {cb_err}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic and timing statistics."""
        with self._lock:
            durations = list(self._callback_durations)
            intervals = list(self._callback_intervals)

            mean_dur = sum(durations) / len(durations) if durations else 0.0
            max_dur = max(durations) if durations else 0.0
            mean_int = sum(intervals) / len(intervals) if intervals else 0.0
            p95_int = float(sorted(intervals)[int(len(intervals) * 0.95)]) if intervals else 0.0

            return {
                "is_running": self._is_running,
                "device_id": self.device_id,
                "device_name": self.actual_device_name,
                "actual_sample_rate": self.actual_sample_rate,
                "actual_channels": self.actual_channels,
                "actual_latency_s": self.actual_latency,
                "total_callbacks": self._callback_count,
                "total_samples_captured": self._total_frames_captured,
                "mean_callback_duration_ms": mean_dur,
                "max_callback_duration_ms": max_dur,
                "mean_callback_interval_ms": mean_int,
                "p95_callback_interval_ms": p95_int,
                "stream_status_flags": list(self._stream_status_flags),
            }

