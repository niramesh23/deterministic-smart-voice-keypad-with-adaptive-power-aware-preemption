"""
audio Subsystem
===============
Responsibilities:
- Microphone audio stream ingestion and capture (hardware / device interface).
- Audio framing, chunking, and hop-size window generation.
- Software-managed buffers (e.g. ping-pong double buffers, frame ring buffers, and overrun/underrun guards).
- Audio stream replay simulation for deterministic benchmarking and offline testing.

Design Principles:
- Isolate hardware driver/device calls behind clean interfaces.
- Non-blocking push/pull semantics for real-time compliance.
- No direct dependency on downstream ML or DSP processing logic.
"""

from src.audio.types import (
    AudioBlock,
    AudioDeviceInfo,
    BufferState,
    QueueOverflowPolicy,
)
from src.audio.buffers import (
    BufferOverrunError,
    SoftwarePingPongBuffer,
)
from src.audio.queue import BoundedAudioQueue
from src.audio.framing import (
    pcm16_to_float32,
    float32_to_pcm16,
    SlidingWindowFramer,
)
from src.audio.microphone import (
    AudioDeviceError,
    AudioFormatError,
    AudioPermissionError,
    MicrophoneCapture,
    list_audio_input_devices,
    get_default_input_device,
    verify_device_support,
    resample_linear,
)

__all__ = [
    "AudioBlock",
    "AudioDeviceInfo",
    "BufferState",
    "QueueOverflowPolicy",
    "BufferOverrunError",
    "SoftwarePingPongBuffer",
    "BoundedAudioQueue",
    "pcm16_to_float32",
    "float32_to_pcm16",
    "SlidingWindowFramer",
    "AudioDeviceError",
    "AudioFormatError",
    "AudioPermissionError",
    "MicrophoneCapture",
    "list_audio_input_devices",
    "get_default_input_device",
    "verify_device_support",
    "resample_linear",
]
