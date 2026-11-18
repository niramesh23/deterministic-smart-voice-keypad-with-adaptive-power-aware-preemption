"""
application Subsystem
=====================
Responsibilities:
- High-level system lifecycle orchestration (initialization, startup, main loop, shutdown).
- Pipeline binding: connecting audio ingest -> DSP/VAD -> RTS queue -> ML inference -> action dispatcher.
- Command dispatching and event handling upon validated keyword recognition.
- Configuration loading, schema validation, and environment parameter resolution.

Design Principles:
- Depend on abstract interfaces of subsystems, preventing tight coupling.
- Clean graceful shutdown handling (signal trapping, resource deallocation).
"""

from src.application.config import load_config, Config
from src.application.live_recognizer import LiveKeywordRecognizer, LivePipelineTelemetry
from src.application.scheduled_recognizer import ScheduledKeywordRecognizer
from src.application.command_fsm import (
    CommandFiniteStateMachine,
    CommandState,
    CommandRecord,
    CommandFSMConfig,
    RejectionReason,
)
from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig

__all__ = [
    "load_config",
    "Config",
    "LiveKeywordRecognizer",
    "LivePipelineTelemetry",
    "ScheduledKeywordRecognizer",
    "CommandFiniteStateMachine",
    "CommandState",
    "CommandRecord",
    "CommandFSMConfig",
    "RejectionReason",
    "VoiceRecognitionApp",
    "OrchestratorConfig",
]

