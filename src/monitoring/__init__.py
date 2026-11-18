"""
monitoring Subsystem
====================
Responsibilities:
- Structured application logging (console and rotating/persistent file logs).
- Real-time performance metrics collection (loop execution duration, jitter, buffer occupancy).
- Deadline miss counters and timing violation diagnostics.
- Telemetry reporting and experiment benchmarking metrics export.

Design Principles:
- Minimal overhead in real-time execution hot-paths.
- Safe logging handlers that never block real-time audio threads.
"""

from src.monitoring.logger import setup_logging, get_logger
from src.monitoring.structured_logger import StructuredExperimentLogger

__all__ = ["setup_logging", "get_logger", "StructuredExperimentLogger"]

