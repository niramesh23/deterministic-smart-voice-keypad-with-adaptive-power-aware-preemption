"""
src/monitoring/structured_logger.py
-----------------------------------
Structured experiment logging and event record management.
Saves JSON Lines (JSONL) and structured JSON records for real-time analytics.

Logged Events:
- SCHEDULER_DECISION: Priority evaluation and job dispatch decisions.
- PREDICTION_RESULT: Candidate classification, confirmation, or rejection.
- CONTROLLER_TRANSITION: Adaptive processing state machine transitions.
- AUDIO_ERROR: Buffer overruns, underruns, or device drops.
- DEADLINE_MISS: Late job dispatch or completion events.
- QUEUE_OVERFLOW: Queue saturation and dropped block events.
- RESOURCE_MEASUREMENT: CPU percentage, memory RSS (MB).

Guarantees:
- Audio recording is strictly opt-in (disabled by default to avoid large files and latency).
- Thread-safe non-blocking appending.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
import pathlib
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rtos.structured_logger")


class LogEventType(str, Enum):
    pass


@dataclass
class StructuredEventRecord:
    event_id: int
    timestamp: float
    event_type: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": round(self.timestamp, 6),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "event_type": self.event_type,
            "details": self.details,
        }


class StructuredExperimentLogger:
    """
    Manages structured event records and persistent JSONL output.
    """

    def __init__(
        self,
        output_dir: str | pathlib.Path = "results/logs",
        session_id: Optional[str] = None,
        enable_audio_recording: bool = False,
        flush_interval_events: int = 50,
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.log_filepath = self.output_dir / f"session_{self.session_id}.jsonl"
        self.enable_audio_recording = enable_audio_recording
        self.flush_interval_events = flush_interval_events

        self._lock = threading.Lock()
        self._events: List[StructuredEventRecord] = []
        self._file_handle = open(self.log_filepath, "a", encoding="utf-8")
        self._counter: int = 0
        self._is_closed: bool = False

        logger.info(f"[STRUCTURED LOGGER] Session log opened: {self.log_filepath} (Audio logging: {self.enable_audio_recording})")

    def _ensure_open(self):
        """Re-opens log file handle if closed during restart."""
        if self._file_handle is None or self._file_handle.closed:
            self._file_handle = open(self.log_filepath, "a", encoding="utf-8")
            self._is_closed = False

    def _record_event(self, event_type: str, details: Dict[str, Any]):
        """Thread-safe event append and stream write."""
        with self._lock:
            self._ensure_open()
            self._counter += 1
            event = StructuredEventRecord(
                event_id=self._counter,
                timestamp=time.time(),
                event_type=event_type,
                details=details,
            )
            self._events.append(event)
            # Write immediate JSONL line
            line = json.dumps(event.to_dict()) + "\n"
            self._file_handle.write(line)
            if self._counter % self.flush_interval_events == 0:
                self._file_handle.flush()

    def log_scheduler_decision(self, decision: Dict[str, Any]):
        self._record_event("SCHEDULER_DECISION", decision)

    def log_prediction_result(self, record: Dict[str, Any]):
        self._record_event("PREDICTION_RESULT", record)

    def log_controller_transition(self, transition: Dict[str, Any]):
        self._record_event("CONTROLLER_TRANSITION", transition)

    def log_audio_error(self, error_type: str, details: Dict[str, Any]):
        payload = {"error_type": error_type, **details}
        self._record_event("AUDIO_ERROR", payload)

    def log_deadline_miss(self, job_id: str, deadline: float, completion: float, details: Dict[str, Any]):
        payload = {
            "job_id": job_id,
            "deadline": round(deadline, 6),
            "completion": round(completion, 6),
            "miss_ms": round((completion - deadline) * 1000.0, 3),
            **details,
        }
        self._record_event("DEADLINE_MISS", payload)

    def log_queue_overflow(self, queue_name: str, occupancy: int, capacity: int):
        self._record_event("QUEUE_OVERFLOW", {
            "queue_name": queue_name,
            "occupancy": occupancy,
            "capacity": capacity,
        })

    def log_resource_measurement(self, cpu_pct: float, memory_mb: float):
        self._record_event("RESOURCE_MEASUREMENT", {
            "cpu_utilization_pct": round(cpu_pct, 2),
            "memory_resident_mb": round(memory_mb, 2),
        })

    def get_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if limit:
                return [e.to_dict() for e in self._events[-limit:]]
            return [e.to_dict() for e in self._events]

    def close(self):
        with self._lock:
            if getattr(self, "_is_closed", False):
                return
            self._is_closed = True
            try:
                if self._file_handle is not None and not self._file_handle.closed:
                    self._file_handle.flush()
                    self._file_handle.close()
            except Exception as e:
                logger.warning(f"Error closing structured log file: {e}")
            logger.info(f"[STRUCTURED LOGGER] Session log closed ({self._counter} events recorded).")
