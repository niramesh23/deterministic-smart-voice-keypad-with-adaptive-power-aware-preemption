"""
src/rts/__init__.py
-------------------
Subsystem: Real-Time Scheduling & Timing Analysis

Exposes:
- RealTimeJob, JobType, JobStatus, and job factory builders.
- RealTimeScheduler, SchedulingPolicy, SchedulingDecision, LateJobStrategy.
- RealTimeMonitor, DistributionStats.
"""

from src.rts.tasks import (
    JobType,
    JobStatus,
    RealTimeJob,
    create_audio_block_job,
    create_feature_extraction_job,
    create_keyword_inference_job,
    create_monitoring_job,
)
from src.rts.scheduler import (
    SchedulingPolicy,
    SchedulingDecision,
    LateJobStrategy,
    RealTimeScheduler,
)
from src.rts.monitor import (
    DistributionStats,
    RealTimeMonitor,
)
from src.rts.adaptive import (
    ControllerState,
    AdaptationAction,
    AdaptationDecision,
    AdaptiveControllerConfig,
    AdaptiveProcessingController,
)

__all__ = [
    "JobType",
    "JobStatus",
    "RealTimeJob",
    "create_audio_block_job",
    "create_feature_extraction_job",
    "create_keyword_inference_job",
    "create_monitoring_job",
    "SchedulingPolicy",
    "SchedulingDecision",
    "LateJobStrategy",
    "RealTimeScheduler",
    "DistributionStats",
    "RealTimeMonitor",
    "ControllerState",
    "AdaptationAction",
    "AdaptationDecision",
    "AdaptiveControllerConfig",
    "AdaptiveProcessingController",
]
