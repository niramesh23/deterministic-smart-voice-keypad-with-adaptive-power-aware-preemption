"""
src/dashboard Subsystem
=======================
Local Monitoring Dashboard and UI Snapshot Infrastructure.
"""

from src.dashboard.server import DashboardServer, DashboardStatusProvider
from src.dashboard.snapshot import render_dashboard_snapshot

__all__ = [
    "DashboardServer",
    "DashboardStatusProvider",
    "render_dashboard_snapshot",
]

