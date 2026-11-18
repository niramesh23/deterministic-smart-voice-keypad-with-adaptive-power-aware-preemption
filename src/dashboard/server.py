"""
src/dashboard/server.py
-----------------------
Lightweight Local Monitoring Dashboard Server.
Zero-external-dependency HTTP server utilizing Python's standard library http.server.

Features:
- Serves responsive HTML5/CSS/JavaScript web dashboard on localhost.
- Live telemetry polling endpoint (/api/status) every 200 ms.
- Command history (/api/commands) and rejection audit (/api/rejected) endpoints.
- Thread-safe, non-blocking status snapshot provider.
- Clean shutdown and interrupt handling.
"""

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import hmac
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import logging
import os
import pathlib
import secrets
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.parse

from src.security.user_manager import UserManager, UserRecord, UserRole, UserStatus
from src.security.audit_logger import AuditLogger, AuditEventType, AuditResult, AuditSource
from src.analytics.access_types import AccessFilter, AccessReport

logger = logging.getLogger("rtos.dashboard")


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]


# Pre-bind Win32 API functions once at module load to avoid per-call overhead
try:
    _K32GetProcessMemoryInfo = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    _K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    _K32GetProcessMemoryInfo.restype = wintypes.BOOL
    _CurrentProcessHandle = ctypes.windll.kernel32.GetCurrentProcess()
except Exception:
    _K32GetProcessMemoryInfo = None
    _CurrentProcessHandle = None

_cached_memory_rss_mb: float = 0.0
_last_memory_check_time: float = 0.0


def get_process_memory_mb() -> float:
    """Retrieves current process resident set memory (RSS) in megabytes on Windows (cached with 1s TTL)."""
    global _cached_memory_rss_mb, _last_memory_check_time
    now = time.perf_counter()
    if now - _last_memory_check_time < 1.0 and _cached_memory_rss_mb > 0.0:
        return _cached_memory_rss_mb

    if _K32GetProcessMemoryInfo is not None and _CurrentProcessHandle is not None:
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if _K32GetProcessMemoryInfo(_CurrentProcessHandle, ctypes.byref(counters), counters.cb):
                _cached_memory_rss_mb = round(counters.WorkingSetSize / (1024.0 * 1024.0), 2)
                _last_memory_check_time = now
                return _cached_memory_rss_mb
        except Exception:
            pass
    return _cached_memory_rss_mb


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RTOS Voice Recognition Dashboard</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-main: #c9d1d9;
            --text-muted: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-orange: #d29922;
            --accent-red: #f85149;
            --accent-purple: #bc8cff;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); padding: 20px; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 16px; margin-bottom: 24px; }
        .title-group h1 { font-size: 20px; font-weight: 600; color: #fff; }
        .title-group p { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
        .badge { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }
        .badge-live { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-idle { background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
        .badge-overload { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }
        .badge-locked { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }
        .badge-unlocked { background: rgba(63, 185, 80, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-entry { background: rgba(88, 166, 255, 0.2); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
        .badge-lockout { background: rgba(248, 81, 73, 0.3); color: #ff7b72; border: 1px solid #ff7b72; font-weight: bold; }
        .badge-success { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-failure { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }
        .badge-warning { background: rgba(210, 153, 34, 0.15); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
        .badge-info { background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
        .badge-admin { background: rgba(188, 140, 255, 0.15); color: var(--accent-purple); border: 1px solid var(--accent-purple); }
        .badge-regular { background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
        .badge-guest { background: rgba(210, 153, 34, 0.15); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
        .badge-active { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-disabled { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }
        .badge-expired { background: rgba(139, 148, 158, 0.2); color: var(--text-muted); border: 1px solid var(--text-muted); }
        .badge-healthy { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-degraded { background: rgba(210, 153, 34, 0.15); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
        .badge-recovering { background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
        .badge-fault { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }
        select { background: #21262d; color: var(--text-main); border: 1px solid var(--border-color); border-radius: 6px; padding: 4px 8px; font-size: 12px; }
        button { background: #21262d; color: var(--text-main); border: 1px solid var(--border-color); border-radius: 6px; padding: 4px 10px; font-size: 12px; cursor: pointer; }
        button:hover { background: #30363d; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; }
        .card-header { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
        .metric-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(48, 54, 61, 0.5); font-size: 13px; }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: var(--text-muted); }
        .metric-val { font-weight: 600; color: #fff; }
        .highlight-large { font-size: 28px; font-weight: 700; color: var(--accent-green); margin: 8px 0; }
        .table-container { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; margin-top: 16px; overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid var(--border-color); }
        th { color: var(--text-muted); font-weight: 600; }
        .status-confirmed { color: var(--accent-green); font-weight: 600; }
        .status-rejected { color: var(--accent-orange); font-weight: 600; }

        :focus-visible { outline: 3px solid #ffcc00 !important; outline-offset: 3px !important; }
        .skip-link {
            position: absolute;
            top: -60px;
            left: 10px;
            background: #ffcc00;
            color: #000;
            padding: 8px 16px;
            font-weight: 700;
            border-radius: 4px;
            z-index: 10000;
            text-decoration: none;
            transition: top 0.2s;
        }
        .skip-link:focus { top: 10px; }

        /* High-Contrast Mode (WCAG AAA Compliant) */
        body.high-contrast {
            --bg-color: #000000 !important;
            --card-bg: #121212 !important;
            --border-color: #ffffff !important;
            --text-main: #ffffff !important;
            --text-muted: #ffff00 !important;
            --accent-blue: #00ffff !important;
            --accent-green: #00ff66 !important;
            --accent-orange: #ffea00 !important;
            --accent-red: #ff3333 !important;
            --accent-purple: #ff77ff !important;
        }
        body.high-contrast .card,
        body.high-contrast .table-container {
            border: 2px solid #ffffff !important;
            background: #121212 !important;
        }
        body.high-contrast button,
        body.high-contrast select,
        body.high-contrast input {
            border: 2px solid #ffff00 !important;
            background: #000000 !important;
            color: #ffffff !important;
        }
        body.high-contrast button:hover,
        body.high-contrast button:focus {
            background: #ffff00 !important;
            color: #000000 !important;
        }
        body.high-contrast .badge {
            border: 2px solid #ffffff !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            background: #000000 !important;
        }
        body.high-contrast .highlight-large {
            color: #ffff00 !important;
            text-shadow: none !important;
        }

        /* Large Text Mode (+25% Font Scaling) */
        body.large-text {
            font-size: 17px !important;
        }
        body.large-text h1 { font-size: 26px !important; }
        body.large-text h3 { font-size: 20px !important; }
        body.large-text .card-header { font-size: 16px !important; }
        body.large-text .metric-row { font-size: 16px !important; padding: 10px 0 !important; }
        body.large-text .highlight-large { font-size: 34px !important; }
        body.large-text button,
        body.large-text select,
        body.large-text input {
            font-size: 15px !important;
            padding: 8px 14px !important;
            min-height: 40px !important;
        }
        body.large-text .badge {
            font-size: 14px !important;
            padding: 6px 12px !important;
        }
        body.large-text table {
            font-size: 15px !important;
        }
        body.large-text th,
        body.large-text td {
            padding: 12px 10px !important;
        }

        /* Reduced Visual Complexity */
        body.reduced-complexity * {
            animation: none !important;
            transition: none !important;
            box-shadow: none !important;
            text-shadow: none !important;
            transform: none !important;
        }
    </style>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <header>
        <div class="title-group">
            <h1>Real-Time Voice Recognition & RTOS Monitor</h1>
            <p>Closed-loop Adaptive Scheduling & Keyword Spotting Dashboard</p>
        </div>
        <div>
            <span id="connBadge" class="badge badge-live">● ENGINE LIVE</span>
            <span id="updateTime" style="font-size: 12px; color: var(--text-muted); margin-left: 12px;">Live</span>
        </div>
    </header>

    <main id="main-content">
        <!-- Accessibility & Usability Toolbar -->
        <section id="accessibility-panel" class="card" aria-label="Accessibility and Usability Settings" style="margin-bottom: 24px;">
            <div class="card-header">
                <span style="display: flex; align-items: center; gap: 8px;">
                    <span aria-hidden="true">♿</span>
                    <strong>Accessibility & Usability Settings</strong>
                </span>
                <span id="a11yStatusBadge" class="badge badge-info" role="status" aria-live="polite">[A11Y READY]</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; align-items: center;">
                <div class="a11y-control-group">
                    <label for="toggleLargeText" style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                        <input type="checkbox" id="toggleLargeText" onchange="applyA11ySettingsFromUI()" aria-label="Enable large text mode for enhanced readability">
                        <span><strong>Large Text Mode</strong> (Aa +25%)</span>
                    </label>
                </div>
                <div class="a11y-control-group">
                    <label for="toggleHighContrast" style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                        <input type="checkbox" id="toggleHighContrast" onchange="applyA11ySettingsFromUI()" aria-label="Enable WCAG AAA high contrast color theme">
                        <span><strong>High-Contrast Mode</strong> (🌓 WCAG AAA)</span>
                    </label>
                </div>
                <div class="a11y-control-group">
                    <label for="toggleClearIndicators" style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                        <input type="checkbox" id="toggleClearIndicators" checked onchange="applyA11ySettingsFromUI()" aria-label="Enable explicit text labels and non-color shape indicators for status">
                        <span><strong>Clear Status Badges</strong> (Explicit Labels)</span>
                    </label>
                </div>
                <div class="a11y-control-group">
                    <label for="toggleSpokenFeedback" style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                        <input type="checkbox" id="toggleSpokenFeedback" checked onchange="applyA11ySettingsFromUI()" aria-label="Enable spoken voice announcements for system actions">
                        <span><strong>Spoken Voice Feedback</strong> (🔊 Speech)</span>
                    </label>
                </div>
                <div class="a11y-control-group">
                    <label for="toggleAudioConfirmation" style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                        <input type="checkbox" id="toggleAudioConfirmation" checked onchange="applyA11ySettingsFromUI()" aria-label="Enable audio confirmation tones and chimes">
                        <span><strong>Audio Confirmation</strong> (🔔 Cues)</span>
                    </label>
                </div>
                <div class="a11y-control-group">
                    <label for="toggleReducedComplexity" style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                        <input type="checkbox" id="toggleReducedComplexity" onchange="applyA11ySettingsFromUI()" aria-label="Reduce visual motion and disable animations">
                        <span><strong>Reduced Complexity</strong> (⚡ No Motion)</span>
                    </label>
                </div>
                <div class="a11y-control-group">
                    <label for="inputPinTimeout" style="display: block; font-size: 12px; margin-bottom: 4px;">
                        <strong>PIN Inactivity Timeout:</strong> <span id="pinTimeoutDisplay" style="color: var(--accent-blue); font-weight: bold;">45.0s</span>
                    </label>
                    <input type="range" id="inputPinTimeout" min="5" max="60" step="1" value="45" oninput="updatePinTimeoutDisplay(this.value)" onchange="applyA11ySettingsFromUI()" style="width: 100%; cursor: pointer;" aria-label="Adjust PIN entry inactivity timeout between 5 and 60 seconds">
                </div>
                <div class="a11y-control-group">
                    <label for="inputVoiceVolume" style="display: block; font-size: 12px; margin-bottom: 4px;">
                        <strong>Voice Feedback Volume:</strong> <span id="voiceVolumeDisplay" style="color: var(--accent-green); font-weight: bold;">100%</span>
                    </label>
                    <input type="range" id="inputVoiceVolume" min="0" max="100" step="5" value="100" oninput="updateVolumeDisplay(this.value)" onchange="applyA11ySettingsFromUI()" style="width: 100%; cursor: pointer;" aria-label="Adjust spoken voice feedback volume from 0 to 100 percent">
                </div>
            </div>
            <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; gap: 8px;">
                    <button type="button" onclick="saveA11ySettingsToServer()" style="background: var(--accent-blue); color: #fff; font-weight: 600; padding: 6px 14px;" aria-label="Save current accessibility preferences">Save Preferences</button>
                    <button type="button" onclick="resetA11ySettingsToDefaults()" style="padding: 6px 12px;" aria-label="Reset accessibility preferences to default settings">Reset Defaults</button>
                    <button type="button" onclick="testSpokenFeedback()" style="background: #21262d; border: 1px solid var(--border-color); padding: 6px 12px;" aria-label="Test spoken feedback">🔊 Test Speech</button>
                </div>
                <span id="a11yFeedbackMsg" style="font-size: 12px;" role="status" aria-live="polite"></span>
            </div>
        </section>

    <div class="grid">
        <!-- 1. Microphone & Ingress -->
        <div class="card">
            <div class="card-header">Microphone & Audio Pipeline <span id="micStateBadge" class="badge badge-live">ACTIVE</span></div>
            <div class="metric-row"><span class="metric-label">Target Device</span><span id="micDevice" class="metric-val">Loading...</span></div>
            <div class="metric-row"><span class="metric-label">Sample Rate</span><span id="sampleRate" class="metric-val">16,000 Hz (Mono)</span></div>
            <div class="metric-row"><span class="metric-label">Queue Occupancy</span><span id="queueOcc" class="metric-val">0 / 100 blocks</span></div>
            <div class="metric-row"><span class="metric-label">Dropped Audio Blocks</span><span id="audioDrops" class="metric-val">0 blocks</span></div>
            <div class="metric-row"><span class="metric-label">Noise Floor RMS</span><span id="noiseFloor" class="metric-val">0.000000</span></div>
        </div>

        <!-- 2. Command Finite-State Machine -->
        <div class="card">
            <div class="card-header">Command FSM State <span id="fsmStateBadge" class="badge badge-live">LISTENING (Ready)</span></div>
            <div style="text-align: center; margin: 12px 0;">
                <div style="font-size: 12px; color: var(--text-muted);">LATEST CONFIRMED COMMAND</div>
                <div id="latestWord" class="highlight-large">---</div>
                <div id="latestConf" style="font-size: 13px; color: var(--text-muted);">Confidence: N/A</div>
            </div>
            <div class="metric-row"><span class="metric-label">Total Confirmed</span><span id="totalConfirmed" class="metric-val">0</span></div>
            <div class="metric-row"><span class="metric-label">Total Rejected / Suppressed</span><span id="totalRejected" class="metric-val">0</span></div>
        </div>

        <!-- 3. Smart Access Control & Simulated Lock -->
        <div class="card">
            <div class="card-header">Smart Access Control <span id="lockBadge" class="badge badge-locked">LOCKED</span></div>
            <div style="text-align: center; margin: 8px 0;">
                <div style="font-size: 11px; color: var(--text-muted);">SIMULATED LOCK BARRIER</div>
                <div id="lockStateHighlight" class="highlight-large" style="color: var(--accent-red);">LOCKED</div>
                <div id="pinMaskedDisplay" style="font-size: 16px; letter-spacing: 5px; color: var(--accent-blue); font-weight: bold; margin-bottom: 4px;">- - - -</div>
            </div>
            <div class="metric-row"><span class="metric-label">PIN Buffer Progress</span><span id="pinProgress" class="metric-val">0 / 4 digits</span></div>
            <div class="metric-row"><span class="metric-label">Failed Attempts</span><span id="failedAttempts" class="metric-val">0 / 3</span></div>
            <div class="metric-row"><span class="metric-label">Active Countdown</span><span id="activeTimer" class="metric-val">None</span></div>
            <div class="metric-row"><span class="metric-label">Last Access Result</span><span id="lastAccessResult" class="metric-val">NONE</span></div>
            <div class="metric-row"><span class="metric-label">Operational Status</span><span id="lockOpStatus" class="metric-val">LOCKED (Ready for PIN)</span></div>
            <div class="metric-row"><span class="metric-label">Active Authenticated User</span><span id="authenticatedUser" class="metric-val" style="color: var(--accent-purple);">None (Locked)</span></div>
        </div>

        <!-- 3b. Emergency Interlock & STOP Status -->
        <div class="card" id="emergencyCard" style="border-color: var(--border-color);">
            <div class="card-header">Emergency Interlock <span id="emergencyBadge" class="badge badge-live">IDLE</span></div>
            <div style="text-align: center; margin: 8px 0;">
                <div style="font-size: 11px; color: var(--text-muted);">INTERLOCK STATUS</div>
                <div id="emergencyStatusHighlight" class="highlight-large" style="color: var(--accent-green); font-size: 22px;">SYSTEM NORMAL</div>
            </div>
            <div class="metric-row"><span class="metric-label">Last STOP Event</span><span id="emergencyLastEvent" class="metric-val">None</span></div>
            <div class="metric-row"><span class="metric-label">Event Timestamp</span><span id="emergencyTimestamp" class="metric-val">---</span></div>
            <div class="metric-row"><span class="metric-label">Event Source</span><span id="emergencySource" class="metric-val">---</span></div>
            <div class="metric-row"><span class="metric-label">Pending Operation Cancelled</span><span id="emergencyPendingCancelled" class="metric-val" style="color: var(--text-muted);">NO</span></div>
            <div class="metric-row"><span class="metric-label">Lock State at Interlock</span><span id="emergencyLockState" class="metric-val">LOCKED</span></div>
            <div class="metric-row"><span class="metric-label">Total Emergency Interventions</span><span id="emergencyTotalCount" class="metric-val">0 accepted (0 rejected)</span></div>
        </div>

        <!-- 3. Latency Breakdown -->
        <div class="card">
            <div class="card-header">Real-Time Timing Breakdown</div>
            <div class="metric-row"><span class="metric-label">Processing Reaction Latency</span><span id="reactionLatency" class="metric-val">N/A</span></div>
            <div class="metric-row"><span class="metric-label">Total Acoustic Latency</span><span id="acousticLatency" class="metric-val">N/A</span></div>
            <div class="metric-row"><span class="metric-label">Model Forward Pass</span><span id="modelLatency" class="metric-val">N/A</span></div>
            <div class="metric-row"><span class="metric-label">Log-Mel Feature Extraction</span><span id="featLatency" class="metric-val">N/A</span></div>
            <div class="metric-row"><span class="metric-label">Scheduler Queue Delay</span><span id="queueDelay" class="metric-val">N/A</span></div>
        </div>

        <!-- 4. RTOS Task Scheduler -->
        <div class="card">
            <div class="card-header">RTOS Scheduler <span id="policyBadge" class="badge badge-live">EDF</span></div>
            <div class="metric-row"><span class="metric-label">Active Policy</span><span id="policyName" class="metric-val">EDF (Deadline-Aware)</span></div>
            <div class="metric-row"><span class="metric-label">Ready Queue Size</span><span id="readyQueue" class="metric-val">0 jobs</span></div>
            <div class="metric-row"><span class="metric-label">Jobs Executed</span><span id="jobsExecuted" class="metric-val">0</span></div>
            <div class="metric-row"><span class="metric-label">Deadline Misses</span><span id="deadlineMisses" class="metric-val">0</span></div>
            <div class="metric-row"><span class="metric-label">Deadline Miss Ratio (DMR)</span><span id="dmrPct" class="metric-val">0.00%</span></div>
        </div>

        <!-- 5. Adaptive Controller -->
        <div class="card">
            <div class="card-header">Adaptive Controller <span id="adaptiveStateBadge" class="badge badge-idle">NORMAL</span></div>
            <div class="metric-row"><span class="metric-label">Active State</span><span id="ctrlState" class="metric-val">NORMAL</span></div>
            <div class="metric-row"><span class="metric-label">State Transitions</span><span id="stateTransitions" class="metric-val">0</span></div>
            <div class="metric-row"><span class="metric-label">Stale Jobs Pruned</span><span id="stalePruned" class="metric-val">0</span></div>
            <div class="metric-row"><span class="metric-label">Inferences Suppressed</span><span id="infSuppressed" class="metric-val">0</span></div>
            <div class="metric-row"><span class="metric-label">Moving Avg Queue Delay</span><span id="avgDelay" class="metric-val">0.00 ms</span></div>
        </div>

        <!-- 6. Host Hardware Resources -->
        <div class="card">
            <div class="card-header">Host Resources (Windows 11)</div>
            <div class="metric-row"><span class="metric-label">Process CPU Utilization</span><span id="cpuUsage" class="metric-val">0.0%</span></div>
            <div class="metric-row"><span class="metric-label">Resident Memory (RSS)</span><span id="memUsage" class="metric-val">0.0 MB</span></div>
            <div class="metric-row"><span class="metric-label">Execution Architecture</span><span class="metric-val">Python 3.12 (PyTorch CPU)</span></div>
            <div class="metric-row"><span class="metric-label">Feature Vector Shape</span><span class="metric-val">(1, 1, 49, 40)</span></div>
        </div>

        <!-- 7. System Health & Fault Monitor -->
        <div class="card" id="healthCard">
            <div class="card-header">System Health & Fault Monitor <span id="systemHealthBadge" class="badge badge-healthy">HEALTHY</span></div>
            <div style="text-align: center; margin: 8px 0;">
                <div style="font-size: 11px; color: var(--text-muted);">OVERALL SYSTEM HEALTH</div>
                <div id="systemHealthHighlight" class="highlight-large" style="color: var(--accent-green); font-size: 22px;">HEALTHY</div>
            </div>
            <div class="metric-row"><span class="metric-label">Microphone Status</span><span id="healthMicStatus" class="metric-val" style="color: var(--accent-green);">OPERATIONAL</span></div>
            <div class="metric-row"><span class="metric-label">Audio Queue Status</span><span id="healthQueueStatus" class="metric-val">NORMAL</span></div>
            <div class="metric-row"><span class="metric-label">Inference Engine</span><span id="healthInfStatus" class="metric-val" style="color: var(--accent-green);">OPERATIONAL</span></div>
            <div class="metric-row"><span class="metric-label">Scheduler Status</span><span id="healthSchedStatus" class="metric-val">NORMAL</span></div>
            <div class="metric-row"><span class="metric-label">Database Subsystem</span><span id="healthDbStatus" class="metric-val" style="color: var(--accent-green);">OPERATIONAL</span></div>
            <div class="metric-row"><span class="metric-label">Dashboard Service</span><span id="healthDashStatus" class="metric-val" style="color: var(--accent-green);">OPERATIONAL</span></div>
            <div class="metric-row"><span class="metric-label">Active Faults</span><span id="healthActiveFaults" class="metric-val">0 active (0 total)</span></div>
            <div class="metric-row"><span class="metric-label">Recovery Events</span><span id="healthRecoveryEvents" class="metric-val">0 succeeded (0 failed)</span></div>
        </div>

        <!-- 8. Real-Time Performance Analytics -->
        <div class="card" id="analyticsCard" style="grid-column: span 2;">
            <div class="card-header">Real-Time Performance Analytics <span id="analyticsPolicyBadge" class="badge badge-live">EDF SCHEDULER</span></div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 10px 0;">
                <div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px; font-weight: bold;">INFERENCE & DSP TIMINGS</div>
                    <div class="metric-row"><span class="metric-label">Inference Mean / Median</span><span id="analyticsInferMean" class="metric-val">0.0 / 0.0 ms</span></div>
                    <div class="metric-row"><span class="metric-label">Inference P95 / P99</span><span id="analyticsInferP95" class="metric-val" style="color: var(--accent-orange);">0.0 / 0.0 ms</span></div>
                    <div class="metric-row"><span class="metric-label">Inference Max Latency</span><span id="analyticsInferMax" class="metric-val">0.0 ms</span></div>
                    <div class="metric-row"><span class="metric-label">Inference Throughput</span><span id="analyticsThroughput" class="metric-val" style="color: var(--accent-green);">0.0 IPS</span></div>
                    <div class="metric-row"><span class="metric-label">DSP Feature Extraction</span><span id="analyticsDspMean" class="metric-val">0.0 ms</span></div>
                </div>
                <div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px; font-weight: bold;">SCHEDULING & END-TO-END</div>
                    <div class="metric-row"><span class="metric-label">Task Response P95</span><span id="analyticsRespP95" class="metric-val">0.0 ms</span></div>
                    <div class="metric-row"><span class="metric-label">Deadline Miss Ratio</span><span id="analyticsDmr" class="metric-val">0.0%</span></div>
                    <div class="metric-row"><span class="metric-label">Task Utilization (CPU)</span><span id="analyticsUtil" class="metric-val">0.0%</span></div>
                    <div class="metric-row"><span class="metric-label">Audio to Inference P95</span><span id="analyticsAudioToInfer" class="metric-val">0.0 ms</span></div>
                    <div class="metric-row"><span class="metric-label">Command to Access P95</span><span id="analyticsCmdToAccess" class="metric-val" style="color: var(--accent-blue);">0.0 ms</span></div>
                </div>
            </div>
            <div style="border-top: 1px solid var(--border-color); padding-top: 8px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="font-size: 11px; color: var(--text-muted);">Process CPU: <strong id="analyticsProcCpu" style="color:#fff;">0.0%</strong> | System CPU: <strong id="analyticsSysCpu" style="color:#fff;">0.0%</strong> | Memory RSS: <strong id="analyticsRss" style="color:#fff;">0.0 MB</strong></span>
                <span id="analyticsBenchmarkCount" style="font-size: 11px; color: var(--text-muted);">0 formal benchmarks recorded</span>
            </div>
        </div>
    </div>

    <!-- Recent Command History Table -->
    <div class="table-container">
        <h3 style="font-size: 14px; margin-bottom: 12px; color: #fff;">Recent Confirmed Commands & Decision Log</h3>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Keyword Command</th>
                    <th>Confidence</th>
                    <th>Margin</th>
                    <th>Reaction Latency</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="historyTableBody">
                <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No commands confirmed yet</td></tr>
            </tbody>
        </table>
    </div>


    <!-- Security Audit Trail Table -->
    <div class="table-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
            <h3 style="font-size: 14px; color: #fff;">Persistent Security Audit Trail</h3>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <label style="font-size: 12px; color: var(--text-muted);">Category:
                    <select id="auditCategory" onchange="fetchAudit()">
                        <option value="">All Categories</option>
                        <option value="AUTH">Authentication</option>
                        <option value="LOCK">Lock State</option>
                        <option value="SECURITY">Security / Access</option>
                        <option value="SYSTEM">System Lifecycle</option>
                    </select>
                </label>
                <label style="font-size: 12px; color: var(--text-muted);">Result:
                    <select id="auditResult" onchange="fetchAudit()">
                        <option value="">All Results</option>
                        <option value="SUCCESS">SUCCESS</option>
                        <option value="FAILURE">FAILURE</option>
                        <option value="WARNING">WARNING</option>
                        <option value="INFO">INFO</option>
                    </select>
                </label>
                <label style="font-size: 12px; color: var(--text-muted);">Limit:
                    <select id="auditLimit" onchange="fetchAudit()">
                        <option value="15">15</option>
                        <option value="30" selected>30</option>
                        <option value="50">50</option>
                        <option value="100">100</option>
                    </select>
                </label>
                <button onclick="fetchAudit()">Refresh</button>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    <th style="width: 50px;">ID</th>
                    <th style="width: 170px;">Timestamp (UTC)</th>
                    <th style="width: 160px;">Event Type</th>
                    <th style="width: 95px;">Outcome</th>
                    <th style="width: 105px;">Lock State</th>
                    <th style="width: 120px;">Source</th>
                    <th>Reason / Details</th>
                </tr>
            </thead>
            <tbody id="auditTableBody">
                <tr><td colspan="7" style="text-align: center; color: var(--text-muted);">Loading audit records...</td></tr>
            </tbody>
        </table>
    </div>


    <!-- User Management & Role-Based Access Control Table -->
    <div class="table-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
            <div>
                <h3 style="font-size: 14px; color: #fff;">User Management & Role-Based Access Control</h3>
                <p style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">Manage user accounts, assign roles (ADMIN, REGULAR_USER, GUEST), and configure PIN credentials.</p>
            </div>
        </div>

        <!-- Admin Session Authentication Card -->
        <div id="adminAuthSection" style="background: #21262d; border: 1px solid var(--border-color); padding: 12px; border-radius: 6px; margin-bottom: 14px;">
            <div id="adminLoginForm" style="display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end;">
                <div>
                    <label style="font-size: 11px; color: var(--text-muted); display: block;">Admin User ID</label>
                    <input type="text" id="adminLoginId" placeholder="admin_01" style="background: #0d1117; color: #fff; border: 1px solid var(--border-color); border-radius: 4px; padding: 4px 8px; font-size: 12px; width: 110px;">
                </div>
                <div>
                    <label style="font-size: 11px; color: var(--text-muted); display: block;">Admin PIN</label>
                    <input type="password" id="adminLoginPin" maxlength="4" placeholder="••••" style="background: #0d1117; color: #fff; border: 1px solid var(--border-color); border-radius: 4px; padding: 4px 8px; font-size: 12px; width: 90px;">
                </div>
                <button onclick="handleAdminLogin()" style="background: var(--accent-blue); color: #fff; border: none; padding: 5px 14px; border-radius: 4px; font-size: 12px; cursor: pointer; font-weight: 600;">Log In as Admin</button>
                <span id="adminLoginFeedback" style="font-size: 12px; margin-left: 10px;"></span>
            </div>
            <div id="adminActiveSession" style="display: none; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span class="badge badge-admin">AUTHENTICATED ADMIN</span>
                    <span id="adminSessionUsername" style="font-weight: 600; color: #fff; margin-left: 8px;">Administrator</span>
                    <span style="font-size: 12px; color: var(--text-muted); margin-left: 8px;">(Session token active)</span>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <button onclick="handleEmergencyStopClick()" style="background: var(--accent-red); color: #fff; border: 1px solid #ff7b72; font-weight: bold; padding: 4px 12px; cursor: pointer;">🚨 EMERGENCY STOP</button>
                    <button onclick="fetchUsers()">Refresh Users</button>
                    <button onclick="handleAdminLogout()" style="color: var(--accent-red);">Log Out</button>
                </div>
            </div>
        </div>

        <div id="adminControlsContainer" style="display: none;">
            <!-- Add User Form -->
            <div style="background: #21262d; padding: 12px; border-radius: 6px; margin-bottom: 14px; display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end;">
                <div>
                    <label style="font-size: 11px; color: var(--text-muted); display: block;">User ID</label>
                    <input type="text" id="newUserId" placeholder="e.g. user_02" style="background: #0d1117; color: #fff; border: 1px solid var(--border-color); border-radius: 4px; padding: 4px 8px; font-size: 12px;">
                </div>
                <div>
                    <label style="font-size: 11px; color: var(--text-muted); display: block;">Full Name</label>
                    <input type="text" id="newUsername" placeholder="e.g. Alice Smith" style="background: #0d1117; color: #fff; border: 1px solid var(--border-color); border-radius: 4px; padding: 4px 8px; font-size: 12px;">
                </div>
                <div>
                    <label style="font-size: 11px; color: var(--text-muted); display: block;">PIN (4 digits 1-9)</label>
                    <input type="password" id="newUserPin" maxlength="4" placeholder="••••" style="background: #0d1117; color: #fff; border: 1px solid var(--border-color); border-radius: 4px; padding: 4px 8px; font-size: 12px; width: 90px;">
                </div>
                <div>
                    <label style="font-size: 11px; color: var(--text-muted); display: block;">Role</label>
                    <select id="newUserRole" style="background: #0d1117; color: #fff; border: 1px solid var(--border-color); border-radius: 4px; padding: 4px 8px; font-size: 12px;">
                        <option value="REGULAR_USER">REGULAR_USER</option>
                        <option value="GUEST">GUEST</option>
                        <option value="ADMIN">ADMIN</option>
                    </select>
                </div>
                <button onclick="handleCreateUser()" style="background: #238636; color: #fff; border: none; padding: 5px 12px; border-radius: 4px; font-size: 12px; cursor: pointer;">Add User</button>
                <span id="userActionFeedback" style="font-size: 12px; margin-left: 10px;"></span>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th style="width: 110px;">User ID</th>
                    <th style="width: 140px;">Full Name</th>
                    <th style="width: 120px;">Role</th>
                    <th style="width: 100px;">Status</th>
                    <th style="width: 170px;">Created At (UTC)</th>
                    <th style="width: 170px;">Expires At (UTC)</th>
                    <th style="width: 170px;">Last Access</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody id="userTableBody">
                <tr><td colspan="8" style="text-align: center; color: var(--text-muted);">Loading user records...</td></tr>
            </tbody>
        </table>
    </div>


    <!-- Active System Faults Table -->
    <div class="table-container">
        <h3 style="font-size: 14px; margin-bottom: 12px; color: #fff;">Active Operational Faults & Diagnostics</h3>
        <table>
            <thead>
                <tr>
                    <th style="width: 70px;">Fault ID</th>
                    <th style="width: 160px;">Timestamp</th>
                    <th style="width: 120px;">Component</th>
                    <th style="width: 180px;">Fault Type</th>
                    <th style="width: 100px;">Severity</th>
                    <th>Diagnostic Description</th>
                </tr>
            </thead>
            <tbody id="faultsTableBody">
                <tr><td colspan="6" style="text-align: center; color: var(--accent-green);">No active faults detected. System operating normally.</td></tr>
            </tbody>
        </table>
    </div>


    <!-- Recent Component Recoveries Table -->
    <div class="table-container">
        <h3 style="font-size: 14px; margin-bottom: 12px; color: #fff;">Subsystem Recovery Events & Verification Trail</h3>
        <table>
            <thead>
                <tr>
                    <th style="width: 70px;">Rec ID</th>
                    <th style="width: 160px;">Timestamp</th>
                    <th style="width: 120px;">Component</th>
                    <th style="width: 220px;">Action Taken</th>
                    <th style="width: 100px;">Outcome</th>
                    <th>Verified Healthy</th>
                </tr>
            </thead>
            <tbody id="recoveryTableBody">
                <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No recovery events recorded.</td></tr>
            </tbody>
        </table>
    </div>

    <!-- 8b. Real-Time Scheduling Benchmark Comparison Table -->
    <div class="table-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
            <h3 style="font-size: 14px; color: #fff;">Performance Benchmark Experiments (FIFO vs RM vs EDF)</h3>
            <span style="font-size: 12px; color: var(--text-muted);">Empirical reproducible workload results saved to <code style="color: var(--accent-blue);">results/benchmarks/</code></span>
        </div>
        <table>
            <thead>
                <tr>
                    <th style="width: 140px;">Experiment ID</th>
                    <th style="width: 70px;">Policy</th>
                    <th style="width: 90px;">Workload</th>
                    <th style="width: 100px;">Infer P95 (ms)</th>
                    <th style="width: 100px;">Resp P95 (ms)</th>
                    <th style="width: 80px;">DMR (%)</th>
                    <th style="width: 90px;">Throughput</th>
                    <th style="width: 80px;">CPU (%)</th>
                    <th>E2E Access P95</th>
                </tr>
            </thead>
            <tbody id="benchmarkTableBody">
                <tr><td colspan="9" style="text-align: center; color: var(--text-muted);">No benchmark experiments recorded.</td></tr>
            </tbody>
        </table>
    </div>

    <!-- Access Analytics & Report Generation Section -->
    <div class="table-container" id="accessAnalyticsSection">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
            <div>
                <h3 style="font-size: 14px; color: #fff; margin-bottom: 4px;">Access Analytics & Report Generation</h3>
                <span style="font-size: 11px; color: var(--text-muted);">Historical authentication telemetry, security metrics, and multi-format downloadable reports</span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <button onclick="downloadAccessReport('csv')" style="background-color: var(--accent-blue);">Download CSV Report</button>
                <button onclick="downloadAccessReport('json')" style="background-color: var(--card-bg); border: 1px solid var(--accent-blue);">Download JSON Report</button>
            </div>
        </div>

        <!-- Filter Bar -->
        <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; background-color: var(--bg-color); padding: 10px; border-radius: 6px; border: 1px solid var(--border-color);">
            <label style="font-size: 12px; color: var(--text-muted);">From:
                <input type="date" id="accessStartDate" onchange="fetchAccessAnalytics()">
            </label>
            <label style="font-size: 12px; color: var(--text-muted);">To:
                <input type="date" id="accessEndDate" onchange="fetchAccessAnalytics()">
            </label>
            <label style="font-size: 12px; color: var(--text-muted);">User:
                <input type="text" id="accessUserId" placeholder="All Users" style="width: 100px;" onchange="fetchAccessAnalytics()">
            </label>
            <label style="font-size: 12px; color: var(--text-muted);">Role:
                <select id="accessRole" onchange="fetchAccessAnalytics()">
                    <option value="">All Roles</option>
                    <option value="ADMIN">ADMIN</option>
                    <option value="REGULAR_USER">REGULAR_USER</option>
                    <option value="GUEST">GUEST</option>
                </select>
            </label>
            <label style="font-size: 12px; color: var(--text-muted);">Result:
                <select id="accessResult" onchange="fetchAccessAnalytics()">
                    <option value="">All Results</option>
                    <option value="SUCCESS">SUCCESS</option>
                    <option value="FAILURE">FAILURE</option>
                </select>
            </label>
            <button onclick="fetchAccessAnalytics()">Apply Filters</button>
            <button onclick="resetAccessFilters()" style="background-color: transparent; border: 1px solid var(--border-color); color: var(--text-muted);">Reset</button>
        </div>

        <!-- Key Stat Metric Cards -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 16px;">
            <div class="card" style="margin: 0; padding: 12px; text-align: center;">
                <div style="font-size: 11px; color: var(--text-muted);">TOTAL ATTEMPTS</div>
                <div id="accTotalAttempts" style="font-size: 20px; font-weight: bold; color: #fff; margin-top: 4px;">0</div>
            </div>
            <div class="card" style="margin: 0; padding: 12px; text-align: center;">
                <div style="font-size: 11px; color: var(--text-muted);">SUCCESSFUL</div>
                <div id="accSuccessAttempts" style="font-size: 20px; font-weight: bold; color: var(--accent-green); margin-top: 4px;">0</div>
            </div>
            <div class="card" style="margin: 0; padding: 12px; text-align: center;">
                <div style="font-size: 11px; color: var(--text-muted);">FAILED</div>
                <div id="accFailedAttempts" style="font-size: 20px; font-weight: bold; color: var(--accent-red); margin-top: 4px;">0</div>
            </div>
            <div class="card" style="margin: 0; padding: 12px; text-align: center;">
                <div style="font-size: 11px; color: var(--text-muted);">SUCCESS RATE</div>
                <div id="accSuccessRate" style="font-size: 20px; font-weight: bold; color: var(--accent-green); margin-top: 4px;">0.0%</div>
            </div>
            <div class="card" style="margin: 0; padding: 12px; text-align: center;">
                <div style="font-size: 11px; color: var(--text-muted);">DENIAL RATE</div>
                <div id="accDenialRate" style="font-size: 20px; font-weight: bold; color: var(--accent-red); margin-top: 4px;">0.0%</div>
            </div>
            <div class="card" style="margin: 0; padding: 12px; text-align: center;">
                <div style="font-size: 11px; color: var(--text-muted);">LOCKOUTS</div>
                <div id="accLockouts" style="font-size: 20px; font-weight: bold; color: var(--accent-orange); margin-top: 4px;">0</div>
            </div>
            <div class="card" style="margin: 0; padding: 12px; text-align: center;">
                <div style="font-size: 11px; color: var(--text-muted);">EMERGENCY STOPS</div>
                <div id="accEmergencyStops" style="font-size: 20px; font-weight: bold; color: var(--accent-red); margin-top: 4px;">0</div>
            </div>
            <div class="card" style="margin: 0; padding: 12px; text-align: center;">
                <div style="font-size: 11px; color: var(--text-muted);">REPEATED FAILS</div>
                <div id="accRepeatedFails" style="font-size: 20px; font-weight: bold; color: var(--accent-orange); margin-top: 4px;">0</div>
            </div>
        </div>

        <!-- Trend Visualizations Grid (Daily & Hourly) -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
            <div class="card" style="margin: 0; padding: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 12px; font-weight: bold; color: #fff;">DAILY ACCESS TRENDS</span>
                    <span id="accPeakDayBadge" class="badge badge-info" style="font-size: 10px;">Peak: N/A</span>
                </div>
                <div id="dailyTrendsContainer" style="display: flex; flex-direction: column; gap: 6px; max-height: 180px; overflow-y: auto;">
                    <div style="text-align: center; color: var(--text-muted); font-size: 12px; padding: 20px 0;">No access records for selected filters.</div>
                </div>
            </div>
            <div class="card" style="margin: 0; padding: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 12px; font-weight: bold; color: #fff;">HOURLY TRAFFIC DISTRIBUTION (24h UTC)</span>
                    <span id="accPeakHourBadge" class="badge badge-info" style="font-size: 10px;">Peak: N/A</span>
                </div>
                <div id="hourlyTrendsContainer" style="display: flex; align-items: flex-end; gap: 3px; height: 140px; padding-top: 10px;">
                    <!-- 24 hourly vertical bars populated via JS -->
                </div>
            </div>
        </div>

        <!-- User Activity Breakdown Table -->
        <h4 style="font-size: 13px; color: #fff; margin-bottom: 8px;">User Access Activity Breakdown</h4>
        <table>
            <thead>
                <tr>
                    <th>User ID</th>
                    <th>Username</th>
                    <th>Role</th>
                    <th>Total Attempts</th>
                    <th>Success</th>
                    <th>Failure</th>
                    <th>Success Rate</th>
                    <th>Last Access Time (UTC)</th>
                </tr>
            </thead>
            <tbody id="userActivityTableBody">
                <tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No user activity found.</td></tr>
            </tbody>
        </table>
    </div>
    </main>

    <script>
        let a11ySettings = {
            large_text: false,
            high_contrast: false,
            clear_status_indicators: true,
            spoken_feedback: true,
            audio_confirmation: true,
            pin_entry_timeout_s: 45.0,
            voice_feedback_volume: 1.0,
            reduced_visual_complexity: false,
        };
        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                if (!res.ok) {
                    const cBadge = document.getElementById('connBadge');
                    if (cBadge) { cBadge.className = 'badge badge-overload'; cBadge.innerText = '⚠ DISCONNECTED'; }
                    const uTime = document.getElementById('updateTime');
                    if (uTime) uTime.innerText = 'Offline (HTTP ' + res.status + ')';
                    return;
                }
                const data = await res.json();
                
                const cBadge = document.getElementById('connBadge');
                if (cBadge) { cBadge.className = 'badge badge-live'; cBadge.innerText = '● ENGINE LIVE'; }
                const elUpdateTime = document.getElementById('updateTime');
                if (elUpdateTime) elUpdateTime.innerText = 'Updated: ' + (data.timestamp || 'Live');
                const elMic = document.getElementById('micDevice');
                if (elMic) elMic.innerText = (data.microphone && data.microphone.device_name) ? data.microphone.device_name : 'Virtual Stream';
                const elQOcc = document.getElementById('queueOcc');
                if (elQOcc) elQOcc.innerText = (data.microphone ? data.microphone.queue_occupancy : 0) + ' / ' + (data.microphone ? data.microphone.queue_capacity : 100) + ' blocks';
                const elDrops = document.getElementById('audioDrops');
                if (elDrops) elDrops.innerText = (data.microphone ? data.microphone.audio_blocks_dropped : 0) + ' blocks';
                const elNoise = document.getElementById('noiseFloor');
                if (elNoise) elNoise.innerText = (data.dsp && data.dsp.noise_floor_rms !== null ? data.dsp.noise_floor_rms.toFixed(6) : 'N/A');
                
                // Command FSM
                if (data.command_fsm) {
                    const elState = document.getElementById('fsmStateBadge');
                    if (elState) {
                        const st = data.command_fsm.current_state || 'WAITING';
                        if (st === 'WAITING') {
                            elState.innerText = 'LISTENING (Ready)';
                            elState.className = 'badge badge-live';
                        } else if (st === 'SPEECH_DETECTED') {
                            elState.innerText = 'SPEECH DETECTED';
                            elState.className = 'badge badge-entry';
                        } else if (st === 'COMMAND_CONFIRMED') {
                            elState.innerText = 'CONFIRMED';
                            elState.className = 'badge badge-live';
                        } else if (st === 'COOLDOWN') {
                            elState.innerText = 'COOLDOWN';
                            elState.className = 'badge badge-warning';
                        } else {
                            elState.innerText = st;
                            elState.className = 'badge badge-idle';
                        }
                    }
                    const elConf = document.getElementById('totalConfirmed');
                    if (elConf) elConf.innerText = data.command_fsm.total_confirmed || 0;
                    const elRej = document.getElementById('totalRejected');
                    if (elRej) elRej.innerText = data.command_fsm.total_rejected || 0;
                    const elWord = document.getElementById('latestWord');
                    const elLConf = document.getElementById('latestConf');
                    if (data.command_fsm.latest_command) {
                        const dispWord = data.command_fsm.latest_digit ? (data.command_fsm.latest_command + ' (' + data.command_fsm.latest_digit + ')') : data.command_fsm.latest_command;
                        if (elWord) elWord.innerText = '"' + dispWord + '"';
                        if (elLConf) elLConf.innerText = 'Confidence: ' + ((data.command_fsm.latest_confidence || 0) * 100).toFixed(1) + '%';
                    } else {
                        if (elWord) elWord.innerText = '---';
                        if (elLConf) elLConf.innerText = 'Confidence: N/A';
                    }
                }

                // Latency
                if (data.timing) {
                    const elReact = document.getElementById('reactionLatency');
                    if (elReact) elReact.innerText = (data.timing.processing_reaction_latency_ms != null) ? data.timing.processing_reaction_latency_ms.toFixed(2) + ' ms' : 'N/A';
                    const elAcous = document.getElementById('acousticLatency');
                    if (elAcous) elAcous.innerText = (data.timing.total_acoustic_latency_ms != null) ? data.timing.total_acoustic_latency_ms.toFixed(2) + ' ms' : 'N/A';
                    const elMod = document.getElementById('modelLatency');
                    if (elMod) elMod.innerText = (data.timing.model_inference_ms != null) ? data.timing.model_inference_ms.toFixed(2) + ' ms' : 'N/A';
                    const elFeat = document.getElementById('featLatency');
                    if (elFeat) elFeat.innerText = (data.timing.feature_extraction_ms != null) ? data.timing.feature_extraction_ms.toFixed(2) + ' ms' : 'N/A';
                    const elQDel = document.getElementById('queueDelay');
                    if (elQDel) elQDel.innerText = (data.timing.queue_delay_ms != null) ? data.timing.queue_delay_ms.toFixed(2) + ' ms' : 'N/A';
                }

                // Scheduler
                if (data.scheduler) {
                    const elPol = document.getElementById('policyName');
                    if (elPol) elPol.innerText = data.scheduler.policy || 'EDF';
                    const elPolBadge = document.getElementById('policyBadge');
                    if (elPolBadge) elPolBadge.innerText = data.scheduler.policy || 'EDF';
                    const elReady = document.getElementById('readyQueue');
                    if (elReady) elReady.innerText = (data.scheduler.ready_jobs_count || 0) + ' jobs';
                    const elExec = document.getElementById('jobsExecuted');
                    if (elExec) elExec.innerText = data.scheduler.total_executed || 0;
                    const elMiss = document.getElementById('deadlineMisses');
                    if (elMiss) elMiss.innerText = data.scheduler.total_deadline_misses || 0;
                    const elDmr = document.getElementById('dmrPct');
                    if (elDmr) elDmr.innerText = (data.scheduler.deadline_miss_ratio_pct != null ? data.scheduler.deadline_miss_ratio_pct.toFixed(2) : '0.00') + '%';
                }

                // Adaptive Controller
                if (data.adaptive_controller) {
                    const elCtrl = document.getElementById('ctrlState');
                    if (elCtrl) elCtrl.innerText = data.adaptive_controller.current_state || 'NORMAL';
                    const elAdBadge = document.getElementById('adaptiveStateBadge');
                    if (elAdBadge) elAdBadge.innerText = data.adaptive_controller.current_state || 'NORMAL';
                    const elTrans = document.getElementById('stateTransitions');
                    if (elTrans) elTrans.innerText = data.adaptive_controller.total_state_transitions || 0;
                    const elPrune = document.getElementById('stalePruned');
                    if (elPrune) elPrune.innerText = data.adaptive_controller.total_stale_jobs_pruned || 0;
                    const elSupp = document.getElementById('infSuppressed');
                    if (elSupp) elSupp.innerText = data.adaptive_controller.total_inferences_suppressed || 0;
                    const elAvgDel = document.getElementById('avgDelay');
                    if (elAvgDel) elAvgDel.innerText = (data.adaptive_controller.mean_recent_queue_delay_ms != null ? data.adaptive_controller.mean_recent_queue_delay_ms.toFixed(2) : '0.00') + ' ms';
                }

                // Resources
                if (data.resources) {
                    const elCpu = document.getElementById('cpuUsage');
                    if (elCpu) elCpu.innerText = (data.resources.cpu_utilization_pct != null ? data.resources.cpu_utilization_pct.toFixed(1) : '0.0') + '%';
                    const elMem = document.getElementById('memUsage');
                    if (elMem) elMem.innerText = (data.resources.memory_rss_mb != null ? data.resources.memory_rss_mb.toFixed(1) : '0.0') + ' MB';
                }

                // Smart Lock & Access Control
                if (data.lock) {
                    const lState = data.lock.state;
                    const lBadge = document.getElementById('lockBadge');
                    const lHighlight = document.getElementById('lockStateHighlight');
                    
                    let lIcon = '🔒';
                    let lLabel = lState;
                    if (lState === 'UNLOCKED') {
                        lIcon = '🔓';
                        lLabel = 'UNLOCKED';
                        lBadge.className = 'badge badge-unlocked';
                        lHighlight.style.color = 'var(--accent-green)';
                    } else if (lState === 'PIN_ENTRY') {
                        lIcon = '📝';
                        lLabel = 'PIN ENTRY';
                        lBadge.className = 'badge badge-entry';
                        lHighlight.style.color = 'var(--accent-blue)';
                    } else if (lState === 'LOCKOUT') {
                        lIcon = '⛔';
                        lLabel = 'LOCKOUT ACTIVE';
                        lBadge.className = 'badge badge-lockout';
                        lHighlight.style.color = 'var(--accent-red)';
                    } else {
                        lIcon = '🔒';
                        lLabel = 'LOCKED';
                        lBadge.className = 'badge badge-locked';
                        lHighlight.style.color = 'var(--accent-red)';
                    }

                    if (a11ySettings && a11ySettings.clear_status_indicators) {
                        lBadge.innerText = '[' + lIcon + ' ' + lLabel + ']';
                        lHighlight.innerText = lIcon + ' ' + lLabel;
                    } else {
                        lBadge.innerText = lState;
                        lHighlight.innerText = lState.replace('_', ' ');
                    }
                    
                    const entered = data.lock.digits_entered || 0;
                    const totalLen = data.lock.pin_length || 4;
                    let maskedStr = '';
                    for (let i = 0; i < totalLen; i++) {
                        maskedStr += (i < entered ? '● ' : '○ ');
                    }
                    document.getElementById('pinMaskedDisplay').innerText = maskedStr.trim();
                    document.getElementById('pinProgress').innerText = data.lock.pin_progress || (entered + ' / ' + totalLen + ' digits');
                    document.getElementById('failedAttempts').innerText = data.lock.failed_attempts + ' / ' + data.lock.max_failed_attempts;
                    
                    let timerText = 'None';
                    if (data.lock.lockout_remaining_s > 0) {
                        timerText = 'Lockout: ' + data.lock.lockout_remaining_s.toFixed(1) + 's';
                    } else if (data.lock.unlock_remaining_s > 0) {
                        timerText = 'Relock: ' + data.lock.unlock_remaining_s.toFixed(1) + 's';
                    }
                    document.getElementById('activeTimer').innerText = timerText;
                    document.getElementById('lastAccessResult').innerText = data.lock.last_access_result || 'NONE';
                    document.getElementById('lockOpStatus').innerText = data.lock.operational_status || 'LOCKED (Ready for PIN)';
                    if (data.lock.authenticated_username) {
                        document.getElementById('authenticatedUser').innerText = data.lock.authenticated_username + ' (' + (data.lock.authenticated_role || 'USER') + ')';
                    } else {
                        document.getElementById('authenticatedUser').innerText = 'None (' + data.lock.state + ')';
                    }
                }

                // Emergency Interlock
                if (data.emergency) {
                    const em = data.emergency;
                    const emBadge = document.getElementById('emergencyBadge');
                    const emHigh = document.getElementById('emergencyStatusHighlight');
                    const emCard = document.getElementById('emergencyCard');
                    if (em.is_active) {
                        if (emBadge) { emBadge.className = 'badge badge-lockout'; emBadge.innerText = 'TRIGGERED'; }
                        if (emHigh) { emHigh.innerText = 'EMERGENCY TRIGGERED'; emHigh.style.color = 'var(--accent-red)'; }
                        if (emCard) emCard.style.borderColor = 'var(--accent-red)';
                    } else {
                        if (emBadge) { emBadge.className = 'badge badge-live'; emBadge.innerText = 'IDLE'; }
                        if (emHigh) { emHigh.innerText = 'SYSTEM NORMAL'; emHigh.style.color = 'var(--accent-green)'; }
                        if (emCard) emCard.style.borderColor = 'var(--border-color)';
                    }
                    if (em.last_stop_event) {
                        const ev = em.last_stop_event;
                        const elEvent = document.getElementById('emergencyLastEvent');
                        const elTime = document.getElementById('emergencyTimestamp');
                        const elSrc = document.getElementById('emergencySource');
                        const elCanc = document.getElementById('emergencyPendingCancelled');
                        const elState = document.getElementById('emergencyLockState');
                        if (elEvent) elEvent.innerText = ev.status || 'ACCEPTED';
                        if (elTime) elTime.innerText = ev.timestamp || '---';
                        if (elSrc) elSrc.innerText = ev.source || '---';
                        if (elCanc) {
                            elCanc.innerText = ev.pending_cancelled ? 'YES (Cancelled)' : 'NO';
                            elCanc.style.color = ev.pending_cancelled ? 'var(--accent-orange)' : 'var(--text-muted)';
                        }
                        if (elState) elState.innerText = ev.lock_state || 'LOCKED';
                    }
                    const elCount = document.getElementById('emergencyTotalCount');
                    if (elCount) elCount.innerText = `${em.total_accepted} accepted (${em.total_rejected} rejected)`;
                }

                // System Health & Fault Monitor
                if (data.health) {
                    const h = data.health;
                    const hBadge = document.getElementById('systemHealthBadge');
                    const hHigh = document.getElementById('systemHealthHighlight');
                    const hCard = document.getElementById('healthCard');
                    if (hBadge && hHigh) {
                        hHigh.innerText = h.system_health || 'HEALTHY';
                        let hIcon = '✓';
                        if (h.system_health === 'HEALTHY') {
                            hIcon = '✓';
                            hBadge.className = 'badge badge-healthy';
                            hHigh.style.color = 'var(--accent-green)';
                            if (hCard) hCard.style.borderColor = 'var(--border-color)';
                        } else if (h.system_health === 'DEGRADED') {
                            hIcon = '⚠';
                            hBadge.className = 'badge badge-degraded';
                            hHigh.style.color = 'var(--accent-orange)';
                            if (hCard) hCard.style.borderColor = 'var(--accent-orange)';
                        } else if (h.system_health === 'RECOVERING') {
                            hIcon = '🔄';
                            hBadge.className = 'badge badge-recovering';
                            hHigh.style.color = 'var(--accent-blue)';
                            if (hCard) hCard.style.borderColor = 'var(--accent-blue)';
                        } else {
                            hIcon = '✕';
                            hBadge.className = 'badge badge-fault';
                            hHigh.style.color = 'var(--accent-red)';
                            if (hCard) hCard.style.borderColor = 'var(--accent-red)';
                        }
                        if (a11ySettings && a11ySettings.clear_status_indicators) {
                            hBadge.innerText = '[' + hIcon + ' ' + (h.system_health || 'HEALTHY') + ']';
                            hHigh.innerText = hIcon + ' ' + (h.system_health || 'HEALTHY');
                        } else {
                            hBadge.innerText = h.system_health || 'HEALTHY';
                            hHigh.innerText = h.system_health || 'HEALTHY';
                        }
                    }

                    const comps = h.components || {};
                    const setCompStatus = (id, st) => {
                        const el = document.getElementById(id);
                        if (!el) return;
                        el.innerText = st || 'OPERATIONAL';
                        if (st === 'OPERATIONAL') el.style.color = 'var(--accent-green)';
                        else if (st === 'DEGRADED') el.style.color = 'var(--accent-orange)';
                        else if (st === 'RECOVERING') el.style.color = 'var(--accent-blue)';
                        else el.style.color = 'var(--accent-red)';
                    };

                    setCompStatus('healthMicStatus', comps.microphone);
                    setCompStatus('healthQueueStatus', comps.audio_queue);
                    setCompStatus('healthInfStatus', comps.inference);
                    setCompStatus('healthSchedStatus', comps.scheduler);
                    setCompStatus('healthDbStatus', comps.database);
                    setCompStatus('healthDashStatus', comps.dashboard);

                    const elAct = document.getElementById('healthActiveFaults');
                    if (elAct) elAct.innerText = `${h.active_faults_count || 0} active (${h.total_faults_detected || 0} total)`;
                    const elRec = document.getElementById('healthRecoveryEvents');
                    if (elRec) elRec.innerText = `${h.succeeded_recoveries || h.succeeded || 0} succeeded (${h.failed_recoveries || h.failed || 0} failed)`;

                    // Active Faults Table
                    const fBody = document.getElementById('faultsTableBody');
                    if (fBody && h.active_faults) {
                        if (h.active_faults.length === 0) {
                            fBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--accent-green);">No active faults detected. System operating normally.</td></tr>';
                        } else {
                            fBody.innerHTML = '';
                            h.active_faults.forEach(f => {
                                const tr = document.createElement('tr');
                                let sevBadge = 'badge-warning';
                                if (f.severity === 'CRITICAL' || f.severity === 'ERROR') sevBadge = 'badge-failure';
                                else if (f.severity === 'INFO') sevBadge = 'badge-info';
                                tr.innerHTML = `
                                    <td style="color: var(--text-muted); font-size: 11px;">#${f.fault_id}</td>
                                    <td style="font-size: 12px;">${f.iso_time}</td>
                                    <td style="font-weight: 600;">${f.component}</td>
                                    <td><span style="font-family: monospace; font-size: 11px;">${f.fault_type}</span></td>
                                    <td><span class="badge ${sevBadge}">${f.severity}</span></td>
                                    <td style="font-size: 12px;">${f.description}</td>
                                `;
                                fBody.appendChild(tr);
                            });
                        }
                    }

                    // Recoveries Table
                    const rBody = document.getElementById('recoveryTableBody');
                    if (rBody && (h.recent_recoveries || [])) {
                        const recs = h.recent_recoveries || [];
                        if (recs.length === 0) {
                            rBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No recovery events recorded.</td></tr>';
                        } else {
                            rBody.innerHTML = '';
                            recs.slice().reverse().forEach(r => {
                                const tr = document.createElement('tr');
                                const outBadge = r.success ? 'badge-success' : 'badge-failure';
                                const verText = r.verified_healthy ? '<span style="color: var(--accent-green); font-weight: 600;">VERIFIED HEALTHY</span>' : '<span style="color: var(--accent-red); font-weight: 600;">UNVERIFIED</span>';
                                tr.innerHTML = `
                                    <td style="color: var(--text-muted); font-size: 11px;">#${r.recovery_id}</td>
                                    <td style="font-size: 12px;">${r.iso_time}</td>
                                    <td style="font-weight: 600;">${r.component}</td>
                                    <td style="font-size: 12px;">${r.action_taken}</td>
                                    <td><span class="badge ${outBadge}">${r.success ? 'SUCCESS' : 'FAILED'}</span></td>
                                    <td style="font-size: 12px;">${verText}</td>
                                `;
                                rBody.appendChild(tr);
                            });
                        }
                    }
                }

                // History Table
                if (data.history && data.history.length > 0) {
                    const tbody = document.getElementById('historyTableBody');
                    tbody.innerHTML = '';
                    data.history.slice(-8).reverse().forEach(row => {
                        const tr = document.createElement('tr');
                        const statusClass = row.is_confirmed ? 'status-confirmed' : 'status-rejected';
                        const statusText = row.is_confirmed ? 'CONFIRMED' : ('REJECTED (' + row.rejection_reason + ')');
                        const cmdLabel = row.digit ? (row.command + ' (' + row.digit + ')') : (row.command || 'N/A');
                        tr.innerHTML = `
                            <td>${new Date(row.detected_timestamp * 1000).toLocaleTimeString()}</td>
                            <td style="font-weight: 600; color: #fff;">${cmdLabel}</td>
                            <td>${(row.confidence * 100).toFixed(1)}%</td>
                            <td>${row.margin.toFixed(2)}</td>
                            <td>${row.reaction_latency_ms !== null ? row.reaction_latency_ms.toFixed(2) + ' ms' : 'N/A'}</td>
                            <td class="${statusClass}">${statusText}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                }

                // 8. Real-Time Performance Analytics Updates
                if (data.analytics) {
                    const an = data.analytics;
                    const elPol = document.getElementById('analyticsPolicyBadge');
                    if (elPol) elPol.innerText = (an.scheduling_policy || 'EDF') + ' SCHEDULER';

                    if (an.inference && an.inference.inference_latency_ms) {
                        const il = an.inference.inference_latency_ms;
                        const elMean = document.getElementById('analyticsInferMean');
                        if (elMean) elMean.innerText = `${(il.mean || 0).toFixed(1)} / ${(il.median || 0).toFixed(1)} ms`;
                        const elP95 = document.getElementById('analyticsInferP95');
                        if (elP95) elP95.innerText = `${(il.p95 || 0).toFixed(1)} / ${(il.p99 || 0).toFixed(1)} ms`;
                        const elMax = document.getElementById('analyticsInferMax');
                        if (elMax) elMax.innerText = `${(il.max || 0).toFixed(1)} ms`;
                        const elTput = document.getElementById('analyticsThroughput');
                        if (elTput) elTput.innerText = `${(an.inference.throughput_ips || 0.0).toFixed(1)} IPS`;
                        const elDsp = document.getElementById('analyticsDspMean');
                        if (elDsp && an.inference.feature_extraction_latency_ms) {
                            elDsp.innerText = `${(an.inference.feature_extraction_latency_ms.mean || 0).toFixed(1)} ms`;
                        }
                    }

                    if (an.scheduling) {
                        const sc = an.scheduling;
                        const elResp = document.getElementById('analyticsRespP95');
                        if (elResp && sc.response_time_ms) elResp.innerText = `${(sc.response_time_ms.p95 || 0).toFixed(1)} ms`;
                        const elDmr = document.getElementById('analyticsDmr');
                        if (elDmr) elDmr.innerText = `${(sc.deadline_miss_ratio_pct || 0).toFixed(1)}% (${sc.deadline_miss_count || 0} misses)`;
                        const elUtil = document.getElementById('analyticsUtil');
                        if (elUtil) elUtil.innerText = `${(sc.task_utilization_pct || 0).toFixed(1)}%`;
                    }

                    if (an.end_to_end) {
                        const e2e = an.end_to_end;
                        const elA2I = document.getElementById('analyticsAudioToInfer');
                        if (elA2I && e2e["1_audio_to_inference_ms"]) elA2I.innerText = `${(e2e["1_audio_to_inference_ms"].p95 || 0).toFixed(1)} ms`;
                        const elC2A = document.getElementById('analyticsCmdToAccess');
                        if (elC2A && e2e["5_command_to_access_decision_ms"]) elC2A.innerText = `${(e2e["5_command_to_access_decision_ms"].p95 || 0).toFixed(1)} ms`;
                    }

                    if (an.resources) {
                        const res = an.resources;
                        const elPCpu = document.getElementById('analyticsProcCpu');
                        if (elPCpu) elPCpu.innerText = `${(res.process_cpu_pct || 0).toFixed(1)}%`;
                        const elSCpu = document.getElementById('analyticsSysCpu');
                        if (elSCpu) elSCpu.innerText = `${(res.system_cpu_pct || 0).toFixed(1)}%`;
                        const elRss = document.getElementById('analyticsRss');
                        if (elRss) elRss.innerText = `${(res.process_memory_rss_mb || 0).toFixed(1)} MB`;
                    }

                    const elBCount = document.getElementById('analyticsBenchmarkCount');
                    if (elBCount) elBCount.innerText = `${an.total_benchmarks_recorded || 0} formal benchmarks recorded`;
                }
            } catch (err) {
                console.error("Dashboard fetch error:", err);
                const cBadge = document.getElementById('connBadge');
                if (cBadge) { cBadge.className = 'badge badge-overload'; cBadge.innerText = '⚠ DISCONNECTED'; }
                const uTime = document.getElementById('updateTime');
                if (uTime) uTime.innerText = 'Reconnecting...';
            }
        }
        async function fetchAudit() {
            try {
                const cat = document.getElementById('auditCategory').value;
                const resFilter = document.getElementById('auditResult').value;
                const limit = document.getElementById('auditLimit').value;
                let url = '/api/audit?limit=' + limit;
                if (cat) url += '&category=' + encodeURIComponent(cat);
                if (resFilter) url += '&result=' + encodeURIComponent(resFilter);

                const res = await fetch(url);
                if (!res.ok) return;
                const records = await res.json();
                const tbody = document.getElementById('auditTableBody');
                if (!records || records.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No matching audit records found.</td></tr>';
                    return;
                }
                tbody.innerHTML = '';
                records.forEach(r => {
                    const tr = document.createElement('tr');
                    let badgeClass = 'badge-info';
                    if (r.result === 'SUCCESS') badgeClass = 'badge-success';
                    else if (r.result === 'FAILURE') badgeClass = 'badge-failure';
                    else if (r.result === 'WARNING') badgeClass = 'badge-warning';

                    const timeStr = r.timestamp ? r.timestamp.replace('T', ' ').substring(0, 19) : 'N/A';
                    const userSrc = r.source ? (r.source + ' (' + (r.user_id || 'system') + ')') : (r.user_id || 'system');
                    tr.innerHTML = `
                        <td style="color: var(--text-muted); font-size: 11px;">#${r.event_id}</td>
                        <td style="font-size: 12px;">${timeStr}</td>
                        <td style="font-weight: 600; color: #fff;">${r.event_type}</td>
                        <td><span class="badge ${badgeClass}">${r.result}</span></td>
                        <td style="font-size: 12px;">${r.lock_state || 'N/A'}</td>
                        <td style="color: var(--text-muted); font-size: 12px;">${userSrc}</td>
                        <td style="font-size: 12px;">${r.reason || ''}</td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (err) {
                console.error("Audit fetch error:", err);
            }
        }

        let adminToken = sessionStorage.getItem('rtos_admin_session_token') || '';
        let adminCsrfToken = sessionStorage.getItem('rtos_admin_csrf_token') || '';

        function getAuthHeaders() {
            const headers = { 'Content-Type': 'application/json' };
            if (adminToken) {
                headers['Authorization'] = 'Bearer ' + adminToken;
            }
            if (adminCsrfToken) {
                headers['X-CSRF-Token'] = adminCsrfToken;
            }
            return headers;
        }

        function updateAuthUI(isLoggedIn, username) {
            const loginForm = document.getElementById('adminLoginForm');
            const activeSection = document.getElementById('adminActiveSession');
            const controls = document.getElementById('adminControlsContainer');
            const fb = document.getElementById('adminLoginFeedback');
            if (isLoggedIn) {
                if (loginForm) loginForm.style.display = 'none';
                if (activeSection) activeSection.style.display = 'flex';
                if (controls) controls.style.display = 'block';
                if (fb) fb.innerHTML = '';
                const uSpan = document.getElementById('adminSessionUsername');
                if (uSpan) uSpan.innerText = username || 'Administrator';
            } else {
                if (loginForm) loginForm.style.display = 'flex';
                if (activeSection) activeSection.style.display = 'none';
                if (controls) controls.style.display = 'none';
                const tbody = document.getElementById('userTableBody');
                if (tbody) tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">Please log in with administrator PIN above to view and manage users.</td></tr>';
            }
        }

        async function handleEmergencyStopClick() {
            if (!adminToken) {
                alert("Administrator authentication required to execute emergency stop from dashboard.");
                return;
            }
            if (!confirm("CONFIRM EMERGENCY STOP?\\n\\nThis will immediately force the simulated lock barrier into LOCKED state, clear any active PIN entry buffer, and cancel all pending unlock operations.")) {
                return;
            }
            const fb = document.getElementById('userActionFeedback') || document.getElementById('adminLoginFeedback');
            try {
                const res = await fetch('/api/emergency/stop', {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: JSON.stringify({ reason: "Manual emergency stop triggered via dashboard control" })
                });
                const data = await res.json();
                if (res.ok) {
                    if (fb) fb.innerHTML = '<span style="color: var(--accent-red); font-weight: bold;">🚨 EMERGENCY STOP EXECUTED: Barrier LOCKED. Pending operations cancelled.</span>';
                    fetchStatus();
                    fetchAudit();
                } else {
                    if (fb) fb.innerHTML = '<span style="color: var(--accent-red);">' + (data.error || 'Emergency stop failed') + '</span>';
                }
            } catch (e) {
                if (fb) fb.innerHTML = '<span style="color: var(--accent-red);">' + e.message + '</span>';
            }
        }

        async function handleAdminLogin() {
            const userId = document.getElementById('adminLoginId').value.trim();
            const pin = document.getElementById('adminLoginPin').value.trim();
            const fb = document.getElementById('adminLoginFeedback');

            if (!userId || !pin) {
                fb.innerHTML = '<span style="color: var(--accent-red);">User ID and PIN are required.</span>';
                return;
            }

            try {
                const res = await fetch('/api/admin/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId, pin: pin })
                });
                const data = await res.json();
                if (res.ok && data.token) {
                    adminToken = data.token;
                    adminCsrfToken = data.csrf_token || '';
                    sessionStorage.setItem('rtos_admin_session_token', adminToken);
                    sessionStorage.setItem('rtos_admin_csrf_token', adminCsrfToken);
                    document.getElementById('adminLoginPin').value = '';
                    updateAuthUI(true, data.user ? data.user.username : userId);
                    fetchUsers();
                } else {
                    fb.innerHTML = '<span style="color: var(--accent-red);">' + (data.error || 'Login failed') + '</span>';
                }
            } catch (e) {
                fb.innerHTML = '<span style="color: var(--accent-red);">' + e.message + '</span>';
            }
        }

        async function handleAdminLogout() {
            if (adminToken) {
                try {
                    await fetch('/api/admin/logout', {
                        method: 'POST',
                        headers: getAuthHeaders()
                    });
                } catch (e) {}
            }
            adminToken = '';
            adminCsrfToken = '';
            sessionStorage.removeItem('rtos_admin_session_token');
            sessionStorage.removeItem('rtos_admin_csrf_token');
            updateAuthUI(false);
        }

        async function fetchUsers() {
            if (!adminToken) {
                updateAuthUI(false);
                return;
            }
            try {
                const res = await fetch('/api/admin/users', {
                    headers: getAuthHeaders()
                });
                if (res.status === 401 || res.status === 403) {
                    adminToken = '';
                    sessionStorage.removeItem('rtos_admin_session_token');
                    updateAuthUI(false);
                    return;
                }
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    const fb = document.getElementById('userActionFeedback');
                    if (fb) fb.innerHTML = '<span style="color: var(--accent-red);">' + (err.error || 'Failed to load users') + '</span>';
                    return;
                }
                const users = await res.json();
                const tbody = document.getElementById('userTableBody');
                if (!tbody) return;
                tbody.innerHTML = '';

                users.forEach(u => {
                    const tr = document.createElement('tr');
                    let roleBadge = 'badge-regular';
                    if (u.role === 'ADMIN') roleBadge = 'badge-admin';
                    else if (u.role === 'GUEST') roleBadge = 'badge-guest';

                    let statusBadge = 'badge-active';
                    if (u.status === 'DISABLED') statusBadge = 'badge-disabled';
                    else if (u.status === 'EXPIRED') statusBadge = 'badge-expired';

                    const created = u.created_at ? u.created_at.replace('T', ' ').substring(0, 19) : 'N/A';
                    const expires = u.expires_at ? u.expires_at.replace('T', ' ').substring(0, 19) : 'Permanent';
                    const lastAcc = u.last_access_at ? u.last_access_at.replace('T', ' ').substring(0, 19) : 'Never';

                    const toggleLabel = u.status === 'ACTIVE' ? 'Disable' : 'Enable';
                    const toggleColor = u.status === 'ACTIVE' ? '#d29922' : '#3fb950';

                    tr.innerHTML = `
                        <td style="font-weight: 600; color: #fff;">${u.user_id}</td>
                        <td>${u.username}</td>
                        <td><span class="badge ${roleBadge}">${u.role}</span></td>
                        <td><span class="badge ${statusBadge}">${u.status}</span></td>
                        <td style="font-size: 12px; color: var(--text-muted);">${created}</td>
                        <td style="font-size: 12px; color: var(--text-muted);">${expires}</td>
                        <td style="font-size: 12px; color: var(--text-muted);">${lastAcc}</td>
                        <td>
                            <button onclick="handleToggleStatus('${u.user_id}', '${u.status}')" style="color: ${toggleColor}; margin-right: 4px;">${toggleLabel}</button>
                            <button onclick="handleResetPin('${u.user_id}')" style="margin-right: 4px;">Reset PIN</button>
                            <button onclick="handleRemoveUser('${u.user_id}')" style="color: var(--accent-red);">Delete</button>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (err) {
                console.error("Fetch users error:", err);
            }
        }

        async function handleCreateUser() {
            if (!adminToken) { updateAuthUI(false); return; }
            const userId = document.getElementById('newUserId').value.trim();
            const username = document.getElementById('newUsername').value.trim();
            const pin = document.getElementById('newUserPin').value.trim();
            const role = document.getElementById('newUserRole').value;
            const fb = document.getElementById('userActionFeedback');

            if (!userId || !username || !pin) {
                fb.innerHTML = '<span style="color: var(--accent-red);">User ID, Name, and PIN are required.</span>';
                return;
            }

            try {
                const res = await fetch('/api/admin/users/create', {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: JSON.stringify({ user_id: userId, username: username, pin: pin, role: role })
                });
                const data = await res.json();
                if (res.status === 401 || res.status === 403) {
                    adminToken = '';
                    sessionStorage.removeItem('rtos_admin_session_token');
                    updateAuthUI(false);
                    return;
                }
                if (res.ok) {
                    fb.innerHTML = '<span style="color: var(--accent-green);">User ' + userId + ' created successfully!</span>';
                    document.getElementById('newUserId').value = '';
                    document.getElementById('newUsername').value = '';
                    document.getElementById('newUserPin').value = '';
                    fetchUsers();
                } else {
                    fb.innerHTML = '<span style="color: var(--accent-red);">' + (data.error || 'Failed to create user') + '</span>';
                }
            } catch (e) {
                fb.innerHTML = '<span style="color: var(--accent-red);">' + e.message + '</span>';
            }
        }

        async function handleToggleStatus(userId, currentStatus) {
            if (!adminToken) { updateAuthUI(false); return; }
            const endpoint = (currentStatus === 'ACTIVE') ? '/api/admin/users/disable' : '/api/admin/users/enable';
            const fb = document.getElementById('userActionFeedback');
            try {
                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: JSON.stringify({ target_user_id: userId })
                });
                const data = await res.json();
                if (res.status === 401 || res.status === 403) {
                    adminToken = '';
                    sessionStorage.removeItem('rtos_admin_session_token');
                    updateAuthUI(false);
                    return;
                }
                if (res.ok) {
                    fb.innerHTML = '<span style="color: var(--accent-green);">User status updated.</span>';
                    fetchUsers();
                } else {
                    fb.innerHTML = '<span style="color: var(--accent-red);">' + (data.error || 'Operation failed') + '</span>';
                }
            } catch (e) {
                fb.innerHTML = '<span style="color: var(--accent-red);">' + e.message + '</span>';
            }
        }

        async function handleRemoveUser(userId) {
            if (!adminToken) { updateAuthUI(false); return; }
            if (!confirm('Are you sure you want to permanently remove user ' + userId + '?')) return;
            const fb = document.getElementById('userActionFeedback');
            try {
                const res = await fetch('/api/admin/users/remove', {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: JSON.stringify({ target_user_id: userId })
                });
                const data = await res.json();
                if (res.status === 401 || res.status === 403) {
                    adminToken = '';
                    sessionStorage.removeItem('rtos_admin_session_token');
                    updateAuthUI(false);
                    return;
                }
                if (res.ok) {
                    fb.innerHTML = '<span style="color: var(--accent-green);">User removed.</span>';
                    fetchUsers();
                } else {
                    fb.innerHTML = '<span style="color: var(--accent-red);">' + (data.error || 'Operation failed') + '</span>';
                }
            } catch (e) {
                fb.innerHTML = '<span style="color: var(--accent-red);">' + e.message + '</span>';
            }
        }

        async function handleResetPin(userId) {
            if (!adminToken) { updateAuthUI(false); return; }
            const newPin = prompt('Enter new 4-digit PIN for ' + userId + ' (digits 1-9 only):');
            if (!newPin) return;
            const fb = document.getElementById('userActionFeedback');
            try {
                const res = await fetch('/api/admin/users/reset-pin', {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: JSON.stringify({ target_user_id: userId, new_pin: newPin })
                });
                const data = await res.json();
                if (res.status === 401 || res.status === 403) {
                    adminToken = '';
                    sessionStorage.removeItem('rtos_admin_session_token');
                    updateAuthUI(false);
                    return;
                }
                if (res.ok) {
                    fb.innerHTML = '<span style="color: var(--accent-green);">PIN reset successfully for ' + userId + '!</span>';
                    fetchUsers();
                } else {
                    fb.innerHTML = '<span style="color: var(--accent-red);">' + (data.error || 'Operation failed') + '</span>';
                }
            } catch (e) {
                fb.innerHTML = '<span style="color: var(--accent-red);">' + e.message + '</span>';
            }
        }

        async function fetchBenchmarks() {
            try {
                const resp = await fetch('/api/analytics');
                if (!resp.ok) return;
                const d = await resp.json();
                const bBody = document.getElementById('benchmarkTableBody');
                if (!bBody) return;
                if (!d.benchmarks || d.benchmarks.length === 0) {
                    bBody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">No benchmark experiments recorded.</td></tr>';
                    return;
                }
                bBody.innerHTML = '';
                d.benchmarks.slice().reverse().forEach(b => {
                    const tr = document.createElement('tr');
                    const inf = b.inference ? (b.inference.inference_latency_ms ? b.inference.inference_latency_ms.p95 : (b.infer_p95_ms || 0)) : 0;
                    const respT = b.scheduling ? (b.scheduling.response_time_ms ? b.scheduling.response_time_ms.p95 : (b.sched_resp_p95_ms || 0)) : 0;
                    const dmr = b.scheduling ? (b.scheduling.deadline_miss_ratio_pct || 0) : 0;
                    const tput = b.inference ? (b.inference.throughput_ips || 0) : 0;
                    const cpu = b.resources ? (b.resources.process_cpu_pct || 0) : 0;
                    const e2e = b.end_to_end ? (b.end_to_end["5_command_to_access_decision_ms"] ? b.end_to_end["5_command_to_access_decision_ms"].p95 : 0) : 0;
                    tr.innerHTML = `
                        <td style="font-family: monospace; font-size: 11px;">${b.experiment_id}</td>
                        <td><span class="badge badge-info">${b.scheduling_policy}</span></td>
                        <td><strong>${b.workload_type}</strong></td>
                        <td>${Number(inf).toFixed(1)}</td>
                        <td>${Number(respT).toFixed(1)}</td>
                        <td>${Number(dmr).toFixed(1)}%</td>
                        <td>${Number(tput).toFixed(1)} IPS</td>
                        <td>${Number(cpu).toFixed(1)}%</td>
                        <td>${e2e > 0 ? Number(e2e).toFixed(1) + ' ms' : 'N/A'}</td>
                    `;
                    bBody.appendChild(tr);
                });
            } catch (err) {
                console.error("Benchmark fetch error:", err);
            }
        }

        async function fetchAccessAnalytics() {
            try {
                const sDate = document.getElementById('accessStartDate').value;
                const eDate = document.getElementById('accessEndDate').value;
                const uId = document.getElementById('accessUserId').value;
                const role = document.getElementById('accessRole').value;
                const res = document.getElementById('accessResult').value;

                let url = '/api/access-analytics/summary?';
                if (sDate) url += '&start_date=' + encodeURIComponent(sDate);
                if (eDate) url += '&end_date=' + encodeURIComponent(eDate);
                if (uId) url += '&user_id=' + encodeURIComponent(uId);
                if (role) url += '&role=' + encodeURIComponent(role);
                if (res) url += '&result=' + encodeURIComponent(res);

                const response = await fetch(url);
                if (!response.ok) return;
                const data = await response.json();

                // Update Overall Cards
                const ov = data.overall || {};
                document.getElementById('accTotalAttempts').innerText = ov.total_attempts || 0;
                document.getElementById('accSuccessAttempts').innerText = ov.successful_attempts || 0;
                document.getElementById('accFailedAttempts').innerText = ov.failed_attempts || 0;
                document.getElementById('accSuccessRate').innerText = (ov.success_rate_pct || 0).toFixed(1) + '%';
                document.getElementById('accDenialRate').innerText = (ov.denial_rate_pct || 0).toFixed(1) + '%';

                // Security metrics
                const sec = data.security || {};
                document.getElementById('accLockouts').innerText = sec.lockout_count || 0;
                document.getElementById('accEmergencyStops').innerText = sec.emergency_stop_count || 0;
                document.getElementById('accRepeatedFails').innerText = sec.repeated_failed_auth_count || 0;

                // Peak badges
                const timeAct = data.time_activity || {};
                const peaks = timeAct.peak_periods || {};
                const pkDay = peaks.peak_day || {};
                document.getElementById('accPeakDayBadge').innerText = pkDay.date && pkDay.date !== 'N/A' ? `Peak Day: ${pkDay.date} (${pkDay.attempts} attempts)` : 'Peak: N/A';
                const pkHour = peaks.peak_hour || {};
                document.getElementById('accPeakHourBadge').innerText = pkHour.label && pkHour.label !== 'N/A' ? `Peak Hour: ${pkHour.label} (${pkHour.attempts} attempts)` : 'Peak: N/A';

                // Daily Trends
                const dailyC = document.getElementById('dailyTrendsContainer');
                const dailyList = timeAct.daily_trends || [];
                if (dailyList.length === 0) {
                    dailyC.innerHTML = '<div style="text-align: center; color: var(--text-muted); font-size: 12px; padding: 20px 0;">No access records for selected filters.</div>';
                } else {
                    dailyC.innerHTML = '';
                    const maxDaily = Math.max(...dailyList.map(d => d.total_attempts), 1);
                    dailyList.forEach(d => {
                        const row = document.createElement('div');
                        row.style = 'display: flex; align-items: center; font-size: 11px; gap: 8px;';
                        row.innerHTML = `
                            <span style="width: 75px; color: var(--text-muted);">${d.date}</span>
                            <div style="flex: 1; background: var(--bg-color); height: 12px; border-radius: 3px; overflow: hidden; display: flex;">
                                <div style="background: var(--accent-green); width: ${(d.successful_attempts / Math.max(1, d.total_attempts) * 100) || 0}%;"></div>
                                <div style="background: var(--accent-red); width: ${(d.failed_attempts / Math.max(1, d.total_attempts) * 100) || 0}%;"></div>
                            </div>
                            <span style="width: 70px; text-align: right; color: #fff;">${d.total_attempts} (${d.success_rate_pct.toFixed(0)}%)</span>
                        `;
                        dailyC.appendChild(row);
                    });
                }

                // Hourly Histogram
                const hourlyC = document.getElementById('hourlyTrendsContainer');
                const hourlyList = timeAct.hourly_trends || [];
                hourlyC.innerHTML = '';
                const maxHourly = Math.max(...hourlyList.map(h => h.total_attempts), 1);
                hourlyList.forEach(h => {
                    const barH = Math.max(4, Math.round((h.total_attempts / maxHourly) * 120));
                    const isPeak = (pkHour.hour !== null && pkHour.hour === h.hour && h.total_attempts > 0);
                    const colBg = isPeak ? 'var(--accent-orange)' : (h.total_attempts > 0 ? 'var(--accent-blue)' : '#2a2f3a');
                    const bDiv = document.createElement('div');
                    bDiv.style = `flex: 1; height: ${barH}px; background: ${colBg}; border-radius: 2px 2px 0 0; position: relative; cursor: pointer;`;
                    bDiv.title = `${h.hour_label}: ${h.total_attempts} attempts (${h.successful_attempts} success, ${h.failed_attempts} fail)`;
                    hourlyC.appendChild(bDiv);
                });

                // User Activity Table
                const uBody = document.getElementById('userActivityTableBody');
                const users = data.user_activity || [];
                if (users.length === 0) {
                    uBody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No user activity found.</td></tr>';
                } else {
                    uBody.innerHTML = '';
                    users.forEach(u => {
                        const tr = document.createElement('tr');
                        const lastT = u.last_access_time ? u.last_access_time.replace('T', ' ').substring(0, 19) : 'N/A';
                        tr.innerHTML = `
                            <td style="font-weight: bold; color: #fff;">${u.user_id}</td>
                            <td>${u.username}</td>
                            <td><span class="badge badge-info">${u.role || 'N/A'}</span></td>
                            <td style="font-weight: 600;">${u.total_attempts}</td>
                            <td style="color: var(--accent-green);">${u.successful_attempts}</td>
                            <td style="color: var(--accent-red);">${u.failed_attempts}</td>
                            <td>${u.success_rate_pct.toFixed(1)}%</td>
                            <td style="color: var(--text-muted); font-size: 11px;">${lastT}</td>
                        `;
                        uBody.appendChild(tr);
                    });
                }
            } catch (err) {
                console.error("Access Analytics fetch error:", err);
            }
        }

        function downloadAccessReport(fmt) {
            const sDate = document.getElementById('accessStartDate').value;
            const eDate = document.getElementById('accessEndDate').value;
            const uId = document.getElementById('accessUserId').value;
            const role = document.getElementById('accessRole').value;
            const res = document.getElementById('accessResult').value;

            let url = `/api/access-analytics/report?format=${fmt}`;
            if (sDate) url += '&start_date=' + encodeURIComponent(sDate);
            if (eDate) url += '&end_date=' + encodeURIComponent(eDate);
            if (uId) url += '&user_id=' + encodeURIComponent(uId);
            if (role) url += '&role=' + encodeURIComponent(role);
            if (res) url += '&result=' + encodeURIComponent(res);

            window.location.href = url;
        }

        function resetAccessFilters() {
            document.getElementById('accessStartDate').value = '';
            document.getElementById('accessEndDate').value = '';
            document.getElementById('accessUserId').value = '';
            document.getElementById('accessRole').value = '';
            document.getElementById('accessResult').value = '';
            fetchAccessAnalytics();
        }

        // ======================================================================
        // Accessibility Management JavaScript Subsystem
        // ======================================================================
        async function fetchA11ySettings() {
            try {
                const res = await fetch('/api/accessibility/settings');
                if (!res.ok) return;
                const data = await res.json();
                a11ySettings = data;
                syncA11yUIWithSettings();
                applyA11yDOMClasses();
            } catch (e) {
                console.debug('Failed to fetch a11y settings:', e);
            }
        }

        function syncA11yUIWithSettings() {
            const chkLarge = document.getElementById('toggleLargeText');
            if (chkLarge) chkLarge.checked = !!a11ySettings.large_text;

            const chkContrast = document.getElementById('toggleHighContrast');
            if (chkContrast) chkContrast.checked = !!a11ySettings.high_contrast;

            const chkClear = document.getElementById('toggleClearIndicators');
            if (chkClear) chkClear.checked = a11ySettings.clear_status_indicators !== false;

            const chkVoice = document.getElementById('toggleSpokenFeedback');
            if (chkVoice) chkVoice.checked = a11ySettings.spoken_feedback !== false;

            const chkAudio = document.getElementById('toggleAudioConfirmation');
            if (chkAudio) chkAudio.checked = a11ySettings.audio_confirmation !== false;

            const chkReduced = document.getElementById('toggleReducedComplexity');
            if (chkReduced) chkReduced.checked = !!a11ySettings.reduced_visual_complexity;

            const rngTimeout = document.getElementById('inputPinTimeout');
            if (rngTimeout) rngTimeout.value = a11ySettings.pin_entry_timeout_s || 10;
            updatePinTimeoutDisplay(a11ySettings.pin_entry_timeout_s || 10);

            const rngVol = document.getElementById('inputVoiceVolume');
            const volPct = Math.round((a11ySettings.voice_feedback_volume !== undefined ? a11ySettings.voice_feedback_volume : 1.0) * 100);
            if (rngVol) rngVol.value = volPct;
            updateVolumeDisplay(volPct);
        }

        function applyA11yDOMClasses() {
            const b = document.body;
            if (a11ySettings.large_text) b.classList.add('large-text');
            else b.classList.remove('large-text');

            if (a11ySettings.high_contrast) b.classList.add('high-contrast');
            else b.classList.remove('high-contrast');

            if (a11ySettings.reduced_visual_complexity) b.classList.add('reduced-complexity');
            else b.classList.remove('reduced-complexity');

            const badge = document.getElementById('a11yStatusBadge');
            if (badge) {
                const modes = [];
                if (a11ySettings.large_text) modes.push('LARGE');
                if (a11ySettings.high_contrast) modes.push('CONTRAST');
                if (a11ySettings.spoken_feedback) modes.push('VOICE');
                badge.innerText = modes.length > 0 ? '[' + modes.join(' + ') + ']' : '[A11Y READY]';
            }
        }

        function applyA11ySettingsFromUI() {
            a11ySettings.large_text = document.getElementById('toggleLargeText').checked;
            a11ySettings.high_contrast = document.getElementById('toggleHighContrast').checked;
            a11ySettings.clear_status_indicators = document.getElementById('toggleClearIndicators').checked;
            a11ySettings.spoken_feedback = document.getElementById('toggleSpokenFeedback').checked;
            a11ySettings.audio_confirmation = document.getElementById('toggleAudioConfirmation').checked;
            a11ySettings.reduced_visual_complexity = document.getElementById('toggleReducedComplexity').checked;
            a11ySettings.pin_entry_timeout_s = parseFloat(document.getElementById('inputPinTimeout').value) || 10.0;
            a11ySettings.voice_feedback_volume = (parseFloat(document.getElementById('inputVoiceVolume').value) || 100) / 100.0;

            applyA11yDOMClasses();
        }

        function updatePinTimeoutDisplay(val) {
            const el = document.getElementById('pinTimeoutDisplay');
            if (el) el.innerText = parseFloat(val).toFixed(1) + 's';
        }

        function updateVolumeDisplay(val) {
            const el = document.getElementById('voiceVolumeDisplay');
            if (el) el.innerText = parseInt(val, 10) + '%';
        }

        async function saveA11ySettingsToServer() {
            applyA11ySettingsFromUI();
            const fb = document.getElementById('a11yFeedbackMsg');
            if (fb) fb.innerText = 'Saving...';
            try {
                const res = await fetch('/api/accessibility/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(a11ySettings)
                });
                if (res.ok) {
                    const data = await res.json();
                    a11ySettings = data;
                    syncA11yUIWithSettings();
                    applyA11yDOMClasses();
                    if (fb) {
                        fb.innerText = '✓ Preferences saved successfully.';
                        fb.style.color = 'var(--accent-green)';
                        setTimeout(() => { fb.innerText = ''; }, 3000);
                    }
                } else {
                    if (fb) {
                        fb.innerText = 'Error saving preferences.';
                        fb.style.color = 'var(--accent-red)';
                    }
                }
            } catch (e) {
                if (fb) {
                    fb.innerText = 'Network error saving preferences.';
                    fb.style.color = 'var(--accent-red)';
                }
            }
        }

        async function resetA11ySettingsToDefaults() {
            const fb = document.getElementById('a11yFeedbackMsg');
            if (fb) fb.innerText = 'Resetting...';
            try {
                const res = await fetch('/api/accessibility/settings/reset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: '{}'
                });
                if (res.ok) {
                    const data = await res.json();
                    a11ySettings = data;
                    syncA11yUIWithSettings();
                    applyA11yDOMClasses();
                    if (fb) {
                        fb.innerText = '✓ Restored factory default settings.';
                        fb.style.color = 'var(--accent-green)';
                        setTimeout(() => { fb.innerText = ''; }, 3000);
                    }
                }
            } catch (e) {
                if (fb) fb.innerText = 'Error resetting settings.';
            }
        }

        async function testSpokenFeedback() {
            const fb = document.getElementById('a11yFeedbackMsg');
            try {
                const res = await fetch('/api/accessibility/feedback/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ event: 'SYSTEM_READY' })
                });
                if (res.ok) {
                    const data = await res.json();
                    if (fb) {
                        fb.innerText = '🔊 Spoke: "' + (data.message || 'System ready.') + '"';
                        fb.style.color = 'var(--accent-blue)';
                        setTimeout(() => { fb.innerText = ''; }, 3500);
                    }
                }
            } catch (e) {
                if (fb) fb.innerText = 'Speech test failed.';
            }
        }

        setInterval(fetchStatus, 200);
        setInterval(fetchAudit, 2000);
        setInterval(fetchUsers, 3000);
        setInterval(fetchBenchmarks, 3000);
        setInterval(fetchAccessAnalytics, 4000);
        setInterval(fetchA11ySettings, 4000);
        fetchStatus();
        fetchA11ySettings();
        fetchAudit();
        fetchBenchmarks();
        fetchAccessAnalytics();
        if (adminToken) {
            updateAuthUI(true);
            fetchUsers();
        } else {
            updateAuthUI(false);
        }
    </script>
</body>
</html>
"""


class LoginRateLimiter:
    """
    Thread-safe sliding-window rate limiter for administrator login attempts.
    Protects against PIN brute-force attacks across both existing and nonexistent user IDs
    without account enumeration leakage.
    """

    def __init__(self, max_failures: int = 5, cooldown_seconds: float = 60.0):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self._failures: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def check_rate_limit(self, key: str) -> Tuple[bool, float]:
        """
        Checks if the key is currently rate-limited.
        Returns:
            (True, 0.0) if attempt is allowed.
            (False, time_remaining_s) if in cooldown.
        """
        now = time.time()
        with self._lock:
            attempts = self._failures.get(key, [])
            valid_attempts = [t for t in attempts if now - t < self.cooldown_seconds]
            if len(valid_attempts) >= self.max_failures:
                oldest = valid_attempts[0]
                time_remaining = max(1.0, (oldest + self.cooldown_seconds) - now)
                self._failures[key] = valid_attempts
                return False, time_remaining
            self._failures[key] = valid_attempts
            return True, 0.0

    def record_failure(self, key: str):
        """Records a failed attempt for the specified key."""
        now = time.time()
        with self._lock:
            attempts = self._failures.get(key, [])
            attempts = [t for t in attempts if now - t < self.cooldown_seconds]
            attempts.append(now)
            self._failures[key] = attempts

    def record_success(self, key: str):
        """Clears failure history for the specified key upon successful authentication."""
        with self._lock:
            self._failures.pop(key, None)

    def reset(self):
        """Clears all tracked failures."""
        with self._lock:
            self._failures.clear()


@dataclass
class AdminSession:
    session_id: str
    user_id: str
    csrf_token: str
    created_at: float
    expires_at: float
    last_activity: float


class AdminSessionManager:
    """
    Manages short-lived, cryptographically random administrative session tokens.
    Provides thread-safe creation, validation, expiration, and explicit revocation.
    """

    def __init__(self, default_ttl_seconds: int = 3600):
        self.default_ttl = default_ttl_seconds
        self._sessions: Dict[str, AdminSession] = {}
        self._lock = threading.Lock()

    def create_session(self, user_id: str, ttl_seconds: Optional[int] = None) -> str:
        """Generates a secure 32-byte URL-safe session token and CSRF synchronizer token, recording the session."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        now = time.time()
        session = AdminSession(
            session_id=token,
            user_id=user_id,
            csrf_token=csrf_token,
            created_at=now,
            expires_at=now + ttl,
            last_activity=now,
        )
        with self._lock:
            self._sessions[token] = session
        return token

    def get_session(self, token: str) -> Optional[AdminSession]:
        """Retrieves active session object by token if present and not expired."""
        if not token:
            return None
        now = time.time()
        with self._lock:
            session = self._sessions.get(token)
            if not session:
                return None
            if now > session.expires_at:
                del self._sessions[token]
                return None
            return session

    def validate_session(self, token: str, user_manager: Optional[UserManager] = None) -> Optional[UserRecord]:
        """
        Validates the session token.
        Checks:
        1. Token existence in session store.
        2. Session expiration (TTL).
        3. Real-time active status and ADMIN role of the user in the database.
        Returns UserRecord if valid and active ADMIN; otherwise None (and revokes invalidated tokens).
        """
        if not token:
            return None

        now = time.time()
        user_id = None
        with self._lock:
            session = self._sessions.get(token)
            if not session:
                return None
            if now > session.expires_at:
                del self._sessions[token]
                return None
            session.last_activity = now
            user_id = session.user_id

        if user_manager is not None:
            user = user_manager.get_user(user_id)
            if user is None or user.status != UserStatus.ACTIVE or user.role != UserRole.ADMIN:
                self.revoke_session(token)
                return None
            return user
        return None

    def revoke_session(self, token: str) -> bool:
        """Revokes an active session token."""
        with self._lock:
            return self._sessions.pop(token, None) is not None

    def revoke_all_for_user(self, user_id: str):
        """Revokes all active sessions belonging to the specified user."""
        with self._lock:
            to_remove = [tok for tok, sess in self._sessions.items() if sess.user_id == user_id]
            for tok in to_remove:
                del self._sessions[tok]

    def cleanup_expired(self):
        """Prunes expired sessions."""
        now = time.time()
        with self._lock:
            expired = [tok for tok, sess in self._sessions.items() if now > sess.expires_at]
            for tok in expired:
                del self._sessions[tok]


class DashboardStatusProvider:
    """
    Thread-safe registry supplying live application state to the dashboard server.
    """

    def __init__(
        self,
        audit_logger: Optional[Any] = None,
        user_manager: Optional[Any] = None,
        session_manager: Optional[AdminSessionManager] = None,
        login_rate_limiter: Optional[LoginRateLimiter] = None,
        emergency_handler: Optional[Any] = None,
        fault_detector: Optional[Any] = None,
        recovery_manager: Optional[Any] = None,
        analytics_collector: Optional[Any] = None,
        workload_runner: Optional[Any] = None,
        access_analytics: Optional[Any] = None,
        accessibility_manager: Optional[Any] = None,
        voice_feedback: Optional[Any] = None,
    ):
        self._lock = threading.Lock()
        self.audit_logger = audit_logger
        self.user_manager = user_manager
        self.session_manager = session_manager or AdminSessionManager()
        self.login_rate_limiter = login_rate_limiter or LoginRateLimiter(max_failures=5, cooldown_seconds=60.0)
        self.emergency_handler = emergency_handler
        self.fault_detector = fault_detector
        self.recovery_manager = recovery_manager
        self.analytics_collector = analytics_collector
        self.workload_runner = workload_runner
        self.access_analytics = access_analytics
        self.accessibility_manager = accessibility_manager
        self.voice_feedback = voice_feedback
        if self.accessibility_manager is None:
            try:
                from src.accessibility import AccessibilitySettingsManager
                self.accessibility_manager = AccessibilitySettingsManager()
            except Exception as e:
                logger.debug(f"[DASHBOARD] Could not initialize AccessibilitySettingsManager: {e}")
        if self.access_analytics is None and self.audit_logger is not None:
            try:
                from src.analytics.access_analytics import AccessAnalyticsManager
                self.access_analytics = AccessAnalyticsManager(
                    db_path=self.audit_logger.db_path,
                    user_manager=self.user_manager,
                    audit_logger=self.audit_logger,
                )
            except Exception as e:
                logger.debug(f"[DASHBOARD] Could not initialize AccessAnalyticsManager: {e}")
        self._status: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "microphone": {
                "device_name": "Virtual Stream",
                "device_id": None,
                "status": "INITIALIZING",
                "sample_rate": 16000,
                "queue_occupancy": 0,
                "queue_capacity": 100,
                "audio_blocks_dropped": 0,
            },
            "dsp": {
                "vad_state": "WAITING",
                "noise_floor_rms": None,
            },
            "command_fsm": {
                "current_state": "WAITING",
                "latest_command": None,
                "latest_digit": None,
                "latest_confidence": None,
                "total_confirmed": 0,
                "total_rejected": 0,
            },
            "timing": {
                "processing_reaction_latency_ms": None,
                "total_acoustic_latency_ms": None,
                "model_inference_ms": None,
                "feature_extraction_ms": None,
                "queue_delay_ms": None,
            },
            "scheduler": {
                "policy": "EDF",
                "ready_jobs_count": 0,
                "total_executed": 0,
                "total_deadline_misses": 0,
                "deadline_miss_ratio_pct": 0.0,
            },
            "adaptive_controller": {
                "current_state": "NORMAL",
                "total_state_transitions": 0,
                "total_stale_jobs_pruned": 0,
                "total_inferences_suppressed": 0,
                "mean_recent_queue_delay_ms": 0.0,
            },
            "resources": {
                "cpu_utilization_pct": 0.0,
                "memory_rss_mb": 0.0,
            },
            "lock": {
                "state": "LOCKED",
                "is_locked": True,
                "is_unlocked": False,
                "is_lockout": False,
                "digits_entered": 0,
                "pin_length": 4,
                "masked_entry": "",
                "pin_progress": "0/4 digits",
                "failed_attempts": 0,
                "max_failed_attempts": 3,
                "lockout_remaining_s": 0.0,
                "unlock_remaining_s": 0.0,
                "last_access_result": "NONE",
                "last_access_message": "System initialized in LOCKED state.",
                "last_command": None,
                "operational_status": "LOCKED (Ready for PIN)",
            },
            "emergency": {
                "is_active": False,
                "last_stop_timestamp": None,
                "last_stop_source": None,
                "last_stop_reason": None,
                "last_lock_state": "LOCKED",
                "pending_cancelled": False,
                "total_received": 0,
                "total_accepted": 0,
                "total_rejected": 0,
                "total_pending_cancelled": 0,
                "last_stop_event": None,
            },
            "health": {
                "system_health": "HEALTHY",
                "components": {
                    "microphone": "OPERATIONAL",
                    "audio_queue": "OPERATIONAL",
                    "inference": "OPERATIONAL",
                    "scheduler": "OPERATIONAL",
                    "database": "OPERATIONAL",
                    "dashboard": "OPERATIONAL",
                    "lock": "OPERATIONAL",
                },
                "active_faults": [],
                "active_faults_count": 0,
                "total_faults_detected": 0,
                "total_recoveries": 0,
                "succeeded_recoveries": 0,
                "failed_recoveries": 0,
                "recent_faults": [],
                "recent_recoveries": [],
                "is_recovering": False,
            },
            "history": [],
        }

    def update_status(self, updates: Dict[str, Any]):
        """Thread-safe update of current status snapshot."""
        with self._lock:
            for k, v in updates.items():
                if isinstance(v, dict) and k in self._status and isinstance(self._status[k], dict):
                    self._status[k].update(v)
                else:
                    self._status[k] = v
            self._status["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # Collect real OS resource metrics
            self._status["resources"]["memory_rss_mb"] = get_process_memory_mb()

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if self.emergency_handler is not None:
                try:
                    self._status["emergency"] = self.emergency_handler.get_emergency_status()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Error reading emergency status: {e}")

            if self.fault_detector is not None:
                try:
                    f_snap = self.fault_detector.get_health_snapshot()
                    self._status["health"]["system_health"] = f_snap["system_health"]
                    self._status["health"]["components"] = f_snap["components"]
                    self._status["health"]["active_faults"] = f_snap["active_faults"]
                    self._status["health"]["active_faults_count"] = f_snap["active_faults_count"]
                    self._status["health"]["total_faults_detected"] = f_snap["total_faults_detected"]
                    self._status["health"]["recent_faults"] = f_snap["recent_faults"]
                    self._status["health"]["is_recovering"] = f_snap["is_recovering"]
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Error reading health snapshot: {e}")

            if self.recovery_manager is not None:
                try:
                    r_stats = self.recovery_manager.get_recovery_stats()
                    self._status["health"]["total_recoveries"] = r_stats["total_recoveries"]
                    self._status["health"]["succeeded_recoveries"] = r_stats["succeeded"]
                    self._status["health"]["failed_recoveries"] = r_stats["failed"]
                    self._status["health"]["recent_recoveries"] = r_stats["recent_recoveries"]
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Error reading recovery stats: {e}")

            if self.analytics_collector is not None:
                try:
                    self._status["analytics"] = self.analytics_collector.get_live_telemetry()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Error reading analytics snapshot: {e}")

            if self.accessibility_manager is not None:
                try:
                    self._status["accessibility"] = self.accessibility_manager.settings.to_dict()
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Error reading accessibility snapshot: {e}")

            return json.loads(json.dumps(self._status))

    def get_accessibility_settings(self) -> Dict[str, Any]:
        """Returns the current accessibility settings dictionary."""
        if self.accessibility_manager is not None:
            return self.accessibility_manager.settings.to_dict()
        return {}

    def update_accessibility_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Applies updates to accessibility settings and returns updated state."""
        if self.accessibility_manager is not None:
            updated = self.accessibility_manager.update(updates, persist=True)
            return updated.to_dict()
        return {}

    def reset_accessibility_settings(self) -> Dict[str, Any]:
        """Resets accessibility settings to defaults and returns state."""
        if self.accessibility_manager is not None:
            res = self.accessibility_manager.reset_defaults(persist=True)
            return res.to_dict()
        return {}

    def emit_test_voice_feedback(self, event_name: str) -> Dict[str, Any]:
        """Dispatches a test voice feedback event."""
        if self.voice_feedback is not None:
            try:
                from src.accessibility.voice_feedback import FeedbackEvent
                evt = FeedbackEvent(event_name.upper())
                rec = self.voice_feedback.emit_event(evt)
                return rec.to_dict()
            except Exception as e:
                return {"error": str(e)}
        return {"error": "Voice feedback not configured"}

    def list_benchmarks(self) -> List[Dict[str, Any]]:
        """Returns list of summaries for all formal benchmark experiments."""
        if self.analytics_collector is not None:
            return self.analytics_collector.list_benchmarks()
        return []

    def query_audit(
        self,
        event_type: Optional[str] = None,
        result: Optional[str] = None,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Queries the persistent audit logger if configured."""
        if self.audit_logger is not None:
            try:
                return self.audit_logger.query(
                    event_type=event_type,
                    result=result,
                    user_id=user_id,
                    category=category,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit,
                    offset=offset,
                )
            except Exception as e:
                logger.error(f"[DASHBOARD] Error querying audit logger: {e}")
                return []
        return []

    def get_users(self) -> List[Dict[str, Any]]:
        """Returns safe user summaries excluding credential hashes and salts."""
        if self.user_manager is not None:
            try:
                users = self.user_manager.get_all_users()
                return [u.to_dict(include_sensitive=False) for u in users]
            except Exception as e:
                logger.error(f"[DASHBOARD] Error retrieving users: {e}")
                return []
        return []

    def get_access_analytics_summary(self, filters: Optional[AccessFilter] = None) -> Dict[str, Any]:
        """Calculates access analytics summary across filtered criteria."""
        if self.access_analytics is not None:
            try:
                summary = self.access_analytics.compute_summary(filters=filters)
                return summary.to_dict()
            except Exception as e:
                logger.error(f"[DASHBOARD] Error computing access analytics summary: {e}")
                return {}
        return {}

    def generate_access_report(self, filters: Optional[AccessFilter] = None, limit: int = 5000) -> Optional[AccessReport]:
        """Generates downloadable AccessReport."""
        if self.access_analytics is not None:
            try:
                return self.access_analytics.generate_report(filters=filters, limit=limit)
            except Exception as e:
                logger.error(f"[DASHBOARD] Error generating access report: {e}")
                return None
        return None


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler dispatching JSON API calls and HTML."""

    protocol_version = "HTTP/1.1"
    status_provider: Optional[DashboardStatusProvider] = None

    def log_message(self, format, *args):
        # Suppress noisy HTTP access logging to keep console clean
        pass

    def _send_bytes(
        self,
        content_bytes: bytes,
        content_type: str = "application/json; charset=utf-8",
        status_code: int = 200,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content_bytes)))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization, X-Admin-Token, X-Admin-User, X-CSRF-Token",
            )
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(content_bytes)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.debug(f"[DASHBOARD] Socket write error: {e}")

    def _send_json(
        self,
        data: Any,
        status_code: int = 200,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        content_bytes = json.dumps(data).encode("utf-8")
        self._send_bytes(
            content_bytes,
            content_type="application/json; charset=utf-8",
            status_code=status_code,
            extra_headers=extra_headers,
        )

    def _extract_session_token(self) -> Tuple[Optional[str], bool]:
        """
        Extracts session token from Authorization header, X-Admin-Token, or Cookie.
        Returns:
            (token, is_cookie_auth)
        """
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if token:
                return token, False

        custom_header = self.headers.get("X-Admin-Token")
        if custom_header:
            token = custom_header.strip()
            if token:
                return token, False

        cookie_header = self.headers.get("Cookie")
        if cookie_header:
            for part in cookie_header.split(";"):
                part = part.strip()
                if part.startswith("session_token="):
                    token = part.split("=", 1)[1].strip()
                    if token:
                        return token, True
        return None, False

    def _validate_origin(self) -> bool:
        """
        Validates Origin and Referer headers against trusted local hosts.
        Returns True if origin is valid or not provided (same-origin / direct API), False if untrusted.
        """
        origin = self.headers.get("Origin") or self.headers.get("Referer")
        if not origin:
            return True
        try:
            parsed = urllib.parse.urlparse(origin)
            host = parsed.netloc.split(":")[0].lower()
            trusted = {"127.0.0.1", "localhost"}
            if self.server and hasattr(self.server, "server_address") and self.server.server_address:
                server_host = str(self.server.server_address[0]).lower()
                if server_host and server_host != "0.0.0.0":
                    trusted.add(server_host)
            return host in trusted
        except Exception:
            return False

    def _authenticate_admin(self) -> Tuple[Optional[UserRecord], Optional[str], bool]:
        """
        Validates administrative authorization using a cryptographically random session token.
        Never trusts user ID or role headers directly.
        Returns (UserRecord, token, is_cookie_auth).
        """
        if not self.status_provider or not self.status_provider.session_manager:
            return None, None, False
        token, is_cookie_auth = self._extract_session_token()
        if not token:
            return None, None, False
        user = self.status_provider.session_manager.validate_session(
            token, self.status_provider.user_manager
        )
        return user, token, is_cookie_auth

    def _reject_unauthorized(self, status_code: int = 401, message: str = "Unauthorized: Administrator session token required."):
        """Sends HTTP 401/403 and dispatches UNAUTHORIZED_OP audit log."""
        if self.status_provider and self.status_provider.audit_logger:
            try:
                self.status_provider.audit_logger.log(
                    event_type=AuditEventType.UNAUTHORIZED_OP,
                    result=AuditResult.WARNING,
                    reason=f"Unauthorized access to administrative endpoint '{self.path}': {message}",
                    user_id="unauthenticated",
                    username="Unauthenticated",
                    source=AuditSource.SYSTEM,
                    lock_state="LOCKED",
                )
            except Exception:
                pass
        headers = {}
        if status_code == 401:
            headers["WWW-Authenticate"] = 'Bearer realm="Admin"'
        self._send_json({"error": message}, status_code=status_code, extra_headers=headers)

    def do_OPTIONS(self):
        self._send_bytes(b"", content_type="text/plain; charset=utf-8", status_code=200)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_bytes(DASHBOARD_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")

        elif self.path == "/api/status":
            snapshot = self.status_provider.get_snapshot() if self.status_provider else {}
            self._send_json(snapshot)

        elif self.path == "/api/accessibility/settings":
            settings = self.status_provider.get_accessibility_settings() if self.status_provider else {}
            self._send_json(settings)

        elif self.path == "/api/accessibility/feedback/history":
            hist = self.status_provider.voice_feedback.get_history() if (self.status_provider and self.status_provider.voice_feedback) else []
            self._send_json(hist)

        elif self.path == "/api/lock":
            snapshot = self.status_provider.get_snapshot() if self.status_provider else {}
            self._send_json(snapshot.get("lock", {}))

        elif self.path == "/api/commands":
            snapshot = self.status_provider.get_snapshot() if self.status_provider else {}
            self._send_json(snapshot.get("history", []))

        elif self.path == "/api/health":
            snapshot = self.status_provider.get_snapshot() if self.status_provider else {}
            self._send_json(snapshot.get("health", {}))

        elif self.path == "/api/analytics":
            snapshot = self.status_provider.get_snapshot() if self.status_provider else {}
            live_analytics = snapshot.get("analytics")
            if not live_analytics and self.status_provider and self.status_provider.analytics_collector:
                live_analytics = self.status_provider.analytics_collector.get_live_telemetry()
            benchmarks = self.status_provider.list_benchmarks() if self.status_provider else []
            self._send_json({
                "live": live_analytics or {},
                "benchmarks": benchmarks,
            })

        elif self.path.startswith("/api/admin/users"):
            admin_user, token, _ = self._authenticate_admin()
            if not admin_user:
                tok, _ = self._extract_session_token()
                code = 403 if tok else 401
                self._reject_unauthorized(status_code=code, message="Administrator session token required or invalid.")
                return
            users = self.status_provider.get_users() if self.status_provider else []
            self._send_json(users)

        elif self.path == "/api/admin/session":
            admin_user, _, _ = self._authenticate_admin()
            if admin_user:
                self._send_json({
                    "authenticated": True,
                    "user": admin_user.to_dict(include_sensitive=False)
                })
            else:
                self._send_json({"authenticated": False, "user": None})

        elif self.path.startswith("/api/audit"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                limit_val = max(1, min(1000, int(qs.get("limit", ["50"])[0]))) if qs.get("limit") else 50
            except (ValueError, TypeError):
                limit_val = 50

            try:
                offset_val = max(0, int(qs.get("offset", ["0"])[0])) if qs.get("offset") else 0
            except (ValueError, TypeError):
                offset_val = 0
            category_val = qs.get("category", [None])[0]
            result_val = qs.get("result", [None])[0]
            user_id_val = qs.get("user_id", [None])[0]
            event_type_val = qs.get("event_type", [None])[0]
            start_time_val = qs.get("start_time", [None])[0]
            end_time_val = qs.get("end_time", [None])[0]

            records = []
            if self.status_provider:
                records = self.status_provider.query_audit(
                    category=category_val,
                    result=result_val,
                    user_id=user_id_val,
                    event_type=event_type_val,
                    start_time=start_time_val,
                    end_time=end_time_val,
                    limit=limit_val,
                    offset=offset_val,
                )
            self._send_json(records)

        elif self.path.startswith("/api/access-analytics/summary"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                filters = AccessFilter(
                    start_date=qs.get("start_date", [None])[0],
                    end_date=qs.get("end_date", [None])[0],
                    user_id=qs.get("user_id", [None])[0],
                    role=qs.get("role", [None])[0],
                    event_type=qs.get("event_type", [None])[0],
                    result=qs.get("result", [None])[0],
                )
            except ValueError as val_err:
                self._send_json({"error": str(val_err)}, status_code=400)
                return

            summary = self.status_provider.get_access_analytics_summary(filters) if self.status_provider else {}
            self._send_json(summary)

        elif self.path.startswith("/api/access-analytics/report"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            fmt = str(qs.get("format", ["json"])[0]).lower()
            try:
                limit_val = max(1, min(10000, int(qs.get("limit", ["5000"])[0])))
            except (ValueError, TypeError):
                limit_val = 5000

            try:
                filters = AccessFilter(
                    start_date=qs.get("start_date", [None])[0],
                    end_date=qs.get("end_date", [None])[0],
                    user_id=qs.get("user_id", [None])[0],
                    role=qs.get("role", [None])[0],
                    event_type=qs.get("event_type", [None])[0],
                    result=qs.get("result", [None])[0],
                )
            except ValueError as val_err:
                self._send_json({"error": str(val_err)}, status_code=400)
                return

            report = self.status_provider.generate_access_report(filters, limit=limit_val) if self.status_provider else None
            if report is None:
                self._send_json({"error": "Failed to generate access report."}, status_code=500)
                return

            if fmt == "csv":
                csv_data = report.to_csv()
                self._send_bytes(
                    csv_data.encode("utf-8"),
                    content_type="text/csv; charset=utf-8",
                    extra_headers={"Content-Disposition": f'attachment; filename="{report.report_id}.csv"'},
                )
            else:
                json_data = report.to_json(indent=2)
                self._send_bytes(
                    json_data.encode("utf-8"),
                    content_type="application/json; charset=utf-8",
                    extra_headers={"Content-Disposition": f'attachment; filename="{report.report_id}.json"'},
                )

        else:
            self._send_bytes(b"Not Found", content_type="text/plain; charset=utf-8", status_code=404)

    def do_POST(self):
        if not (self.path.startswith("/api/admin/") or self.path == "/api/emergency/stop" or self.path == "/api/health/recover" or self.path == "/api/analytics/benchmark" or self.path.startswith("/api/access-analytics/") or self.path.startswith("/api/accessibility/")):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            body = json.loads(post_data) if post_data else {}
        except Exception:
            body = {}

        # ------------------------------------------------------------------
        # Accessibility Mode Endpoints (Settings & Voice Feedback)
        # ------------------------------------------------------------------
        if self.path == "/api/accessibility/settings":
            if self.status_provider:
                new_settings = self.status_provider.update_accessibility_settings(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(new_settings).encode("utf-8"))
            else:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"error": "Status provider unavailable"}')
            return

        elif self.path == "/api/accessibility/settings/reset":
            if self.status_provider:
                res = self.status_provider.reset_accessibility_settings()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            else:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"error": "Status provider unavailable"}')
            return

        elif self.path == "/api/accessibility/feedback/test":
            evt_name = str(body.get("event", "SYSTEM_READY"))
            if self.status_provider:
                res = self.status_provider.emit_test_voice_feedback(evt_name)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            else:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"error": "Status provider unavailable"}')
            return

        um = self.status_provider.user_manager if self.status_provider else None
        if um is None:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "User management service unavailable."}).encode("utf-8"))
            return

        # ------------------------------------------------------------------
        # Public Admin Login Endpoint (Rate Limited, Protected against PIN Brute-Force)
        # ------------------------------------------------------------------
        if self.path == "/api/admin/login":
            client_ip = self.client_address[0] if self.client_address else "127.0.0.1"
            raw_user_id = str(body.get("user_id", "")).strip()
            user_key = f"user:{raw_user_id.lower()}"
            ip_key = f"ip:{client_ip}"

            limiter = self.status_provider.login_rate_limiter if self.status_provider else None
            if limiter:
                allowed_u, rem_u = limiter.check_rate_limit(user_key)
                allowed_ip, rem_ip = limiter.check_rate_limit(ip_key)
                if not allowed_u or not allowed_ip:
                    retry_after = max(rem_u, rem_ip)
                    if self.status_provider and self.status_provider.audit_logger:
                        try:
                            self.status_provider.audit_logger.log(
                                event_type=AuditEventType.UNAUTHORIZED_OP,
                                result=AuditResult.WARNING,
                                reason=f"Administrator login rate limit exceeded. Cooldown active for {int(retry_after)}s.",
                                user_id=raw_user_id or "unauthenticated",
                                username=raw_user_id or "Unauthenticated",
                                source=AuditSource.SYSTEM,
                                lock_state="LOCKED",
                            )
                        except Exception:
                            pass
                    self.send_response(429)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Retry-After", str(int(retry_after)))
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": f"Too many failed login attempts. Please retry after {int(retry_after)} seconds."
                    }).encode("utf-8"))
                    return

            pin = body.get("pin")
            if not raw_user_id or not pin:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing user_id or pin"}).encode("utf-8"))
                return

            ok, user_rec, reason = um.verify_user_credentials(raw_user_id, pin)
            if not ok:
                if limiter:
                    limiter.record_failure(user_key)
                    limiter.record_failure(ip_key)
                if self.status_provider and self.status_provider.audit_logger:
                    try:
                        self.status_provider.audit_logger.log(
                            event_type=AuditEventType.USER_ACCESS_DENIED,
                            result=AuditResult.FAILURE,
                            reason=f"Administrator login failed for '{raw_user_id}': {reason}",
                            user_id=raw_user_id,
                            username=user_rec.username if user_rec else raw_user_id,
                            source=AuditSource.SYSTEM,
                            lock_state="LOCKED",
                        )
                    except Exception:
                        pass
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid user ID or PIN."}).encode("utf-8"))
                return

            if user_rec.role != UserRole.ADMIN:
                if limiter:
                    limiter.record_failure(user_key)
                    limiter.record_failure(ip_key)
                if self.status_provider and self.status_provider.audit_logger:
                    try:
                        self.status_provider.audit_logger.log(
                            event_type=AuditEventType.UNAUTHORIZED_OP,
                            result=AuditResult.WARNING,
                            reason=f"Non-admin user '{raw_user_id}' attempted administrator login (Role: {user_rec.role.value}).",
                            user_id=raw_user_id,
                            username=user_rec.username,
                            source=AuditSource.SYSTEM,
                            lock_state="LOCKED",
                        )
                    except Exception:
                        pass
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Forbidden: User does not have administrative privileges."}).encode("utf-8"))
                return

            # Success -> clear failures in rate limiter
            if limiter:
                limiter.record_success(user_key)
                limiter.record_success(ip_key)

            token = self.status_provider.session_manager.create_session(user_rec.user_id)
            session = self.status_provider.session_manager.get_session(token)
            csrf_token = session.csrf_token if session else ""
            if self.status_provider and self.status_provider.audit_logger:
                try:
                    self.status_provider.audit_logger.log(
                        event_type=AuditEventType.USER_ACCESS_GRANTED,
                        result=AuditResult.SUCCESS,
                        reason=f"Administrator '{user_rec.username}' logged in successfully.",
                        user_id=user_rec.user_id,
                        username=user_rec.username,
                        source=AuditSource.SYSTEM,
                        lock_state="LOCKED",
                    )
                except Exception:
                    pass

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Set-Cookie", f"session_token={token}; HttpOnly; SameSite=Strict; Path=/")
            self.send_header("Set-Cookie", f"csrf_token={csrf_token}; SameSite=Strict; Path=/")
            self.end_headers()
            self.wfile.write(
                json.dumps({
                    "authenticated": True,
                    "token": token,
                    "csrf_token": csrf_token,
                    "user": user_rec.to_dict(include_sensitive=False)
                }).encode("utf-8")
            )
            return

        # ------------------------------------------------------------------
        # Admin Logout Endpoint
        # ------------------------------------------------------------------
        if self.path == "/api/admin/logout":
            token, is_cookie_auth = self._extract_session_token()
            if is_cookie_auth:
                if not self._validate_origin():
                    self._reject_unauthorized(status_code=403, message="CSRF verification failed: Untrusted request Origin.")
                    return
                req_csrf = self.headers.get("X-CSRF-Token")
                session = self.status_provider.session_manager.get_session(token) if token else None
                if not req_csrf or not session or not session.csrf_token or not hmac.compare_digest(req_csrf.encode("utf-8"), session.csrf_token.encode("utf-8")):
                    self._reject_unauthorized(status_code=403, message="CSRF verification failed: Invalid or missing CSRF token.")
                    return

            if token and self.status_provider and self.status_provider.session_manager:
                self.status_provider.session_manager.revoke_session(token)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Set-Cookie", "session_token=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
            self.send_header("Set-Cookie", "csrf_token=; SameSite=Strict; Path=/; Max-Age=0")
            self.end_headers()
            self.wfile.write(json.dumps({"authenticated": False, "message": "Logged out successfully."}).encode("utf-8"))
            return

        # ------------------------------------------------------------------
        # Access Analytics Report Generation Endpoint
        # ------------------------------------------------------------------
        if self.path == "/api/access-analytics/report":
            fmt = str(body.get("format", "json")).lower()
            try:
                limit_val = max(1, min(10000, int(body.get("limit", 5000))))
            except (ValueError, TypeError):
                limit_val = 5000

            try:
                filters = AccessFilter(
                    start_date=body.get("start_date"),
                    end_date=body.get("end_date"),
                    user_id=body.get("user_id"),
                    role=body.get("role"),
                    event_type=body.get("event_type"),
                    result=body.get("result"),
                )
            except ValueError as val_err:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(val_err)}).encode("utf-8"))
                return

            report = self.status_provider.generate_access_report(filters, limit=limit_val) if self.status_provider else None
            if report is None:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Failed to generate access report."}).encode("utf-8"))
                return

            if fmt == "csv":
                csv_data = report.to_csv()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{report.report_id}.csv"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(csv_data.encode("utf-8"))
            else:
                json_data = report.to_json(indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{report.report_id}.json"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json_data.encode("utf-8"))
            return

        # ------------------------------------------------------------------
        # Admin Mutation Endpoints (Require Valid Authenticated ADMIN Session & CSRF Defense)
        # ------------------------------------------------------------------
        admin_user, token, is_cookie_auth = self._authenticate_admin()
        if not admin_user or not token:
            tok, _ = self._extract_session_token()
            code = 403 if tok else 401
            self._reject_unauthorized(status_code=code, message="Administrator session token required or invalid.")
            return

        # Explicit CSRF defense for cookie-authenticated mutations
        if is_cookie_auth:
            if not self._validate_origin():
                self._reject_unauthorized(status_code=403, message="CSRF verification failed: Untrusted request Origin.")
                return

            req_csrf = self.headers.get("X-CSRF-Token")
            session = self.status_provider.session_manager.get_session(token)
            if not req_csrf or not session or not session.csrf_token:
                self._reject_unauthorized(status_code=403, message="CSRF verification failed: Missing required X-CSRF-Token header.")
                return

            if not hmac.compare_digest(req_csrf.encode("utf-8"), session.csrf_token.encode("utf-8")):
                self._reject_unauthorized(status_code=403, message="CSRF verification failed: Invalid CSRF token.")
                return
        else:
            # Even for Bearer auth, if Origin is present, reject untrusted origins
            if not self._validate_origin():
                self._reject_unauthorized(status_code=403, message="CSRF verification failed: Untrusted request Origin.")
                return

        admin_id = admin_user.user_id

        try:
            if self.path == "/api/emergency/stop":
                eh = self.status_provider.emergency_handler if self.status_provider else None
                if eh is None:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Emergency handler service unavailable."}).encode("utf-8"))
                    return

                reason_text = body.get("reason", f"Dashboard emergency stop initiated by {admin_user.username}")
                ok, snap = eh.trigger_emergency_stop(
                    source=AuditSource.DASHBOARD,
                    reason=reason_text,
                    user_id=admin_user.user_id,
                    username=admin_user.username,
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": ok,
                    "message": "Emergency STOP executed. Simulated barrier is LOCKED.",
                    "lock": snap.to_dict(),
                    "emergency": eh.get_emergency_status(),
                }).encode("utf-8"))
                return

            elif self.path == "/api/health/recover":
                rm = self.status_provider.recovery_manager if self.status_provider else None
                if rm is None:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Recovery manager service unavailable."}).encode("utf-8"))
                    return

                comp = body.get("component", "all")
                results = {}
                if comp in ("all", "microphone", "audio"):
                    ok_mic, msg_mic = rm.recover_audio_subsystem()
                    results["microphone"] = {"success": ok_mic, "message": msg_mic}
                if comp in ("all", "inference"):
                    ok_inf, msg_inf = rm.recover_inference_subsystem()
                    results["inference"] = {"success": ok_inf, "message": msg_inf}
                if comp in ("all", "database"):
                    ok_db, msg_db = rm.recover_database_subsystem()
                    results["database"] = {"success": ok_db, "message": msg_db}
                if comp in ("all", "dashboard"):
                    ok_dash, msg_dash = rm.recover_dashboard_subsystem()
                    results["dashboard"] = {"success": ok_dash, "message": msg_dash}

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": all(r.get("success", True) for r in results.values()),
                    "results": results,
                    "health": self.status_provider.get_snapshot().get("health", {}),
                }).encode("utf-8"))
                return

            elif self.path == "/api/analytics/benchmark":
                # Protected administrative benchmark execution
                runner = self.status_provider.workload_runner if self.status_provider else None
                if runner is None:
                    if self.status_provider and self.status_provider.analytics_collector:
                        from src.analytics.workloads import WorkloadRunner
                        runner = WorkloadRunner(collector=self.status_provider.analytics_collector)
                        self.status_provider.workload_runner = runner
                    else:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Analytics workload runner unavailable."}).encode("utf-8"))
                        return

                from src.rts.scheduler import SchedulingPolicy
                from src.analytics.types import WorkloadType

                pol_str = str(body.get("policy", "EDF")).upper()
                pol = SchedulingPolicy.EDF
                if pol_str == "FIFO":
                    pol = SchedulingPolicy.FIFO
                elif pol_str == "RM":
                    pol = SchedulingPolicy.RM

                workload_str = str(body.get("workload", "NORMAL")).upper()
                workload = WorkloadType.NORMAL
                if workload_str == "HIGH":
                    workload = WorkloadType.HIGH
                elif workload_str == "OVERLOAD":
                    workload = WorkloadType.OVERLOAD
                elif workload_str == "RECOVERY":
                    workload = WorkloadType.RECOVERY

                num_blocks = max(5, min(500, int(body.get("num_blocks", 50))))
                try:
                    record = runner.run_workload(policy=pol, workload_type=workload, num_blocks=num_blocks)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "success": True,
                        "benchmark": record.to_dict(),
                        "summary": record.to_flat_csv_row(),
                    }).encode("utf-8"))
                except Exception as ex:
                    logger.error(f"[ANALYTICS] Benchmark execution failed: {ex}", exc_info=True)
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Benchmark failed: {ex}"}).encode("utf-8"))
                return

            elif self.path == "/api/admin/users/create":
                user_id = body.get("user_id")
                username = body.get("username")
                pin = body.get("pin")
                role = body.get("role", "REGULAR_USER")
                expires_at = body.get("expires_at")
                if not user_id or not username or not pin:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing required fields: user_id, username, pin"}).encode("utf-8"))
                    return
                new_user = um.create_user(
                    admin_user_id=admin_id,
                    user_id=user_id,
                    username=username,
                    pin=pin,
                    role=role,
                    expires_at=expires_at,
                )
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(new_user.to_dict(include_sensitive=False)).encode("utf-8"))

            elif self.path == "/api/admin/users/disable":
                target_user_id = body.get("target_user_id")
                updated = um.disable_user(admin_user_id=admin_id, target_user_id=target_user_id)
                if self.status_provider and self.status_provider.session_manager:
                    self.status_provider.session_manager.revoke_all_for_user(target_user_id)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(updated.to_dict(include_sensitive=False)).encode("utf-8"))

            elif self.path == "/api/admin/users/enable":
                target_user_id = body.get("target_user_id")
                updated = um.enable_user(admin_user_id=admin_id, target_user_id=target_user_id)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(updated.to_dict(include_sensitive=False)).encode("utf-8"))

            elif self.path == "/api/admin/users/remove":
                target_user_id = body.get("target_user_id")
                um.remove_user(admin_user_id=admin_id, target_user_id=target_user_id)
                if self.status_provider and self.status_provider.session_manager:
                    self.status_provider.session_manager.revoke_all_for_user(target_user_id)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "message": f"User '{target_user_id}' removed."}).encode("utf-8"))

            elif self.path == "/api/admin/users/change-role":
                target_user_id = body.get("target_user_id")
                new_role = body.get("role")
                updated = um.change_user_role(admin_user_id=admin_id, target_user_id=target_user_id, new_role=new_role)
                if updated.role != UserRole.ADMIN and self.status_provider and self.status_provider.session_manager:
                    self.status_provider.session_manager.revoke_all_for_user(target_user_id)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(updated.to_dict(include_sensitive=False)).encode("utf-8"))

            elif self.path == "/api/admin/users/reset-pin":
                target_user_id = body.get("target_user_id")
                new_pin = body.get("new_pin")
                updated = um.reset_user_pin(admin_user_id=admin_id, target_user_id=target_user_id, new_pin=new_pin)
                if self.status_provider and self.status_provider.session_manager:
                    self.status_provider.session_manager.revoke_all_for_user(target_user_id)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(updated.to_dict(include_sensitive=False)).encode("utf-8"))

            elif self.path == "/api/admin/users/set-expiration":
                target_user_id = body.get("target_user_id")
                expires_at = body.get("expires_at")
                updated = um.set_guest_expiration(admin_user_id=admin_id, target_user_id=target_user_id, expires_at=expires_at)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(updated.to_dict(include_sensitive=False)).encode("utf-8"))

            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")

        except (ValueError, PermissionError) as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        except Exception as e:
            logger.error(f"[DASHBOARD] Admin endpoint internal error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Internal server error."}).encode("utf-8"))


class RobustHTTPServer(ThreadingHTTPServer):
    """
    Multi-threaded HTTP server subclass with allow_reuse_address=True to ensure port re-bindings
    reliably succeed without TIME_WAIT stalls across Windows and POSIX.
    Dispatches each request to a daemon thread to prevent socket starvation or browser keep-alive hangs.
    """
    allow_reuse_address = True
    daemon_threads = True


class DashboardServer:
    """
    Embedded HTTP Server managing the live monitoring dashboard.
    Runs asynchronously in a background daemon thread.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, provider: Optional[DashboardStatusProvider] = None):
        self.host = host
        self.port = port
        self.provider = provider or DashboardStatusProvider()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._is_running = False

    def start(self):
        """Starts HTTP server in background thread."""
        if self._is_running:
            return

        DashboardRequestHandler.status_provider = self.provider

        # Attempt port binding with fallback
        for candidate_port in [self.port, self.port + 1, self.port + 2, 8888, 0]:
            try:
                self._server = RobustHTTPServer((self.host, candidate_port), DashboardRequestHandler)
                self.port = self._server.server_port
                break
            except OSError:
                continue

        if self._server is None:
            raise RuntimeError(f"Could not bind DashboardServer to {self.host}:{self.port}")

        self._is_running = True
        self._thread = threading.Thread(target=self._run_server, name="DashboardServer", daemon=True)
        self._thread.start()
        logger.info(f"[DASHBOARD] Monitoring dashboard live at http://{self.host}:{self.port}")

    def _run_server(self):
        if self._server:
            try:
                self._server.serve_forever()
            except Exception:
                pass

    def stop(self):
        """Stops the dashboard server cleanly."""
        if not self._is_running:
            return
        self._is_running = False
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        logger.info("[DASHBOARD] Monitoring dashboard stopped cleanly.")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
