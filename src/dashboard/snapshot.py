"""
src/dashboard/snapshot.py
-------------------------
Visual Dashboard Screenshot Generator.
Renders high-resolution graphical monitoring dashboard snapshots using matplotlib.
Saves directly to results/dashboard_screenshot.png for reports and UI verification.
"""

import os
import pathlib
import time
from typing import Any, Dict, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def render_dashboard_snapshot(
    snapshot: Dict[str, Any],
    output_path: str | pathlib.Path = "results/dashboard_screenshot.png",
    dpi: int = 150,
) -> pathlib.Path:
    """
    Renders the complete monitoring dashboard into a high-resolution PNG image.
    Accurately visualizes all 11 required metrics with clear units and timestamps.
    """
    out_file = pathlib.Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Figure setup (16:9 ratio, dark theme)
    fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Outer border / header
    ax_head = fig.add_axes([0.03, 0.89, 0.94, 0.09], facecolor="#161b22")
    ax_head.axis("off")
    ax_head.text(0.02, 0.65, "REAL-TIME ADAPTIVE VOICE RECOGNITION & RTOS MONITOR", color="#ffffff", fontsize=15, weight="bold")
    ax_head.text(0.02, 0.22, "Closed-Loop Deadline-Aware Scheduling, Ping-Pong Buffering & Adaptive Audio Processing", color="#8b949e", fontsize=10)

    ts = snapshot.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    ax_head.text(0.98, 0.60, f"STATUS: LIVE  |  {ts}", color="#3fb950", fontsize=11, weight="bold", ha="right")

    lock_info = snapshot.get("lock", {})
    if lock_info:
        l_state = lock_info.get("state", "LOCKED")
        l_col = "#3fb950" if l_state == "UNLOCKED" else ("#58a6ff" if l_state == "PIN_ENTRY" else "#f85149")
        l_text = f"LOCK: {l_state}  |  PIN: {lock_info.get('pin_progress', '0/4')}  |  RESULT: {lock_info.get('last_access_result', 'NONE')}"
        ax_head.text(0.98, 0.22, l_text, color=l_col, fontsize=10, weight="bold", ha="right")

    # Grid parameters: 2 rows of 3 cards
    card_w, card_h = 0.295, 0.35
    gap_x, gap_y = 0.025, 0.04
    x0, y0 = 0.03, 0.49

    cards_data = [
        # (title, metrics_list, border_color)
        (
            "1. MICROPHONE & INGRESS",
            [
                ("Target Device", str(snapshot.get("microphone", {}).get("device_name", "Virtual Stream"))[:24]),
                ("Device ID", str(snapshot.get("microphone", {}).get("device_id", "N/A"))),
                ("Sample Rate", f"{snapshot.get('microphone', {}).get('sample_rate', 16000)} Hz (Mono)"),
                ("Queue Occupancy", f"{snapshot.get('microphone', {}).get('queue_occupancy', 0)} / {snapshot.get('microphone', {}).get('queue_capacity', 100)} blocks"),
                ("Dropped Audio Blocks", f"{snapshot.get('microphone', {}).get('audio_blocks_dropped', 0)} blocks"),
                ("Noise Floor RMS", f"{snapshot.get('dsp', {}).get('noise_floor_rms', 0.0):.6f}" if snapshot.get('dsp', {}).get('noise_floor_rms') is not None else "N/A"),
            ],
            "#58a6ff",
        ),
        (
            "2. COMMAND FSM STATE",
            [
                ("FSM State", str(snapshot.get("command_fsm", {}).get("current_state", "WAITING"))),
                ("Latest Command", f"'{snapshot.get('command_fsm', {}).get('latest_command')}'" + (f" ({snapshot.get('command_fsm', {}).get('latest_digit')})" if snapshot.get('command_fsm', {}).get('latest_digit') else "") if snapshot.get("command_fsm", {}).get("latest_command") else "N/A"),
                ("Confidence Score", f"{snapshot.get('command_fsm', {}).get('latest_confidence', 0.0)*100:.1f}%" if snapshot.get("command_fsm", {}).get("latest_confidence") is not None else "N/A"),
                ("Total Confirmed", f"{snapshot.get('command_fsm', {}).get('total_confirmed', 0)}"),
                ("Total Rejected", f"{snapshot.get('command_fsm', {}).get('total_rejected', 0)}"),
                ("Vocabulary Class", "19 Commands (10 Core + 9 Numbers)"),
            ],
            "#3fb950",
        ),
        (
            "3. REAL-TIME LATENCY (ms)",
            [
                ("Reaction Latency", f"{snapshot.get('timing', {}).get('processing_reaction_latency_ms', 0.0):.2f} ms" if snapshot.get('timing', {}).get('processing_reaction_latency_ms') is not None else "N/A"),
                ("Acoustic Latency", f"{snapshot.get('timing', {}).get('total_acoustic_latency_ms', 0.0):.2f} ms" if snapshot.get('timing', {}).get('total_acoustic_latency_ms') is not None else "N/A"),
                ("Model Inference", f"{snapshot.get('timing', {}).get('model_inference_ms', 0.0):.2f} ms" if snapshot.get('timing', {}).get('model_inference_ms') is not None else "N/A"),
                ("Log-Mel Extraction", f"{snapshot.get('timing', {}).get('feature_extraction_ms', 0.0):.2f} ms" if snapshot.get('timing', {}).get('feature_extraction_ms') is not None else "N/A"),
                ("Queue Delay", f"{snapshot.get('timing', {}).get('queue_delay_ms', 0.0):.2f} ms" if snapshot.get('timing', {}).get('queue_delay_ms') is not None else "N/A"),
                ("Measurement Base", "Dequeue -> Confirmed Decision"),
            ],
            "#bc8cff",
        ),
        (
            "4. RTOS TASK SCHEDULER",
            [
                ("Active Policy", str(snapshot.get("scheduler", {}).get("policy", "EDF"))),
                ("Ready Jobs Queue", f"{snapshot.get('scheduler', {}).get('ready_jobs_count', 0)} jobs"),
                ("Total Jobs Executed", f"{snapshot.get('scheduler', {}).get('total_executed', 0)}"),
                ("Total Deadline Misses", f"{snapshot.get('scheduler', {}).get('total_deadline_misses', 0)}"),
                ("Deadline Miss Ratio", f"{snapshot.get('scheduler', {}).get('deadline_miss_ratio_pct', 0.0):.2f}%"),
                ("Late Job Strategy", "EXECUTE_AND_RECORD"),
            ],
            "#d29922",
        ),
        (
            "5. ADAPTIVE CONTROLLER",
            [
                ("Controller State", str(snapshot.get("adaptive_controller", {}).get("current_state", "NORMAL"))),
                ("State Transitions", f"{snapshot.get('adaptive_controller', {}).get('total_state_transitions', 0)}"),
                ("Stale Jobs Pruned", f"{snapshot.get('adaptive_controller', {}).get('total_stale_jobs_pruned', 0)}"),
                ("Inferences Suppressed", f"{snapshot.get('adaptive_controller', {}).get('total_inferences_suppressed', 0)}"),
                ("Mean Queue Delay", f"{snapshot.get('adaptive_controller', {}).get('mean_recent_queue_delay_ms', 0.0):.2f} ms"),
                ("Telemetry Cadence", "Dynamic (200-1000 ms)"),
            ],
            "#58a6ff",
        ),
        (
            "6. HOST HARDWARE RESOURCES",
            [
                ("Process CPU Utilization", f"{snapshot.get('resources', {}).get('cpu_utilization_pct', 0.0):.1f}%"),
                ("Resident Memory (RSS)", f"{snapshot.get('resources', {}).get('memory_rss_mb', 0.0):.1f} MB"),
                ("Execution Architecture", "Python 3.12 (PyTorch CPU)"),
                ("Input Tensor Shape", "(1, 1, 49, 40) Float32"),
                ("Audio Hop / Frame", "20.0 ms / 40.0 ms"),
                ("OS Platform", "Windows 11 x64"),
            ],
            "#f85149",
        ),
    ]

    for idx, (title, metrics, border_color) in enumerate(cards_data):
        row = idx // 3
        col = idx % 3
        cx = x0 + col * (card_w + gap_x)
        cy = y0 - row * (card_h + gap_y)

        ax = fig.add_axes([cx, cy, card_w, card_h], facecolor="#161b22")
        ax.axis("off")

        # Card Border
        rect = patches.FancyBboxPatch(
            (0.01, 0.01), 0.98, 0.98,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#30363d",
            facecolor="#161b22",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Header
        ax.text(0.06, 0.88, title, color=border_color, fontsize=10.5, weight="bold", transform=ax.transAxes)

        # Rows
        n_rows = len(metrics)
        for r_i, (k, v) in enumerate(metrics):
            y_pos = 0.72 - (r_i * (0.64 / n_rows))
            ax.text(0.06, y_pos, k, color="#8b949e", fontsize=9, transform=ax.transAxes)
            ax.text(0.94, y_pos, str(v), color="#ffffff", fontsize=9, weight="bold", ha="right", transform=ax.transAxes)

    # Bottom Table: Recent Commands History
    ax_tbl = fig.add_axes([0.03, 0.04, 0.94, 0.28], facecolor="#161b22")
    ax_tbl.axis("off")
    tbl_rect = patches.FancyBboxPatch(
        (0.005, 0.005), 0.99, 0.99,
        boxstyle="round,pad=0.01",
        linewidth=1.2,
        edgecolor="#30363d",
        facecolor="#161b22",
        transform=ax_tbl.transAxes,
    )
    ax_tbl.add_patch(tbl_rect)

    ax_tbl.text(0.02, 0.88, "RECENT COMMAND DECISION LOG & VERIFICATION AUDIT", color="#ffffff", fontsize=11, weight="bold", transform=ax_tbl.transAxes)

    # Table columns
    col_x = [0.03, 0.18, 0.35, 0.50, 0.68, 0.84]
    headers = ["TIMESTAMP", "COMMAND", "CONFIDENCE", "MARGIN", "REACTION LATENCY", "STATUS"]
    for i, h in enumerate(headers):
        ax_tbl.text(col_x[i], 0.76, h, color="#8b949e", fontsize=8.5, weight="bold", transform=ax_tbl.transAxes)

    history = snapshot.get("history", [])
    recent_history = history[-5:] if history else []

    if recent_history:
        for r_i, rec in enumerate(reversed(recent_history)):
            y_r = 0.60 - (r_i * 0.12)
            t_str = time.strftime("%H:%M:%S", time.localtime(rec.get("detected_timestamp", time.time())))
            cmd_str = str(rec.get("command") or "N/A")
            conf_str = f"{rec.get('confidence', 0.0)*100:.1f}%"
            margin_str = f"{rec.get('margin', 0.0):.2f}"
            lat_str = f"{rec.get('reaction_latency_ms', 0.0):.2f} ms" if rec.get("reaction_latency_ms") is not None else "N/A"
            stat_str = "CONFIRMED" if rec.get("is_confirmed") else f"REJECTED ({rec.get('rejection_reason', 'UNKNOWN')})"
            stat_color = "#3fb950" if rec.get("is_confirmed") else "#d29922"

            ax_tbl.text(col_x[0], y_r, t_str, color="#ffffff", fontsize=8.5, transform=ax_tbl.transAxes)
            ax_tbl.text(col_x[1], y_r, cmd_str, color="#58a6ff", fontsize=8.5, weight="bold", transform=ax_tbl.transAxes)
            ax_tbl.text(col_x[2], y_r, conf_str, color="#ffffff", fontsize=8.5, transform=ax_tbl.transAxes)
            ax_tbl.text(col_x[3], y_r, margin_str, color="#ffffff", fontsize=8.5, transform=ax_tbl.transAxes)
            ax_tbl.text(col_x[4], y_r, lat_str, color="#ffffff", fontsize=8.5, transform=ax_tbl.transAxes)
            ax_tbl.text(col_x[5], y_r, stat_str, color=stat_color, fontsize=8.5, weight="bold", transform=ax_tbl.transAxes)
    else:
        ax_tbl.text(0.50, 0.45, "No command events recorded yet (Awaiting speech activity)", color="#8b949e", fontsize=10, ha="center", transform=ax_tbl.transAxes)

    plt.savefig(str(out_file), dpi=dpi, facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    return out_file

