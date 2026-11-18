# Voice-Authenticated Real-Time Smart Access Control System

**Sub-title:** A Soft Real-Time Voice-Authenticated Access Control System Featuring Deadline-Aware Scheduling, PBKDF2 Security, Multi-Role RBAC, Fault Recovery, and High-Priority Emergency Preemption.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/Tests-251%20Passed%20(100%25)-brightgreen.svg)]()
[![Scheduling](https://img.shields.io/badge/Scheduling-FIFO%20%7C%20RM%20%7C%20EDF-blueviolet.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey.svg)]()

---

## 1. Executive Summary & Project Architecture

This application is an end-to-end, voice-driven smart access-control and security appliance built entirely in Python. Combining acoustic digital signal processing (DSP), edge deep learning keyword spotting (Residual DS-CNN), real-time task scheduling (FIFO, Rate Monotonic, Earliest Deadline First), cryptographic security (PBKDF2-SHA256, session authentication, CSRF tokens), and closed-loop fault tolerance, the project provides a fully integrated voice-based physical entry simulation.

### Multi-Stage Pipeline Architecture

```
                                [ Physical Microphone / Stream Ingress ]
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Ingress & Buffering: Ping-Pong Double Buffer (320-sample PCM16) ➔ Bounded Queue (Cap 100)      │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Acoustic DSP & VAD: Adaptive Noise Floor ➔ Circular Pre-Roll (200ms) ➔ 3-State Energy VAD      │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Soft Real-Time Task Scheduler: Task Queue ➔ Policies: FIFO / RM / EDF ➔ Worker Thread           │
│ Jobs: AudioBlock (T=20ms) ➔ FeatureExtract (D=40ms) ➔ Inference (D=100ms) ➔ Monitoring (T=200)│
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Edge Deep Learning: 40 Mel Spectrogram Extractor ➔ Residual DS-CNN (21-Class Spoken Vocabulary) │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Command State Machine: WAITING ➔ SPEECH_DETECTED ➔ CANDIDATE ➔ CONFIRMED ➔ COOLDOWN             │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                       ┌─────────────────────────┴─────────────────────────┐
                       ▼                                                   ▼
┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────────────┐
│ Smart Lock & Security Subsystem              │  │ Safety, Preemption & Fault Subsystems        │
│ ├── Masked PIN Ingress Buffer                │  │ ├── Emergency STOP Preemption (Instant Lock) │
│ ├── PBKDF2-SHA256 Multi-User RBAC Engine     │  │ ├── Fault Detector (Audio, Sched, ML, DB)    │
│ │   (Admin, Regular User, Expiring Guest)    │  │ ├── Recovery Manager (Automated Self-Heal)   │
│ ├── Fail-Secure Lock Controller              │  │ └── Adaptive Controller (Overload Shedding)  │
│ └── Lockout Manager (Max 3 failed attempts)  │  └──────────────────────┬───────────────────────┘
└──────────────────────┬───────────────────────┘                         │
                       │                                                 │
                       └─────────────────────────┬───────────────────────┘
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Telemetry, Analytics & Presentation Layer                                                        │
│ ├── Immutable SQLite Security Audit Logger (WAL Mode, Parameterized Queries)                   │
│ ├── Real-Time Scheduling Performance Analytics (Latency, DMR %, CPU, Memory, Queue)           │
│ ├── Access Analytics & CSV/JSON Export Engine                                                   │
│ ├── Accessibility Engine (High-Contrast, Large-Text, Spoken Voice Feedback, Screen-Reader)      │
│ └── Local Web Monitoring Dashboard (Zero-dependency HTTP server, port 8080)                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Integrated Subsystems (Stages 1 – 10)

| Stage | Subsystem | Key Components & Files | Primary Responsibility |
| :---: | :--- | :--- | :--- |
| **1** | **Smart Lock Simulation** | [`src/security/lock_controller.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/security/lock_controller.py)<br>[`src/security/pin_manager.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/security/pin_manager.py) | Fail-secure finite-state machine (`LOCKED`, `PIN_ENTRY`, `UNLOCKED`, `LOCKED_OUT`, `EMERGENCY_LOCKED`). In-memory masked PIN buffering, auto-relock timers, and lockout management. |
| **2** | **Security Audit Logging** | [`src/security/audit_logger.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/security/audit_logger.py) | Non-blocking, worker-thread SQLite audit logger (`data/security_audit.db`). Structured JSON metadata, event severity levels (`INFO`, `WARNING`, `SECURITY_ALERT`, `CRITICAL`), and strict parameterization. |
| **3** | **RBAC User Management** | [`src/security/user_manager.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/security/user_manager.py)<br>[`src/dashboard/server.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/dashboard/server.py) | Role-Based Access Control (`ADMIN`, `USER`, `GUEST`). PBKDF2-SHA256 salted PIN storage (100,000 iterations), unique PIN constraints, guest expiration timestamps, CSRF tokens, and brute-force rate limiters. |
| **4** | **Emergency STOP** | [`src/security/emergency_handler.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/security/emergency_handler.py) | Zero-latency emergency command preemption. Utterance `"stop"` immediately purges in-flight PIN buffers, invalidates pending unlock tickets, forces `LOCKED` state, and dispatches high-priority audit alerts. |
| **5** | **Fault Detection & Recovery** | [`src/fault/detector.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/fault/detector.py)<br>[`src/fault/recovery.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/fault/recovery.py) | Monitors audio buffer overflows, inference execution timeouts, audio read failures, and scheduler errors. Performs automated self-healing without application crashes. |
| **6** | **Performance Analytics** | [`src/analytics/collector.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/analytics/collector.py)<br>[`src/analytics/workloads.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/analytics/workloads.py) | Captures 5 discrete end-to-end latency boundaries, task execution time ($C_i$), response time ($R_i$), deadline miss ratio (`DMR %`), and process CPU/RSS memory footprints. |
| **7** | **Access Analytics & Reports** | [`src/analytics/access_analytics.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/analytics/access_analytics.py) | Aggregates authentication success/failure ratios, hourly traffic curves, user activity rankings, and generates downloadable audit reports in JSON and CSV formats. |
| **8** | **Accessibility Mode** | [`src/accessibility/settings.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/accessibility/settings.py)<br>[`src/accessibility/feedback.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/accessibility/feedback.py) | High-contrast visual palettes, enlarged typography, screen-reader semantic ARIA markup, configurable spoken voice feedback (synthesized/pre-recorded audio), and safe PIN timeout extension. |
| **9** | **System Integration** | [`src/application/orchestrator.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/src/application/orchestrator.py) | Unified life-cycle coordination bridging real-time audio threads, scheduler workers, audit queues, HTTP dashboard services, and security managers into a single robust runtime. |
| **10** | **Final Validation & Benchmarking** | [`scripts/final_project_validation.py`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/scripts/final_project_validation.py) | Complete empirical benchmarking suite across FIFO, RM, and EDF under normal, high, and overload stress profiles; automated generation of 10 publication charts and 251 unit/integration tests. |

---

## 3. Installation & Quickstart

### Prerequisites
- **Operating System:** Windows 10 or Windows 11 (64-bit)
- **Python:** Python 3.12 (Tested on Python 3.12.10 x64)
- **Hardware:** Microphone (physical array, USB headset, or virtual audio device)

### Setup (PowerShell)

```powershell
# 1. Clone repository and navigate to root directory
cd c:\Users\golu6\Desktop\RTOS_SHUBHAM

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Allow local script execution (if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 4. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 5. Install all dependencies
pip install -r requirements.txt
```

---

## 4. Running the Application

### A. Live Interactive Voice Mode with Web Dashboard
To launch the complete real-time application with live microphone capture and the interactive web interface:

```powershell
.\.venv\Scripts\python.exe -m src.application.orchestrator
```
*Access the Web Dashboard at:* **`http://127.0.0.1:8080/`**

### B. Automated Headless System Benchmarking
To run the automated empirical benchmark matrix and generate all 10 validation charts:

```powershell
.\.venv\Scripts\python.exe scripts/final_project_validation.py
```

### C. Running Regression Tests
To execute the comprehensive 251-test verification suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
*(All 251 tests pass cleanly across unit, integration, and security test files).*

---

## 5. Security & RBAC Design

The security model is architected to resist standard physical and software access-control attacks:

1. **Zero Plaintext Storage:** PINs are never stored in plaintext on disk, in SQLite databases, or in memory. All PIN credentials use **PBKDF2-SHA256** with 100,000 iterations and 16-byte cryptographically secure random salts.
2. **Duplicate PIN Prevention:** Duplicate PIN assignment is atomically checked and prevented across all active users.
3. **Role-Based Access Control (RBAC):**
   - **`ADMIN`:** Full administrative privileges (create/disable users, reset PINs, inspect security audits, change settings).
   - **`USER`:** Standard authentication privileges for unlocking the lock.
   - **`GUEST`:** Ephemeral access with mandatory `expires_at` timestamp. Once expired, access is instantly denied.
4. **Lockout Policy:** Three consecutive incorrect PIN attempts trigger an automatic 30-second lockout period (`LOCKED_OUT`), during which all voice input is ignored.
5. **Session Security & CSRF Defense:** Admin sessions are authenticated via cryptographically random 32-byte tokens with `HttpOnly`, `SameSite=Strict` cookies, constant-time verification, per-session CSRF tokens, and rate limiting (5 failed attempts per 30-second window).
6. **Fail-Secure Architecture:** Any crash, restart, or power-loss event guarantees that the smart lock boots into the strictly `LOCKED` state.

---

## 6. Real-Time Task Scheduling (FIFO vs RM vs EDF)

The application implements a soft real-time task scheduler to prevent high computational loads from causing audio frame drops or latency spikes:

- **FIFO (First-In, First-Out):** Baseline policy serving tasks strictly in order of arrival.
- **RM (Rate Monotonic):** Static priority policy where shorter-period recurring tasks receive higher execution priority ($T_{audio} = 20\text{ ms} > T_{monitor} = 200\text{ ms}$).
- **EDF (Earliest Deadline First):** Dynamic priority policy scheduling tasks by earliest absolute deadline ($d_i = t_{release} + D_i$). Proven optimal on preemptive uniprocessors.

### Benchmark Comparison Matrix (Empirical Host Measurements)

| Workload Profile | Policy | Mean Latency (ms) | P95 Latency (ms) | Max Latency (ms) | Deadline Miss Ratio (%) | Process CPU (%) | RSS Memory (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal (Nominal)** | **FIFO** | 4.97 | 21.85 | 55.81 | 2.27% | 52.1% | 273.5 |
| | **RM** | **4.18** | **16.91** | **35.38** | **1.14%** | 102.5% | 277.9 |
| | **EDF** | 6.53 | 32.29 | 59.34 | 3.95% | 52.1% | 283.0 |
| **High (2x Ingress)** | **FIFO** | **8.86** | 37.17 | **52.67** | **7.54%** | 101.0% | 286.4 |
| | **RM** | 14.25 | 68.66 | 100.29 | 17.38% | 0.0% | 290.6 |
| | **EDF** | 10.03 | 37.44 | 53.86 | 13.67% | 14.4% | 294.4 |
| **Overload (Contention)**| **FIFO** | 282.33 | 389.48 | 2681.12 | 36.21% | 554.6% | 297.9 |
| | **RM** | 161.91 | 270.66 | 1500.56 | 35.14% | 391.4% | 302.7 |
| | **EDF** | **2.08** | **3.70** | **5.24** | **0.00%** | 484.0% | 304.9 |

> [!NOTE]
> Under heavy overload and thread contention, **EDF dynamically prioritizes near-deadline jobs**, preventing queue starvation and achieving a **0.00% deadline miss ratio**, whereas static policies suffer queue backpressure.

---

## 7. Deep Learning Model (Residual DS-CNN)

- **Architecture:** Residual Depthwise Separable CNN (DS-CNN) with 28,693 parameters (< 120 KB memory footprint).
- **Acoustic Input:** 40-channel Log-Mel spectrogram computed over 1.0-second circular audio frames (49 frames $\times$ 40 Mel bins).
- **Forward-Pass Latency:** $\sim 2.3\text{ ms}$ on host CPU (sub-3 ms execution comfortably fulfills the 100 ms inference deadline).
- **Target Vocabulary (21 Classes):**
  - **10 Core Commands:** `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`
  - **9 Spoken Digits:** `one`, `two`, `three`, `four`, `five`, `six`, `seven`, `eight`, `nine`
  - **2 Background/Noise Classes:** `_unknown_`, `_silence_`
- **Offline Test Accuracy:** **94.01%** on official Google Speech Commands v0.02 test split (8,563 held-out samples). Spoken digits achieve a mean F1-score of **95.4%**.

---

## 8. Repeatable College Demonstration Script

Follow this 16-step sequence for an interactive live demonstration:

1. **Boot Application:** Run `.\.venv\Scripts\python.exe -m src.application.orchestrator`.
2. **Open Dashboard:** Navigate to `http://127.0.0.1:8080/`. Point out the initial state: Lock is `LOCKED`, VAD is `WAITING`, Health is `HEALTHY`.
3. **Spoken Digit Buffer Entry:** Speak clearly into the microphone: *"ONE"*. Observe the dashboard transition to `PIN_ENTRY` with `Progress: 1/4 (Masked: *)`.
4. **Buffer Clearing Timeout:** Remain silent for 10 seconds. Observe the PIN entry inactivity timeout: buffer resets to empty and state returns to `LOCKED`.
5. **Successful Unlock:** Speak the 4-digit default administrator PIN: *"ONE"*, pause, *"TWO"*, pause, *"THREE"*, pause, *"FOUR"*.
6. **Access Granted State:** Observe lock transition to `UNLOCKED` (Green banner, audio chime, audit logged).
7. **Automatic Relocking:** Wait 5 seconds without speaking. Observe the lock automatically transitioning from `UNLOCKED` back to `LOCKED`.
8. **Failed Attempt:** Speak an incorrect sequence: *"NINE"*, *"NINE"*, *"NINE"*, *"NINE"*. Observe rejection banner (`Access Denied - Incorrect PIN`, attempt count 1/3).
9. **Lockout Trigger:** Enter incorrect PINs two more times (total 3 failed attempts).
10. **Lockout Enforcement:** Observe transition to `LOCKED_OUT` (Purple banner, 30-second cooldown timer). Speak *"ONE"*; note that the system completely ignores input during lockout.
11. **Emergency STOP Preemption:** After lockout expires, speak *"ONE"*, *"TWO"* (enter `PIN_ENTRY` state). Immediately shout: *"STOP"*.
12. **Preemption Verification:** Observe the system immediately cancelling the PIN buffer, invalidating any pending unlock, forcing state `LOCKED`, and dispatching an emergency audit alert.
13. **Accessibility Mode:** Toggle "High-Contrast Mode" and "Large Text Mode" on the dashboard. Observe immediate UI restyling with zero impact on audio latency.
14. **Audit Log Inspection:** Scroll to the audit log table on the dashboard. Point out the timestamped entries for logins, failed attempts, and the emergency stop event.
15. **Report Download:** Click "Download Audit Report (JSON / CSV)" to demonstrate export capabilities.
16. **Clean Termination:** Press `Ctrl + C` in the terminal to demonstrate graceful shutdown of all threads, database connections, and audio hardware.

---

## 9. Soft Real-Time Limitations & Disclaimers

> [!WARNING]
> - **Soft Real-Time Only:** Windows is a general-purpose operating system without a hard real-time kernel. Latencies are subject to OS thread scheduling jitter, deferred procedure calls (DPCs), and background service preemption.
> - **CPython GIL:** Python execution utilizes a Global Interpreter Lock. Real-time deadlines are achieved via application-level prioritisation, non-blocking I/O callbacks, and thread delegation.
> - **Software Lock Simulation:** The smart lock is a software model and state machine; it is not connected to a certified physical solenoid or lock bolt.
> - **Acoustic Environment:** Keyword spotter accuracy depends on ambient room noise, microphone frequency response, and speaker articulation.
> - **Untested Threats:** The software does not include biometric voiceprint matching (anti-spoofing/replay protection); any user speaking the authorized PIN digits will be recognized.

---

## 10. Deliverables & Documentation Directory

- **Validation Report:** [`stage10_final_validation_report.md`](file:///C:/Users/golu6/.gemini/antigravity/brain/95286633-941e-4937-a2f0-8fef4b1fdcc7/stage10_final_validation_report.md)
- **Validation Charts:** [`results/charts/`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/results/charts) (10 publication-quality PNG charts)
- **Benchmark Data:** [`results/benchmarks/stage10_validation_data.json`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/results/benchmarks/stage10_validation_data.json)
- **Evaluation Heatmap:** [`results/evaluation_21class_confusion_matrix.png`](file:///c:/Users/golu6/Desktop/RTOS_SHUBHAM/results/evaluation_21class_confusion_matrix.png)
