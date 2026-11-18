"""
scripts/verify_physical_mic.py
------------------------------
Standalone Physical Microphone Live Validation for Stage 8 Audit.
Tests real hardware capture through ScheduledKeywordRecognizer with Adaptive Controller enabled.
"""

import logging
import sys
import pathlib
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.scheduled_recognizer import ScheduledKeywordRecognizer
from src.audio.microphone import get_default_input_device
from src.rts.scheduler import SchedulingPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verify_physical_mic")

def main():
    logger.info("=== Physical Microphone Live Hardware Validation ===")
    dev = get_default_input_device()
    logger.info(f"Target Input Device: '{dev.name}' (Device ID: {dev.device_id}, Rate: {dev.default_sample_rate} Hz)")

    recognizer = ScheduledKeywordRecognizer(
        policy=SchedulingPolicy.EDF,
        device_id=dev.device_id,
        enable_adaptation=True,
    )

    confirmed_cmds = []
    def on_cmd(res, tel):
        if res.is_confirmed:
            confirmed_cmds.append((res.command, res.confidence, tel.reaction_latency_ms))
            logger.info(f"LIVE MIC RECOGNIZED: '{res.command}' (Conf: {res.confidence*100:.1f}%, Reaction: {tel.reaction_latency_ms:.2f} ms)")

    recognizer.on_command_callback = on_cmd

    t0 = time.perf_counter()
    recognizer.start(use_microphone=True)

    # Capture 3.5 seconds of physical microphone input
    test_duration = 3.5
    logger.info(f"Streaming live audio from physical microphone for {test_duration} seconds...")
    time.sleep(test_duration)

    elapsed_actual = time.perf_counter() - t0
    recognizer.stop()

    report = recognizer.get_analysis_report()
    decisions = recognizer.controller.decision_history

    print("\n" + "="*60)
    print("PHYSICAL MICROPHONE VALIDATION AUDIT RESULTS")
    print("="*60)
    print(f"Device Name:                 {dev.name}")
    print(f"Device ID:                   {dev.device_id}")
    print(f"Physical Duration:           {elapsed_actual:.3f} s")
    print(f"Total Audio Blocks Ingested: {recognizer.total_audio_blocks_ingested}")
    print(f"Audio Blocks Dropped:        {recognizer.total_audio_blocks_dropped}")
    print(f"Scheduler Ready Jobs Exec:   {report.get('total_jobs_executed', 0)}")
    print(f"Total Deadline Misses:       {report.get('total_deadline_misses', 0)}")
    print(f"Deadline Miss Ratio (DMR):   {report.get('deadline_miss_ratio_pct', 0.0):.2f}%")
    print(f"Controller State Transitions:{recognizer.controller.total_state_transitions}")
    print(f"Final Controller State:      {recognizer.controller.state.value}")
    print(f"Inferences Run:              {recognizer.total_inferences_run}")
    print(f"Confirmed Commands Count:    {len(confirmed_cmds)}")
    if confirmed_cmds:
        print(f"Confirmed Commands:          {confirmed_cmds}")
    else:
        print("Confirmed Commands:          None (ambient room silence / background noise safely rejected)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
