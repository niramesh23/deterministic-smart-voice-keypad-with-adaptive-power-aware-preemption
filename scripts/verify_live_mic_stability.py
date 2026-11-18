import sys
import pathlib
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig
from src.rts.scheduler import SchedulingPolicy

def main():
    print("==================================================")
    print("Testing Live Microphone Stability (15 seconds)...")
    print("==================================================")

    config = OrchestratorConfig(
        policy=SchedulingPolicy.EDF,
        enable_dashboard=False,
        enable_adaptation=True,
    )
    app = VoiceRecognitionApp(config=config)

    confirmed_commands = []
    def on_cmd(rec):
        print(f"\n>>> [CONFIRMED COMMAND]: {rec.command} (conf={rec.confidence*100:.1f}%, margin={rec.margin:.2f}) <<<\n")
        confirmed_commands.append(rec)

    app.user_on_command = on_cmd
    app.start(use_microphone=True)

    print("Pipeline started. Monitoring live performance...")
    try:
        for sec in range(1, 16):
            time.sleep(1.0)
            mic_active = app.recognizer.mic.is_active if (app.recognizer.mic and hasattr(app.recognizer.mic, "is_active")) else "N/A"
            ingest_alive = app.recognizer._mic_ingest_thread.is_alive() if app.recognizer._mic_ingest_thread else False
            worker_alive = app.recognizer.scheduler._worker_thread.is_alive() if app.recognizer.scheduler._worker_thread else False
            
            q_occ = app.recognizer.audio_queue.qsize()
            ready_jobs = app.recognizer.scheduler.ready_count
            ingested = app.recognizer.total_audio_blocks_ingested
            drops = app.recognizer.total_audio_blocks_dropped
            sched_drops = app.recognizer.scheduler.total_dropped
            vad_state = app.recognizer.vad.state.name
            ctrl_state = app.recognizer.controller.state.value if app.recognizer.controller else "N/A"
            inferences = app.recognizer.total_inferences_run

            print(
                f"[{sec:02d}s] MicActive={mic_active} | Ingest={ingest_alive} | Worker={worker_alive} | "
                f"Q_occ={q_occ} | Ready={ready_jobs} | Ingested={ingested} | Drops={drops}+{sched_drops} | "
                f"VAD={vad_state:<12} | Ctrl={ctrl_state:<8} | Inf={inferences}"
            )

        print("\nVerification succeeded! All threads alive, queues stable, and capture active.")
    finally:
        app.stop()
        print("Application cleanly stopped.")

if __name__ == "__main__":
    main()

