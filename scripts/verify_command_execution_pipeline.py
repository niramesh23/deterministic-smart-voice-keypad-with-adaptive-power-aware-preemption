import sys
import pathlib
import time
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig
from src.rts.scheduler import SchedulingPolicy
from src.ml.dataset import load_wav_16k_mono

def main():
    print("================================================================")
    print("Verifying Post-Job Execution Microphone & Command Pipeline...")
    print("================================================================")

    config = OrchestratorConfig(
        policy=SchedulingPolicy.EDF,
        enable_dashboard=False,
        enable_adaptation=True,
    )
    app = VoiceRecognitionApp(config=config)

    confirmed = []
    def on_cmd(rec):
        print(f"\n[FSM EVENT] Confirmed command: '{rec.command}' (digit={rec.digit}, conf={rec.confidence*100:.1f}%)")
        confirmed.append(rec.command)

    app.user_on_command = on_cmd
    app.start(use_microphone=True)

    try:
        # 1. First let mic run live for 2 seconds
        print("\nStep 1: Listening on live microphone for 2 seconds...")
        time.sleep(2.0)
        assert app.recognizer.mic.is_active, "Microphone stream should be active"
        print(f"Hops ingested so far: {app.recognizer.total_audio_blocks_ingested}")

        # 2. Simulate user speaking "one" and executing lock job
        print("\nStep 2: Feeding spoken command 'one' to trigger recognition and smart lock job...")
        wav_files = list(pathlib.Path("data/raw/one").glob("*.wav"))
        assert len(wav_files) > 0, "No 'one' wav files found"
        audio_f32 = load_wav_16k_mono(str(wav_files[0]))
        audio = np.clip(audio_f32 * 32767.0, -32768, 32767).astype(np.int16)

        # Feed audio in 20ms hops (320 samples)
        hop_size = 320
        for i in range(0, len(audio), hop_size):
            chunk = audio[i:i + hop_size]
            if len(chunk) < hop_size:
                chunk = np.pad(chunk, (0, hop_size - len(chunk)))
            app.push_audio_hop(chunk)
            time.sleep(0.005)

        # Allow time for scheduler to execute the inference and lock controller jobs
        time.sleep(1.0)
        lock_snap = app.lock_controller.get_snapshot()
        print(f"Lock Controller State: {lock_snap.state.value} | PIN Progress: '{lock_snap.pin_progress}'")

        # 3. Verify microphone is STILL healthy and capturing after job execution!
        print("\nStep 3: Verifying live microphone continues capturing uninterrupted after job execution...")
        initial_ingested = app.recognizer.total_audio_blocks_ingested
        for s in range(1, 6):
            time.sleep(1.0)
            now_ingested = app.recognizer.total_audio_blocks_ingested
            delta = now_ingested - initial_ingested
            initial_ingested = now_ingested
            is_active = app.recognizer.mic.is_active
            q_occ = app.recognizer.audio_queue.qsize()
            ready = app.recognizer.scheduler.ready_count
            ctrl_state = app.recognizer.controller.state.value
            print(f"[{s}s post-job] Mic Active={is_active} | Delta Hops={delta} (~50/s expected) | Q={q_occ} | Ready={ready} | State={ctrl_state}")
            assert is_active, "Microphone must remain active!"
            assert delta > 30, f"Expected active audio stream (delta={delta} hops)"

        print("\nSUCCESS: Microphone remained 100% responsive and active after job execution!")

    finally:
        app.stop()
        print("Application stopped.")

if __name__ == "__main__":
    main()
