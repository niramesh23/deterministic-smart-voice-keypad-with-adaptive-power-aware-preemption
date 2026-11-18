import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig
from src.rts.scheduler import SchedulingPolicy

print("Initializing VoiceRecognitionApp...")
config = OrchestratorConfig(
    policy=SchedulingPolicy.EDF,
    enable_dashboard=False,  # No need for dashboard webserver during diagnosis
    enable_adaptation=True,
)
orch = VoiceRecognitionApp(config=config)

def on_cmd(rec):
    print(f"\n>>> [CMD CALLBACK] Confirmed: {rec.command} (conf={rec.confidence:.2f}, digit={rec.digit}) <<<\n")

orch.user_on_command = on_cmd
orch.start(use_microphone=True)

print("\nSystem running. Listening to live microphone for 6 seconds...")
for i in range(12):
    time.sleep(0.5)
    vad_state = orch.recognizer.vad.state.name
    fsm_state = orch.fsm.state.value
    q_occ = orch.recognizer.audio_queue.qsize()
    inf_count = orch.recognizer.total_inferences_run
    vad_trig = orch.recognizer.total_vad_triggers
    floor = orch.recognizer.vad.noise_floor.noise_floor
    thresh = orch.recognizer.vad.noise_floor.threshold
    print(f"t={0.5*(i+1):.1f}s | FSM={fsm_state:<15} | VAD={vad_state:<14} | Q={q_occ} | inf={inf_count} | vad_trig={vad_trig} | floor={floor:.5f} | thresh={thresh:.5f}")

orch.stop()
print("Diagnosis complete.")
