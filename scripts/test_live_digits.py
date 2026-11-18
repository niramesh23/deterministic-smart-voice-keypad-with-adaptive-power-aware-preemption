import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.application.orchestrator import VoiceRecognitionApp, OrchestratorConfig
from src.rts.scheduler import SchedulingPolicy

print("Testing live digit recognition. Initializing VoiceRecognitionApp...")
config = OrchestratorConfig(
    policy=SchedulingPolicy.EDF,
    enable_dashboard=False,
    enable_adaptation=True,
)
app = VoiceRecognitionApp(config=config)

detected_records = []
def on_cmd(rec):
    detected_records.append(rec)
    print(f"\n*** DETECTED COMMAND: '{rec.command}' (digit={rec.digit}, conf={rec.confidence:.3f}, margin={rec.margin:.3f}) ***\n")

app.user_on_command = on_cmd
app.start(use_microphone=True)

# Also monitor VAD state in real time
print("\n--- Listening to microphone for 5 seconds ---")
for i in range(10):
    time.sleep(0.5)
    fsm_state = app.fsm.state.value
    vad_state = app.recognizer.vad.state.name
    print(f"[{0.5*(i+1):.1f}s] FSM={fsm_state:<15} | VAD={vad_state:<12} | Confirmed={app.fsm.total_confirmed} | Inferences={app.recognizer.total_inferences_run}")

app.stop()
print("\nFinal Results:")
print(f"Total Inferences Run: {app.recognizer.total_inferences_run}")
print(f"Total Confirmed Commands: {app.fsm.total_confirmed}")
for r in detected_records:
    print(f"  - Command: {r.command}, Digit: {r.digit}, Conf: {r.confidence:.3f}")

