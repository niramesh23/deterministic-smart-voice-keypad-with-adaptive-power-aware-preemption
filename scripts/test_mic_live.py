import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.application.scheduled_recognizer import ScheduledKeywordRecognizer
from src.dsp.energy import calculate_rms
import numpy as np

rec = ScheduledKeywordRecognizer(enable_adaptation=True)
rec.start(use_microphone=True)

print("Started recognizer with microphone. Monitoring for 4 seconds...")
for i in range(8):
    time.sleep(0.5)
    floor = rec.vad.noise_floor.noise_floor
    thresh = rec.vad.noise_floor.threshold
    v_state = rec.vad.state.name
    print(f"t={0.5*(i+1):.1f}s | VAD={v_state:<14} | floor={floor:.5f} | thresh={thresh:.5f} | triggers={rec.total_vad_triggers} | inferences={rec.total_inferences_run}")

rec.stop()
print("Test complete.")
