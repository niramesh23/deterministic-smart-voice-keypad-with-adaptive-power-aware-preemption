"""
run_live.py
-----------
Convenience entry point to launch the Real-Time Adaptive Voice Recognition system.
"""

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.orchestrator import main

if __name__ == "__main__":
    main()

