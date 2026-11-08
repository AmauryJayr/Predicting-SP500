"""Paths for the TimesFM-3 study. Its own files stay in this folder; shared project code
(common.py, methods.py, run_test.py) is imported from ../experiments, as chronos_experiment/ does."""
import sys
from pathlib import Path

TF_DIR = Path(__file__).resolve().parent
sys.path.insert(1, str(TF_DIR.parent / "experiments"))
RES_DIR = TF_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
