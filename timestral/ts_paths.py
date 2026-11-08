"""Paths for the Timestral study. Its own files stay in this folder; shared project code
(common.py, methods.py, run_test.py) is imported from ../experiments, as timesfm3_experiment/ does.
Do not add a file called timestral.py here: it would shadow the installed `timestral` package."""
import sys
from pathlib import Path

TS_DIR = Path(__file__).resolve().parent
sys.path.insert(1, str(TS_DIR.parent / "experiments"))
RES_DIR = TS_DIR / "results"
FIG_DIR = TS_DIR / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TFM_RES = TS_DIR.parent / "timesfm3_experiment" / "results"      # TimesFM-3's saved outputs, read only
