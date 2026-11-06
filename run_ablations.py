"""Later platform stage: S/42 -> E/42; requires all U/A seeds completed."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
python=ROOT/'.venv/bin/python'
if Path(sys.prefix).resolve()!=(ROOT/'.venv').resolve():
    if not python.exists():raise SystemExit(f'Project environment missing: {python}')
    os.execv(str(python),[str(python),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])

from dagrpo.platform import main
from dagrpo.startup_retry import run_ablation_stage
if __name__=='__main__':raise SystemExit(run_ablation_stage(main,ROOT))
