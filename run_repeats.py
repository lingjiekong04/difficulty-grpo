"""Persistent platform entry: U/43 -> A/43 -> U/44 -> A/44."""
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
if __name__=='__main__':raise SystemExit(main('repeats'))
