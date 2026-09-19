"""File-based launcher compatible with Windows PowerShell 5.1."""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
if sys.version_info[:2] == (3, 12):
    sys.path.insert(0, str(root / '.venv' / 'Lib' / 'site-packages'))
sys.path.insert(0, str(root / 'wall_climb'))
os.chdir(root / 'wall_climb')
runpy.run_path(str(root / 'wall_climb' / 'main.py'), run_name='__main__')
