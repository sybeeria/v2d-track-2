"""Run downloaded upstream scripts in isolated Python with explicit source paths."""
import os
from pathlib import Path
import runpy
import sys

script = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(script.parent), *os.environ['PYTHONPATH'].split(os.pathsep)]
sys.argv = sys.argv[1:]
runpy.run_path(str(script), run_name='__main__')
