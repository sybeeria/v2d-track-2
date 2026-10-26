"""Validate frame coverage and name a video for upstream's wild-export contract."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

source, output = map(lambda p: Path(p).resolve(), sys.argv[1:3])
expected=int(sys.argv[3])
info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(source)]))['streams'][0]
if int(info['nb_frames']) != expected:
    raise ValueError('manifest frame count differs from video')
output.parent.mkdir(parents=True,exist_ok=True)
# A copy, not a symlink: upstream pack_masks resolves the path and requires the .0.color.mp4 name (Oct 26 batch).
shutil.copyfile(source,output)
