"""Apply every retained upstream patch idempotently; a drifted target fails setup."""
from pathlib import Path
import os
import subprocess

base = Path(os.environ['NATIVE_MODULES'])
for patch in sorted((Path(__file__).resolve().parent / 'patches').glob('*.diff')):
    forward = ['patch', '-p1', '--batch', '--forward', '--fuzz=0', '-i', str(patch)]
    reverse = ['patch', '-p1', '--batch', '--reverse', '--force', '--fuzz=0', '--dry-run', '-i', str(patch)]
    if subprocess.run(reverse, cwd=base, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(forward + ['--dry-run'], cwd=base, check=True)
        subprocess.run(forward, cwd=base, check=True)
    print(patch.name + ' present')
