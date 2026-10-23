"""Carry alignment-staged scene assets into an independently owned IK work tree."""
import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--sequences', type=Path, required=True)
    p.add_argument('--alignment-motion', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('choose a fresh owned IK work tree')
    # write_sequence stages support USDA here; soma_to_g1 does not create it.
    shutil.copytree(a.alignment_motion, a.out / 'motion')
    subprocess.run(['bash', str(a.project / 'scripts/retarget_tier.sh'), '--sequences',
                    str(a.sequences), str(a.out), '1'], check=True)


if __name__ == '__main__':
    main()
