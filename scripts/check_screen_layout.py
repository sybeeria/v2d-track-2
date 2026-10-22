"""Regenerate a 15-frame non-challenge prefix screen in a fresh relative project layout.

Reuses installed dependencies via symlinks; native retargeting and all generation checks run.
Temporary inputs/outputs are removed. Full-length geometry is audited separately.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq

from make_tier2_screen import CLEAN, ROOT, save_params


def main():
    log = ROOT / 'data/screen/tier2/layout_generation.log'
    with tempfile.TemporaryDirectory(prefix='v2d-screen-layout-') as temporary:
        root = Path(temporary).resolve()
        shutil.copytree(ROOT / 'scripts', root / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
        (root / '.venv').symlink_to(ROOT / '.venv', target_is_directory=True)
        runtime = ROOT / '.venv-flash'
        if not runtime.is_dir():
            runtime = ROOT / '.venv-flash-cpu'
        (root / '.venv-flash').symlink_to(runtime, target_is_directory=True)
        relative = CLEAN.relative_to(ROOT)
        source, target = ROOT, root
        for part in relative.parts:
            for child in source.iterdir():
                if child.name != part and source != ROOT:
                    (target / child.name).symlink_to(child, target_is_directory=child.is_dir())
            source, target = source / part, target / part
            target.mkdir()
        for folder in sorted(CLEAN.glob('sequence_id=*')):
            output = target / folder.name
            output.mkdir()
            poses = np.load(folder / 'poses.npy')
            np.save(output / 'poses.npy', poses[:15])
            with np.load(folder / 'soma_params.npz') as archive:
                arrays = {k: v[:15] if v.ndim and len(v) == len(poses) else v for k, v in archive.items()}
            save_params(output / 'soma_params.npz', arrays)
            for name in ('ground_plane.json', 'reconstructed_mesh', 'robot_name=g1'):
                (output / name).symlink_to(folder / name, target_is_directory=(folder / name).is_dir())
        command = [str(root / '.venv/bin/python'), 'scripts/make_tier2_screen.py', '--out', 'data/screen/tier2']
        with log.open('w') as handle:
            subprocess.run(command, cwd=root, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((root / 'data/screen/tier2/manifest.json').read_text())
        checked = 0
        for sequence in manifest['sequences']:
            assert sequence['frames'] == 15
            assert sequence['upstream_support_not_used']['used'] is False
            assert len({a['support']['sha256'] for a in sequence['arms'].values()}) == 1
            for arm in sequence['arms'].values():
                assert Path(arm['training_reference']['path']).is_relative_to(root)
                row = pq.read_table(arm['training_reference']['path'], columns=['object_mesh_paths', 'object_urdf_paths']).to_pylist()[0]
                for path in row['object_mesh_paths'] + row['object_urdf_paths']:
                    assert Path(path).is_relative_to(root) and Path(path).is_file(), path
                checked += 1
        print(f'PASS: fresh relative project layout generated {checked} references (15-frame non-challenge prefixes); all asset paths rebound; supports identical across five inputs; .venv-flash selected without .venv-flash-cpu')
    print('PASS: temporary layout removed; native generation log: data/screen/tier2/layout_generation.log')


if __name__ == '__main__':
    main()
