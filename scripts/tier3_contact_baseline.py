"""Measure identical contact gaps on the three external screen clean G1 references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier3_quality import contact_gaps


def external(path):
    path = Path(path).resolve(strict=True)
    if 'challenge' in path.parts or any(p.startswith(('tier_1_', 'tier_2_')) for p in path.parts):
        raise ValueError('baseline must not read challenge data')
    return path


def baseline(manifest):
    import torch
    from robotic_grounding.retarget.read_soma import SOMA
    torch.set_num_threads(2)
    soma = SOMA(device=torch.device('cpu'))
    records = []
    for sequence in json.loads(external(manifest).read_text())['sequences']:
        clean = sequence['arms']['clean']
        path = external(clean['training_reference']['path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != clean['training_reference']['sha256']:
            raise ValueError('external clean reference hash changed')
        source = external(Path(clean['work_dir']) / 'sequences' / sequence['sequence'])
        row = pq.ParquetFile(path).read().to_pylist()[0]
        mesh = trimesh.load(external(source / 'reconstructed_mesh/output_aligned.glb'), force='mesh', process=False)
        poses = np.load(external(source / 'poses.npy'), allow_pickle=False)
        joints = soma.load_motion(external(source / 'soma_params.npz'))['joints']
        records.append({'sequence': sequence['sequence'], 'reference': str(path), 'sha256': digest,
                        'frames': len(poses), 'contacts': contact_gaps(mesh.vertices, poses, joints, row)})
    return {'evidence': 'measured on external screen clean inputs only; no challenge trajectory calibration',
            'limitation': 'Geometric labels, vertex gaps, human fingertips versus G1 palms; no G1 fingertip frames in these parquets.',
            'sequences': records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = baseline(a.manifest)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
