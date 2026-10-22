"""Fixed label-free smoothing and full-mesh grounding, after alignment and before IK."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from denoise import smooth, smooth_rotations
from tier3_align import CV_TO_SOMA, load_soma
from tier3_quality import mesh_plane_bounds, summary

WINDOW = 15
CLEARANCE_M = 0.0001


def smooth_pinned(values):
    result = smooth(values, WINDOW)
    return result + values[0] - result[0]


def rotations_pinned(matrices):
    result = smooth_rotations(matrices, WINDOW)
    return matrices[0] @ result[0].T @ result


def repair_arrays(data, poses, vertices, plane):
    result = {k: v.copy() for k, v in data.items()}
    result['transl'] = smooth_pinned(data['transl'])
    result['bone_length_flexibles'] = smooth_pinned(data['bone_length_flexibles'])
    result['poses'] = np.stack([
        Rotation.from_matrix(rotations_pinned(Rotation.from_rotvec(data['poses'][:, j]).as_matrix())).as_rotvec()
        for j in range(data['poses'].shape[1])], axis=1)
    repaired = poses.copy()
    repaired[:, :3, 3] = smooth_pinned(poses[:, :3, 3])
    repaired[:, :3, :3] = rotations_pinned(poses[:, :3, :3])
    # Pin exactly, avoiding SO(3) conversion roundoff at the calibration frame.
    result['poses'][0] = data['poses'][0]
    repaired[0] = poses[0]
    bottom = mesh_plane_bounds(vertices, repaired, plane)[:, 0]
    lift = np.maximum(0., CLEARANCE_M - bottom)
    normal = np.asarray(plane[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    shift = lift[:, None] * normal
    repaired[:, :3, 3] += shift
    result['transl'] += shift @ CV_TO_SOMA
    return result, repaired, lift


def repair(source, out):
    if out.exists():
        raise FileExistsError(f'choose a new repair destination: {out}')
    data = load_soma(source / 'soma_params.npz')
    poses = np.load(source / 'poses.npy', allow_pickle=False)
    if poses.shape != (len(data['poses']), 4, 4) or not np.isfinite(poses).all():
        raise ValueError('object/human frame mismatch or non-finite pose')
    if not np.allclose(poses[:, 3], [0, 0, 0, 1]) or not np.allclose(
            poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3], np.eye(3), atol=1e-5
    ) or not np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-5):
        raise ValueError('object transforms must be SE(3)')
    plane = json.loads((source / 'ground_plane.json').read_text())['plane']
    if len(plane) != 4 or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) == 0:
        raise ValueError('invalid ground plane')
    mesh = trimesh.load(source / 'reconstructed_mesh/output_aligned.glb', force='mesh', process=False)
    result, repaired, lift = repair_arrays(data, poses, np.asarray(mesh.vertices), plane)
    shutil.copytree(source, out)
    np.savez(out / 'soma_params.npz', **result)
    np.save(out / 'poses.npy', repaired)
    report = {
        'evidence': 'measured transformation of inferred inputs; no trajectory labels read',
        'source': str(source.resolve()), 'frames': len(poses),
        'parameters': {'window_frames': WINDOW, 'polynomial_order': 3, 'clearance_m': CLEARANCE_M},
        'ground_shift_applied_to': ['object_translation_OpenCV', 'human_translation_SOMA'],
        'lift_m': summary(lift), 'lift_per_frame_m': lift.tolist(),
        'grounded_bottom_m': summary(mesh_plane_bounds(mesh.vertices, repaired, plane)[:, 0]),
        'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ('soma_params.npz', 'poses.npy', 'ground_plane.json', 'reconstructed_mesh/output_aligned.glb')},
        'frame0_ground_shift_cv_m': (repaired[0, :3, 3] - poses[0, :3, 3]).tolist(),
        'limitation': 'Ground projection changes calibrated frame-0 height; equal source shifts preserve human/object displacement, while IK may change contact gaps.'}
    (out / 'tier3_repair.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = repair(a.source, a.out)
    print(json.dumps({k: report[k] for k in ('frames', 'parameters', 'lift_m', 'grounded_bottom_m')}, allow_nan=False))


if __name__ == '__main__':
    main()
