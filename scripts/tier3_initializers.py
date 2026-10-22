"""Convert trusted SAM3D + FoundationPose initializers to tier3_export.py's input.

Run stages 01-05 first. This is a reconstruction bundle, never a policy rollout.
Only the pinned wild-export identity camera and aligned-mesh contract is supported.
"""
import argparse
import hashlib
import json
import pickle
import runpy
from pathlib import Path

import numpy as np


def rigid(a, frames, *, repair_roundoff=False):
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (frames, 4, 4) or not np.isfinite(a).all():
        raise ValueError('Expected finite [frames,4,4] poses')
    if not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Invalid homogeneous row')
    r = a[:, :3, :3]
    error = np.max(np.abs(r @ r.transpose(0, 2, 1) - np.eye(3)))
    if repair_roundoff:
        if error > 1e-4 or np.any(np.linalg.det(r) <= 0):
            raise ValueError('Rotation drift exceeds float32 tracking roundoff bound')
        u, _, vt = np.linalg.svd(r)
        a = a.copy()
        a[:, :3, :3] = u @ vt
    elif error > 1e-5 or not np.allclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError('Invalid rotation')
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--human', type=Path, required=True)
    p.add_argument('--object', type=Path, required=True)
    p.add_argument('--wild-export', type=Path, required=True)
    p.add_argument('--expected-frames', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--released-mesh-engineering-only', type=Path, help='Unresolved eligibility: source released mesh, transformed with the retained wild-export matrix')
    p.add_argument('--mesh-out', type=Path)
    args = p.parse_args()
    schema = args.project / 'external/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/schema.py'
    dims = runpy.run_path(str(schema))['MHR_PARAM_DIMS']
    # Only locally produced, trusted pickle files from the pinned upstream stages.
    with args.human.open('rb') as f:
        human = pickle.load(f)
    with args.object.open('rb') as f:
        obj = pickle.load(f)
    marker = json.loads(args.wild_export.read_text())
    expected = [f'{i:06d}' for i in range(args.expected_frames)]
    if args.expected_frames <= 0 or human['frames'] != expected or obj['frames'] != expected or marker['frame_count'] != args.expected_frames:
        raise ValueError('Frame timelines differ or are incomplete')
    meta = obj['metadata']
    if meta.get('first_usable_frame_gt_rotation_oracle') is not False or meta.get('foundationpose_selection_uses_gt_pose') is not False or meta.get('gt_object_pose_available') is not False:
        raise ValueError('Object source used or could read ground truth')
    if human.get('metadata', {}).get('source') != 'sam3d_body_rgb' or human.get('metadata', {}).get('alignment') != 'none':
        raise ValueError('Expected unaligned RGB-only SAM3D human initializer')
    if meta.get('gt_translation_oracle') is not False:
        raise ValueError('Object translation oracle must be explicitly disabled')
    if human.get('body_model') != 'mhr' or obj['camera_id'] != 0 or marker['camera_id'] != 0:
        raise ValueError('Expected native MHR and camera zero')
    if marker.get('object_pose_frame') != 'centered_axis_aligned' or not np.allclose(marker['object_mesh_to_training_transform'], np.eye(4), atol=1e-6) or not np.allclose(marker['object_pose_storage_to_training_transform'], np.eye(4), atol=1e-6):
        raise ValueError('Only current aligned-mesh wild exports are supported')
    original = np.asarray(obj['obj_pose_world'], dtype=np.float64)
    poses = rigid(original, args.expected_frames, repair_roundoff=True)
    if not np.allclose(original, np.asarray(obj['fp_poses_cam'])[:, 0], atol=1e-5):
        raise ValueError('Expected wild export identity camera/world frame')
    pr = {'pose_abs': poses}
    for key, width in dims.items():
        values = np.asarray(human[key], dtype=np.float32)
        if values.shape != (args.expected_frames, width) or not np.isfinite(values).all():
            raise ValueError(f'Invalid {key}')
        pr[key] = values
    import torch
    bundle = {'schema': 't3c.initializers.v1', 'frames': expected, 'pr': pr,
              'metadata': {'ground_truth_used': False,
                           'object_mesh_to_training_transform': np.eye(4).tolist(),
                           'rotation_so3_projection_max_element_change': float(np.max(np.abs(poses-original))),
                           'source': 'SAM3D human initializer + FoundationPose; no CoCoNet/refinement',
                           'sources': {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in (args.human, args.object, args.wild_export)}}}
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.released_mesh_engineering_only is not None:
        if args.mesh_out is None or args.mesh_out.exists():
            raise ValueError('Supply a new --mesh-out')
        import trimesh
        mesh = trimesh.load(args.released_mesh_engineering_only, force='scene', process=False)
        mesh.apply_transform(np.asarray(marker['source_object_mesh_to_aligned_transform']))
        args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(args.mesh_out)
        bundle['metadata']['mesh_provenance'] = 'released-mesh-engineering-only; organizer ruling pending'
    elif args.mesh_out is not None:
        raise ValueError('--mesh-out requires --released-mesh-engineering-only')
    torch.save(bundle, args.out)
    print(json.dumps({'frames': len(expected), 'output': str(args.out), 'bytes': args.out.stat().st_size}))


if __name__ == '__main__':
    main()
