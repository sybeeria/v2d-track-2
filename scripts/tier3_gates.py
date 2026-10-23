"""Frozen T3-RECON label-free gates, using native joints saved by the retargeter.

Amended by data/brainstorm6/DECISIONS.md D1: isolated dynamics flags (no run of 3 frames at either playback rate) are
recorded under 'inspected' and do not hold an episode back; sustained runs and the RGB gate still do.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier3_quality import continuity, summary
from tier3_align import load_soma


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def runs(mask, length):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if b-a >= length]


def evaluate(base, sequence, expected, camera, image_evidence=None):
    source = base / 'sequences' / sequence
    parquet = base / 'assets/human_motion_data/whole_body/soma' / f'sequence_id={sequence}/robot_name=g1/data.parquet'
    from flash_chord.data.motion_v1 import load_motion_v1
    reference = load_motion_v1(str(parquet), control_fps=50., source_frame_playback=True)
    if reference.num_frames != expected:
        raise ValueError('G1 loader frame coverage mismatch')
    rows = pq.ParquetFile(parquet).read().to_pylist()
    if len(rows) != 1:
        raise ValueError('expected one G1 motion row')
    row = rows[0]; payload = pickle.loads(row['source_payload'])  # Trusted generated references only.
    data = load_soma(source / 'soma_params.npz')
    aligned = np.load(source / 'poses.npy', allow_pickle=False)
    if np.asarray(row['object_body_position']).shape != (expected, 1, 3) or np.asarray(row['object_body_wxyz']).shape != (expected, 1, 4):
        raise ValueError('expected one object on full frame grid')
    pos = np.asarray(row['object_body_position'])[:, 0]
    quat = np.asarray(row['object_body_wxyz'])[:, 0]
    robot = np.asarray(row['robot_joint_positions'])
    if any(len(x) != expected for x in (data['poses'], aligned, pos, quat, robot)) or expected < 15:
        raise ValueError('video/SOMA/object/G1 coverage mismatch')
    for key in ('robot_joint_positions', 'robot_root_position', 'object_body_position', 'object_body_wxyz', 'ee_pose_w'):
        if not np.isfinite(np.asarray(row[key], float)).all():
            raise ValueError('non-finite G1 ' + key)
    if not np.isfinite(aligned).all() or not np.allclose(aligned[:, 3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(aligned[:, :3, :3].transpose(0, 2, 1) @ aligned[:, :3, :3], np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(aligned[:, :3, :3]), 1, atol=1e-5):
        raise ValueError('non-rigid source poses')
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-5):
        raise ValueError('non-unit G1 object quaternion')
    poses = np.tile(np.eye(4), (expected, 1, 1)); poses[:, :3, 3] = pos
    poses[:, :3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    alignment = json.loads((source / 'tier3_alignment.json').read_text())
    if alignment['reference_frames_used'] != [0]:
        raise ValueError('alignment provenance is not frame-0 only')
    vertices = np.asarray(trimesh.load(parquet.parent.parent / 'object/textured_mesh.obj', force='mesh', process=False).vertices)
    if len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError('invalid runtime mesh')
    # This is the same SOMA-native -> G1-world rotation as the existing retargeter.
    world = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    joints = np.asarray(payload['soma_joints']) @ world.T
    names = list(data['joint_names'])
    if joints.shape != (expected, len(names), 3) or not np.isfinite(joints).all():
        raise ValueError('invalid saved native joints')
    tip_ids = [i for i, n in enumerate(names) if str(n).startswith(('LeftHand', 'RightHand')) and str(n).endswith('End')]
    if len(tip_ids) != 10:
        raise ValueError('expected ten native fingertips')
    tree = cKDTree(vertices); bottom, gap = [], []
    for pose, tips in zip(poses, joints[:, tip_ids]):
        bottom.append(float((vertices @ pose[2, :3] + pose[2, 3]).min()))
        gap.append(float(tree.query((tips - pose[:3, 3]) @ pose[:3, :3])[0].min()))
    bottom, gap = np.asarray(bottom), np.asarray(gap)
    smooth_speed = np.r_[0, np.linalg.norm(np.diff(savgol_filter(pos, 15, 3, axis=0), axis=0), axis=-1)*30]
    raw_speed = np.r_[0, np.linalg.norm(np.diff(pos, axis=0), axis=-1)*30]
    physical = {'penetration_runs': runs(bottom < -.02, 6), 'unsupported_runs': runs((bottom > .03) & (gap > .03), 6),
                'detached_runs': runs((smooth_speed > .03) & (gap > .08), 6)}
    dynamics = {}; unknown = []; reasons = []; inspected = []
    for fps in (30, 50):
        for name, values, rotation, speed, accel in [('translation', pos, False, 2., 20.), ('rotation', poses[:, :3, :3], True, 10., 100.)]:
            item = continuity(values, fps, rotation=rotation, speed_limit=speed, acceleration_limit=accel)
            for key in ('speed', 'acceleration'):
                frames = item[key + '_outlier_destination_frames']; mask = np.zeros(expected, bool); mask[frames] = True
                item[key + '_sustained_runs'] = runs(mask, 3)
                if item[key + '_sustained_runs']:
                    reasons.append(f'{fps}Hz {name} {key} sustained')
                elif frames:
                    inspected.append(f'{fps}Hz {name} {key}: isolated flags at frames {frames}')
            dynamics[f'{fps}_{name}'] = item
    for key, intervals in physical.items():
        if intervals:
            reasons.append(key + ': ' + str(intervals))
    if np.any(bottom < -.02) and not physical['penetration_runs']:
        unknown.append('isolated penetration below -2cm requires inspection')
    A = np.eye(4); A[:3, :3] = alignment['rotation']; A[:3, 3] = alignment['translation']
    unlift = poses[0].copy(); lift = payload['ground_alignment']['object_ground_correction']['applied_lift']; unlift[2, 3] -= lift
    C = np.linalg.inv(A) @ aligned[0] @ np.linalg.inv(unlift)
    recovered = np.linalg.inv(C) @ np.linalg.inv(A) @ aligned; recovered[:, 2, 3] += lift
    chain = float(np.abs(recovered - poses).max())
    if chain >= 3e-6:
        raise ValueError('source/G1 frame chain disagrees')
    # Empty support stages are qualified; elevated support/contact assignment needs
    # per-episode review rather than pretending the ground is the only surface.
    support = base / 'assets/human_motion_data/whole_body/reconstructed_stage' / (sequence + '_support.usda')
    if not support.is_file():
        raise ValueError('missing portable support stage')
    from pxr import Usd
    stage = Usd.Stage.Open(str(support))
    if stage is None:
        raise ValueError('invalid support stage')
    if any(p.GetTypeName() in ('Cube', 'Cylinder') for p in stage.Traverse()):
        unknown.append('elevated support requires video-supported surface/contact assignment')
    rgb = []; K = np.asarray(json.loads(camera.read_text())['intrinsics'])
    extrema = {0, expected//4, expected//2, 3*expected//4, expected-1, int(bottom.argmin()), int(bottom.argmax())}
    for values, offset in [(np.linalg.norm(np.diff(pos, axis=0), axis=-1), 1), (np.linalg.norm(np.diff(pos, n=2, axis=0), axis=-1), 2),
                           (Rotation.from_matrix(poses[1:, :3, :3] @ poses[:-1, :3, :3].transpose(0, 2, 1)).magnitude(), 1)]:
        extrema.add(int(values.argmax()) + offset)
    angular = (Rotation.from_matrix(poses[1:, :3, :3]) * Rotation.from_matrix(poses[:-1, :3, :3]).inv()).as_rotvec()
    extrema.add(int(np.linalg.norm(np.diff(angular, axis=0), axis=-1).argmax()) + 2)
    if image_evidence is None:
        unknown.append('independent RGB evidence missing')
    else:
        evidence = json.loads(image_evidence.read_text())
        if evidence.get('video_only') is not True or evidence.get('independent_of_tracker') is not True:
            raise ValueError('RGB evidence must explicitly be video-only and independent of tracker masks')
        polygons = evidence['polygons']; width, height = evidence['image_width'], evidence['image_height']
        hull = vertices[ConvexHull(vertices).vertices]
        for key, polygon in polygons.items():
            frame = int(key)
            if not 0 <= frame < expected:
                raise ValueError('RGB frame outside video')
            xyz = hull @ (C @ poses[frame])[:3, :3].T + (C @ poses[frame])[:3, 3]
            if np.any(xyz[:, 2] <= .01):
                raise ValueError('mesh crosses camera plane')
            points = xyz @ K.T; points = points[:, :2] / points[:, 2, None]
            contour = points[ConvexHull(points).vertices]
            masks = []
            for poly in (contour, polygon):
                image = Image.new('1', (width, height)); ImageDraw.Draw(image).polygon([tuple(x) for x in poly], fill=1)
                masks.append(np.asarray(image))
            pred, target = masks
            yy, xx = np.where(target); py, px = np.where(pred)
            if not len(xx) or not len(px):
                raise ValueError('empty independent/projected mask')
            iou = float((pred & target).sum() / (pred | target).sum())
            error = float(np.hypot(px.mean()-xx.mean(), py.mean()-yy.mean()) / np.hypot(xx.max()-xx.min(), yy.max()-yy.min()))
            rgb.append({'frame': frame, 'iou': iou, 'centroid_error_normalized': error, 'occluded': frame in evidence.get('partly_occluded_frames', [])})
        visible = [r for r in rgb if not r['occluded']]
        if not visible:
            unknown.append('no independent unoccluded RGB checks')
        elif sum(r['iou'] < .5 or r['centroid_error_normalized'] > .1 for r in visible) > .1 * len(visible):
            reasons.append('independent RGB gate failed')
        missing = sorted(extrema - {r['frame'] for r in rgb})
        if missing:
            unknown.append('independent RGB at even/extreme frames missing: ' + str(missing))
        if any(r['occluded'] for r in rgb):
            unknown.append('occluded RGB/contact intervals require review')
    if 'released' in alignment.get('scene_assets_mode', ''):
        unknown.append('released scene assets: organizer ruling pending')
    return {'status': 'HOLD' if reasons or unknown else 'PASS', 'reasons': reasons + unknown, 'inspected': inspected, 'physical': physical,
            'bottom_m': summary(bottom), 'penetration_frames': np.flatnonzero(bottom < -.02).tolist(),
            'moving_without_human_nearby_frames': np.flatnonzero((smooth_speed > .03) & (gap > .08)).tolist(),
            'raw_speed_30Hz': summary(raw_speed), 'smoothed_speed_30Hz': summary(smooth_speed), 'dynamics': dynamics,
            'rgb': rgb, 'coverage': expected, 'G1_loader_frames': reference.num_frames, 'reference_frames_used': [0], 'frame_chain_max_element_error': chain,
            'reference_sha256': sha(parquet), 'source_soma_sha256': sha(source / 'soma_params.npz'),
            'runtime_mesh_vertices': len(vertices), 'source_joint_mode': 'native joints in generated retargeter payload; no model re-decode',
            'support': {'path': str(support.resolve()), 'sha256': sha(support)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--sequence', required=True)
    p.add_argument('--frames', type=int, required=True); p.add_argument('--camera', type=Path, required=True)
    p.add_argument('--rgb', type=Path); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if sha(a.protocol) != 'eec81f1beac187d63cca82ca6b098999be62dff56e7fe1be64c9f874d3af41da':
        raise ValueError('frozen protocol hash differs')
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'tier3_native'))
    from batch import atomic
    try:
        result = evaluate(a.base, a.sequence, a.frames, a.camera, a.rgb)
    except (ValueError, KeyError, OSError) as e:
        result = {'status': 'FAIL', 'reasons': [type(e).__name__ + ': ' + str(e)]}
    result['protocol_sha256'] = sha(a.protocol)
    atomic(a.out, result)
    print(json.dumps({'status': result['status'], 'reasons': result['reasons']}))
    raise SystemExit(1 if result['status'] == 'FAIL' else 0)


if __name__ == '__main__':
    main()
