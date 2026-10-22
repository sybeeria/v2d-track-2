"""Physical trajectory priors for released SOMA sequence folders, in their original world frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from contacts import _runs
from hoi import CV_TO_SOMA, FEET, FPS, Sequence
from robotic_grounding.retarget.params import SOMA_JOINTS_ORDER
from robotic_grounding.retarget.ground_alignment import compute_plane_leveling_transform, load_ground_plane_robot_frame
from scipy.ndimage import binary_closing
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Params:
    window: int = 15
    rest_gap: float = 0.25
    rest_speed: float = 0.08
    min_rest: int = 15
    root_strength: float = 1.0
    rest_strength: float = 1.0
    support_strength: float = 0.0
    grasp_strength: float = 0.0
    propagate_rest: bool = False
    vertical_only: bool = False
    foot_height: float | None = None
    rotation_frame: Literal['world', 'local'] = 'world'
    support_guard: bool = False
    support_margin: float = 0.005


DEFAULT_PARAMS = Params(root_strength=0.5, rest_strength=0.5, propagate_rest=True, vertical_only=True,
                        rotation_frame='local', support_guard=True)


def input_signature(folder: Path) -> str | None:
    files = ['poses.npy', 'ground_plane.json', 'filled_frames.json', 'reconstructed_mesh/output_aligned.glb']
    if not all((folder / name).is_file() for name in [*files, 'soma_params.npz']):
        return None
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update((folder / name).read_bytes())
    with np.load(folder / 'soma_params.npz') as archive:
        for name in sorted(archive.files):
            array = np.ascontiguousarray(archive[name])
            digest.update(f'{name}:{array.dtype}:{array.shape}'.encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def smooth(x: np.ndarray, window: int = 15) -> np.ndarray:
    window = min(window, len(x) if len(x) % 2 else len(x) - 1)
    return savgol_filter(x, window, min(3, window - 1), axis=0) if window > 1 else x.copy()


def smooth_rotations(r: np.ndarray, window: int) -> np.ndarray:
    q = Rotation.from_matrix(r).as_quat()
    q[np.cumsum(np.r_[False, (q[1:] * q[:-1]).sum(axis=1) < 0]) % 2 == 1] *= -1
    q = smooth(q, window)
    return Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()


def plane(folder: Path) -> tuple[np.ndarray, float]:
    p = np.asarray(json.loads((folder / "ground_plane.json").read_text())["plane"])
    length = np.linalg.norm(p[:3])
    return p[:3] / length, float(p[3] / length)


def rest_mask(seq: Sequence, p: Params, gaps: np.ndarray | None = None,
              foot_gaps: np.ndarray | None = None) -> np.ndarray:
    away = (seq.hand_gaps() if gaps is None else gaps).min(axis=0) > p.rest_gap
    feet_away = (seq.foot_gaps() if foot_gaps is None else foot_gaps) > 0.15
    mask = away & feet_away & ~seq.object_moving(speed=p.rest_speed)
    mask = binary_closing(mask, structure=np.ones(5), border_value=0)
    mask &= away & feet_away
    for a, b in _runs(mask, True):
        if b - a < p.min_rest:
            mask[a:b] = False
    return mask


def root_correction(seq: Sequence, p: Params) -> np.ndarray:
    """Translation correction from the lowest foot and low-speed stance increments."""
    n, d = plane(seq.folder)
    feet = seq.joints[:, FEET] @ CV_TO_SOMA
    height = feet @ n + d
    low = height.min(axis=1)
    target = np.median(low[:15]) if p.foot_height is None else p.foot_height
    correction = (target - smooth(low, 31))[:, None] * n
    if p.vertical_only:
        return correction * p.root_strength
    vel = np.gradient(smooth(feet, 15), axis=0)
    horizontal = vel - (vel @ n)[..., None] * n
    stance = (height < low[:, None] + 0.025) & (np.linalg.norm(horizontal, axis=-1) * FPS < 0.12)
    drift = np.zeros((len(feet), 3))
    for t in range(len(feet)):
        if stance[t].any():
            drift[t] = np.median(horizontal[t, stance[t]], axis=0)
    correction -= np.cumsum(drift, axis=0) - drift[0]
    return smooth(correction, 31) * p.root_strength


def support_disks(folder: Path) -> list[tuple[np.ndarray, float, float]]:
    """Released USD disks in the retargeter's levelled G1 frame."""
    stage = (folder.parent.parent / 'reconstructed_stage' if folder.name.startswith('sequence_id=') else
             folder.parent.parent / 'motion/whole_body/reconstructed_stage')
    path = stage / f"{folder.name.removeprefix('sequence_id=')}_support.usda"
    if not path.exists():
        return []
    disks = []
    for block in path.read_text().split('def Cylinder ')[1:]:
        def scalar(name):
            return float(re.search(rf'double {name} = ([^\n]+)', block)[1])
        xyz = np.array([float(x) for x in re.search(r'xformOp:translate = \(([^)]+)\)', block)[1].split(',')])
        disks.append((xyz, scalar('radius'), scalar('height')))
    return disks


def cv_to_robot(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    config = PROJECT / 'external/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/retarget/configs/g1/frame_alignment.json'
    swap = np.asarray(json.loads(config.read_text())['world_axis_swap_matrix'])
    with np.load(folder / 'soma_params.npz') as a:
        inverse = Rotation.from_rotvec(a['poses'][0, 0]).as_matrix().T
        anchor = np.eye(4)
        anchor[:3, :3] = inverse
        anchor[:3, 3] = -inverse @ a['transl'][0]
    cv = np.eye(4)
    cv[:3, :3] = CV_TO_SOMA
    ground = load_ground_plane_robot_frame(folder / 'ground_plane.json', cv_to_source=cv,
                                           first_frame_anchor=anchor, source_to_robot=swap)
    level_r, level_t = compute_plane_leveling_transform(ground)
    return level_r @ swap @ inverse @ CV_TO_SOMA, level_r @ swap @ anchor[:3, 3] + level_t


def support_clearance(seq: Sequence) -> tuple[np.ndarray, np.ndarray]:
    n, d = plane(seq.folder)
    poses = seq.object_poses_cv
    points = np.einsum('tij,nj->tni', poses[:, :3, :3], seq.mesh_points) + poses[:, None, :3, 3]
    lowest = (points @ n + d).min(axis=1)
    height = np.zeros(len(poses))
    disks = support_disks(seq.folder)
    if disks:
        r, t = cv_to_robot(seq.folder)
        world = points @ r.T + t
        for center, radius, thickness in disks:
            bottom = world[:, :, 2].min(axis=1)
            # Disks are inferred supports, not a licence to snap a lifted object down.
            near = np.linalg.norm(world.mean(axis=1)[:, :2] - center[:2], axis=1) < radius + 0.12
            candidate = center[2] + thickness / 2
            use = near & (np.abs(bottom - candidate) < 0.25) & (candidate > height)
            height[use] = candidate
    return lowest - height, height


def ground_heights(seq: Sequence, poses: np.ndarray) -> np.ndarray:
    n, d = plane(seq.folder)
    direction = np.einsum('tij,i->tj', poses[:, :3, :3], n)
    return (seq.mesh_points @ direction.T).min(axis=0) + poses[:, :3, 3] @ n + d


def grasp_prior(seq: Sequence, poses: np.ndarray, joints: np.ndarray, strength: float,
                gaps: np.ndarray) -> np.ndarray:
    output = poses.copy()
    for side, name in enumerate(('Left', 'Right')):
        ids = [SOMA_JOINTS_ORDER.index(name + suffix) for suffix in
               ('Hand', 'HandMiddle1', 'HandIndex1', 'HandPinky1')]
        palm = joints[:, ids] @ CV_TO_SOMA
        origin = palm[:, 0]
        y = palm[:, 1] - origin
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        x = palm[:, 2] - palm[:, 3]
        z = np.cross(x, y)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        x = np.cross(y, z)
        r = np.stack((x, y, z), axis=-1)
        active = (gaps[side] < 0.15) & (gaps[side] <= gaps[1-side])
        active = binary_closing(active, structure=np.ones(5))
        for a, b in _runs(active, True):
            if b - a < 30:
                continue
            local_t = np.einsum('tji,tj->ti', r[a:b], poses[a:b, :3, 3] - origin[a:b])
            local_r = r[a:b].transpose(0, 2, 1) @ poses[a:b, :3, :3]
            target_t = np.einsum('tij,j->ti', r[a:b], np.median(local_t, axis=0)) + origin[a:b]
            target_r = r[a:b] @ Rotation.from_matrix(local_r).mean().as_matrix()
            weight = np.minimum(np.minimum(np.arange(b-a), np.arange(b-a)[::-1]) / 10, 1) * strength
            output[a:b, :3, 3] += weight[:, None] * (target_t - output[a:b, :3, 3])
            delta = Rotation.from_matrix(target_r @ output[a:b, :3, :3].transpose(0, 2, 1)).as_rotvec()
            output[a:b, :3, :3] = Rotation.from_rotvec(weight[:, None] * delta).as_matrix() @ output[a:b, :3, :3]
    return output


def denoise(seq: Sequence, p: Params = DEFAULT_PARAMS,
            gaps: np.ndarray | None = None, foot_gaps: np.ndarray | None = None) -> tuple[Sequence, np.ndarray, dict]:
    poses = seq.object_poses_cv.copy()
    poses[:, :3, 3] = smooth(poses[:, :3, 3], p.window)
    poses[:, :3, :3] = smooth_rotations(poses[:, :3, :3], p.window)
    smoothed = poses.copy()
    shift = root_correction(seq, p) if p.root_strength else np.zeros((len(poses), 3))
    joints = seq.joints + (shift @ CV_TO_SOMA)[:, None]
    if gaps is None:
        gaps = seq.hand_gaps()
    rest = rest_mask(seq, p, gaps, foot_gaps)
    rest_frames = int(rest.sum())
    if p.propagate_rest:
        bias_t = np.zeros_like(poses[:, :3, 3])
        bias_r = np.tile(np.eye(3), (len(poses), 1, 1))
        previous_t, previous_r, last = np.zeros(3), np.eye(3), 0
        for a, b in _runs(rest, True):
            bias_t[last:a], bias_r[last:a] = previous_t, previous_r
            bias_t[a:b] = previous_t + poses[a:b, :3, 3] - poses[a, :3, 3]
            bias_r[a:b] = (previous_r @ poses[a, :3, :3].T @ poses[a:b, :3, :3]
                if p.rotation_frame == 'local' else poses[a:b, :3, :3] @ poses[a, :3, :3].T @ previous_r)
            previous_t, previous_r, last = bias_t[b-1], bias_r[b-1], b
        bias_t[last:], bias_r[last:] = previous_t, previous_r
        poses[:, :3, 3] -= p.rest_strength * bias_t
        correction = Rotation.from_rotvec(-p.rest_strength * Rotation.from_matrix(bias_r).as_rotvec()).as_matrix()
        poses[:, :3, :3] = (poses[:, :3, :3] @ correction if p.rotation_frame == 'local' else
                            correction @ poses[:, :3, :3])
    else:
        for a, b in _runs(rest, True):
            poses[a:b, :3, 3] += p.rest_strength * (poses[a:b, :3, 3].mean(axis=0) - poses[a:b, :3, 3])
            mean = Rotation.from_matrix(poses[a:b, :3, :3]).mean().as_matrix()
            delta = Rotation.from_matrix(mean @ poses[a:b, :3, :3].transpose(0, 2, 1)).as_rotvec()
            poses[a:b, :3, :3] = Rotation.from_rotvec(p.rest_strength * delta).as_matrix() @ poses[a:b, :3, :3]
    if p.grasp_strength:
        poses = grasp_prior(seq, poses, joints, p.grasp_strength, gaps)
    if p.support_strength:
        n, d = plane(seq.folder)
        points = np.einsum('tij,nj->tni', poses[:, :3, :3], seq.mesh_points)
        clearance = (points @ n).min(axis=1) + poses[:, :3, 3] @ n + d
        grounded = rest & (np.abs(clearance) < 0.15)
        poses[grounded, :3, 3] -= p.support_strength * clearance[grounded, None] * n
    rejected = False
    matched = np.zeros(len(poses), dtype=bool)
    candidate_error = baseline_error = None
    if p.support_guard and rest.any():
        before, height = support_clearance(seq)
        matched = rest & ((height > 0) | (before + height <= 0.25))
        if matched.any():
            candidate_error = float(np.mean(np.abs(ground_heights(seq, poses)[matched] - height[matched])))
            baseline_error = float(np.mean(np.abs(ground_heights(seq, smoothed)[matched] - height[matched])))
            rejected = candidate_error > baseline_error + p.support_margin
            if rejected:
                poses = smoothed
    return replace(seq, object_poses_cv=poses, joints=joints), shift, {
        'rest_frames': rest_frames, 'root_shift_max_cm': float(np.linalg.norm(shift, axis=1).max() * 100),
        'support_frames': int(matched.sum()), 'support_rejected': rejected,
        'support_candidate_error_cm': None if candidate_error is None else candidate_error * 100,
        'support_baseline_error_cm': None if baseline_error is None else baseline_error * 100,
    }


def main() -> None:
    from robotic_grounding.retarget.read_soma import SOMA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT / 'data/work/tier2')
    parser.add_argument('--out', type=Path, default=PROJECT / 'data/work/tier2_denoised')
    args = parser.parse_args()
    if args.source.resolve() != PROJECT / 'data/work/tier2' or args.out.resolve() != PROJECT / 'data/work/tier2_denoised':
        parser.error('this command accepts only data/work/tier2 -> data/work/tier2_denoised')
    soma = SOMA(identity_model_type='mhr')
    rows = []
    for folder in sorted((args.source / 'sequences').iterdir()):
        seq = Sequence.load(folder, soma)
        result, shift, row = denoise(seq)
        out = args.out / 'sequences' / folder.name
        previous = input_signature(out)
        (out / 'reconstructed_mesh').mkdir(parents=True, exist_ok=True)
        for name in ('ground_plane.json', 'filled_frames.json', 'reconstructed_mesh/output_aligned.glb'):
            shutil.copyfile(folder / name, out / name)
        np.save(out / 'poses.npy', result.object_poses_cv)
        with np.load(folder / 'soma_params.npz') as archive:
            params = dict(archive)
        params['transl'] = params['transl'] + shift @ CV_TO_SOMA
        np.savez(out / 'soma_params.npz', **params)
        row['sequence'] = folder.name
        row['input_signature'] = input_signature(out)
        row['input_changed'] = previous != row['input_signature']
        rows.append(row)
        print(json.dumps(row), flush=True)
    source_stage = args.source / 'motion/whole_body/reconstructed_stage'
    shutil.copytree(source_stage, args.out / 'motion/whole_body/reconstructed_stage', dirs_exist_ok=True)
    (args.out / 'denoise_manifest.json').write_text(json.dumps({'params': DEFAULT_PARAMS.__dict__, 'sequences': rows}, indent=2))


if __name__ == '__main__':
    main()
