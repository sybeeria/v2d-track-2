"""Evaluate physical denoising only on self-corrupted upstream examples with the kit's metrics."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_denoise import CLEAN, EVIDENCE, ROOT, load_cached
from denoise import DEFAULT_PARAMS, Params, denoise, smooth
from hoi import CV_TO_SOMA, Sequence
from noise_stats import rotation_residual_deg, rms
from robotic_grounding.retarget.read_soma import SOMA
from robotic_grounding.retarget.params import SOMA_JOINTS_ORDER
from scipy.spatial.transform import Rotation

# Maximum excursion, not per-step standard deviation. Raw walks match the Tier 2 lag variograms.
LEVELS = {'none': (0., 0., 0.), 'jitter': (0., 0., 0.), 'mild': (3., 15., 2.),
          'medium': (6., 30., 3.), 'severe': (12., 45., 4.), 'extreme': (20., 60., 5.)}
SEEDS = range(5)
SOMA_ROOT = SOMA_JOINTS_ORDER.index('Hips')


def bounded_walk(rng: np.random.Generator, frames: int, bound: float, start_zero: bool = True) -> np.ndarray:
    walk = np.cumsum(rng.normal(size=(frames, 3)), axis=0)
    walk -= walk[0] if start_zero else walk.mean(axis=0)
    return walk * bound / max(np.linalg.norm(walk, axis=1).max(), 1e-12)


def corrupt(seq: Sequence, rng: np.random.Generator, level: tuple[float, float, float],
            jitter: bool = True, start_zero: bool = True, local_rotation: bool = False) -> Sequence:
    cm, deg, human_cm = level
    frames = len(seq.joints)
    poses = seq.object_poses_cv.copy()
    poses[:, :3, 3] += bounded_walk(rng, frames, cm / 100, start_zero)
    rv = bounded_walk(rng, frames, np.radians(deg), start_zero)
    if jitter:
        # The jitter-only case is a separate negative control; mixed walks need little white jitter.
        scale = 1.0 if cm == deg == human_cm == 0 else 0.15
        poses[:, :3, 3] += rng.normal(0, 0.0006 * scale, (frames, 3))
        rv += rng.normal(0, np.radians(1) / np.sqrt(3) * scale, (frames, 3))
    perturbation = Rotation.from_rotvec(rv).as_matrix()
    poses[:, :3, :3] = (poses[:, :3, :3] @ perturbation if local_rotation else
                        perturbation @ poses[:, :3, :3])
    shift = bounded_walk(rng, frames, human_cm / 100, start_zero) @ CV_TO_SOMA
    human_rotation = Rotation.from_rotvec(bounded_walk(rng, frames, np.radians(deg / 4), start_zero)).as_matrix()
    root = seq.joints[:, SOMA_ROOT]
    joints = np.einsum('tij,tkj->tki', human_rotation, seq.joints - root[:, None]) + root[:, None] + shift[:, None]
    return replace(seq, object_poses_cv=poses, joints=joints)


def load_metrics() -> dict:
    metrics = {}
    for name in ('AUC', 'SP-SR', 'MP-SR', 'MPPE'):
        path = ROOT / f'external/kit/v2d_submission_kit/metric_code/track_2/{name}.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module._ALIGN_INITIAL_POSE is False
        metrics[name] = module
    assert json.loads((ROOT / 'external/kit/v2d_submission_kit/config/tracks.json').read_text())['tracks']['track_2_tier2']['align_initial_pose'] is False
    return metrics


def corrupt_matched(seq: Sequence, rng: np.random.Generator, scale: float,
                    centered: bool = False, local_rotation: bool = False) -> Sequence:
    def walk(step):
        w = np.cumsum(rng.normal(0, step * scale, (len(seq.joints), 3)), axis=0)
        return w - (w.mean(axis=0) if centered else w[0])
    poses = seq.object_poses_cv.copy()
    poses[:, :3, 3] += walk(0.00075)
    r = Rotation.from_rotvec(walk(np.radians(0.71))).as_matrix()
    poses[:, :3, :3] = poses[:, :3, :3] @ r if local_rotation else r @ poses[:, :3, :3]
    shift = walk(0.0004) @ CV_TO_SOMA
    human_r = Rotation.from_rotvec(walk(np.radians(0.2))).as_matrix()
    root = seq.joints[:, SOMA_ROOT]
    joints = np.einsum('tij,tkj->tki', human_r, seq.joints-root[:, None]) + root[:, None] + shift[:, None]
    return replace(seq, object_poses_cv=poses, joints=joints)


def pose7(poses: np.ndarray) -> np.ndarray:
    return np.column_stack([poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()])


def metric_frames(seq: Sequence) -> pd.DataFrame:
    rows = len(seq.object_poses_cv)
    columns = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    frame = pd.DataFrame(pose7(seq.object_poses_cv), columns=['ref_' + c for c in columns])
    for name, value in {'row_id': np.arange(rows).astype(str), 'episode_index': 0, 'world_index': 0,
                        'frame_index': np.arange(rows), 'object_slot': 0, 'object_name': 'self_corruption'}.items():
        frame[name] = value
    payload = io.BytesIO()
    np.savez(payload, self_corruption=seq.mesh_points)
    frame['auc_vertex_samples_b64'] = ''
    frame.loc[0, 'auc_vertex_samples_b64'] = base64.b64encode(payload.getvalue()).decode()
    return frame


def score(metrics: dict, pred: np.ndarray, reference: pd.DataFrame) -> dict[str, float]:
    submission = pd.DataFrame(pose7(pred), columns=metrics['AUC']._POSE_COLUMNS)
    submission['row_id'] = reference['row_id']
    # Synthetic local fixture provenance; this is never a submission or challenge reference.
    for column in metrics['AUC']._PROVENANCE_COLUMNS:
        submission[column] = ('https://github.com/nvidia-isaac/video_to_data/commit/7eb9721'
                              if column == 'code_commit_url' else '0' * 64)
    return {name: float(module.score(reference, submission, 'row_id')) for name, module in metrics.items()}


def variants() -> dict[str, Params]:
    base = Params()
    return {
        'smooth7': replace(base, window=7, rest_strength=0, root_strength=0),
        'smooth15': replace(base, rest_strength=0, root_strength=0),
        'smooth31': replace(base, window=31, rest_strength=0, root_strength=0),
        'default': base,
        'rest_gap20': replace(base, propagate_rest=True, rest_gap=0.20),
        'rest_gap35': replace(base, propagate_rest=True, rest_gap=0.35),
        'rest_speed05': replace(base, propagate_rest=True, rest_speed=0.05),
        'rest_speed12': replace(base, propagate_rest=True, rest_speed=0.12),
        'root_half': replace(base, root_strength=0.5),
        'grasp_half': replace(base, grasp_strength=0.5),
        'grasp_full': replace(base, grasp_strength=1.0),
        'ground_half': replace(base, support_strength=0.5),
        'rest_walk': replace(base, propagate_rest=True),
        'rest_walk_half': replace(base, propagate_rest=True, rest_strength=0.5),
        'root_vertical_init': replace(base, vertical_only=True),
        'root_vertical15': replace(base, vertical_only=True, foot_height=0.015),
        'candidate': replace(base, propagate_rest=True, vertical_only=True, root_strength=0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--centered', action='store_true', help='Stress a nonzero initial corruption.')
    parser.add_argument('--local-rotation', action='store_true', help='Stress object-frame rotation walks.')
    parser.add_argument('--methods', nargs='+', choices=list(variants()))
    parser.add_argument('--matched', action='store_true', help='Per-step variances fitted to Tier 2 rest residuals.')
    parser.add_argument('--selected', action='store_true', help='Evaluate the frozen production parameters.')
    parser.add_argument('--grasp-check', action='store_true', help='Isolate grasp priors from horizontal root integration.')
    parser.add_argument('--frame-check', action='store_true', help='Compare world- and local-frame rotation bias propagation.')
    parser.add_argument('--guard-check', action='store_true', help='Compare physical rejection with unguarded drift correction.')
    args = parser.parse_args()
    soma = SOMA(identity_model_type='mhr')
    metrics = load_metrics()
    levels = {'none': 0.0, 'half': 0.5, 'matched': 1.0, 'double': 2.0} if args.matched else LEVELS
    rows = []
    noise_rows = []
    configs = variants()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    if args.guard_check:
        configs = {'selected': DEFAULT_PARAMS,
            'local_unguarded': replace(DEFAULT_PARAMS, support_guard=False),
            'world_unguarded': replace(DEFAULT_PARAMS, rotation_frame='world', support_guard=False),
            'smooth15': Params(rest_strength=0, root_strength=0)}
    elif args.frame_check:
        base = Params(root_strength=0.5, rest_strength=0.5, propagate_rest=True, vertical_only=True)
        configs = {'world': replace(base, rotation_frame='world'),
                   'local': replace(base, rotation_frame='local')}
    elif args.grasp_check:
        base = Params(root_strength=0.5, rest_strength=0.5, propagate_rest=True, vertical_only=True)
        configs = {'selected': base,
            'grasp_vertical_half': replace(base, grasp_strength=0.5),
            'grasp_vertical_full': replace(base, grasp_strength=1.0),
            'grasp_no_root': replace(base, root_strength=0, grasp_strength=0.5)}
    elif args.selected:
        configs = {'selected': DEFAULT_PARAMS}
    elif args.methods:
        configs = {k: configs[k] for k in args.methods}
    elif args.quick:
        configs = {k: configs[k] for k in ('smooth15', 'default', 'grasp_half', 'ground_half', 'rest_walk', 'rest_walk_half')}
    for folder in sorted(CLEAN.glob('sequence_id=*')):
        seq = load_cached(folder, soma)
        truth = metric_frames(seq)
        identity = score(metrics, seq.object_poses_cv, truth)
        assert identity == {'AUC': 1.0, 'SP-SR': 1.0, 'MP-SR': 1.0, 'MPPE': 0.0}, identity
        for (level, values), seed in itertools.product(levels.items(), range(args.seeds)):
            noisy = (corrupt_matched(seq, np.random.default_rng(seed), values, args.centered, args.local_rotation)
                if args.matched else corrupt(seq, np.random.default_rng(seed), values, level != 'none',
                                              not args.centered, args.local_rotation))
            gaps = noisy.hand_gaps()
            foot_gaps = noisy.foot_gaps()
            noise_rows.append(dict(sequence=folder.name, level=level, seed=seed,
                obj_t_hf_mm=rms(np.linalg.norm(noisy.object_poses_cv[:, :3, 3] -
                    smooth(noisy.object_poses_cv[:, :3, 3]), axis=1)) * 1000,
                obj_r_hf_deg=rms(rotation_residual_deg(Rotation.from_matrix(noisy.object_poses_cv[:, :3, :3]).as_quat())),
                carrying_gap_cm=float(np.median(gaps.min(axis=0)[noisy.object_moving()])) * 100))
            for name, params in [('baseline', None), *configs.items()]:
                output, shift, diagnostic = (noisy, None, {}) if params is None else denoise(noisy, params, gaps, foot_gaps)
                result = score(metrics, output.object_poses_cv, truth)
                result['human_joint_rmse_cm'] = float(np.sqrt(np.mean(np.sum((output.joints - seq.joints) ** 2, axis=-1))) * 100)
                rows.append(dict(sequence=folder.name, level=level, seed=seed, method=name, **result, **diagnostic))
            print(f'{folder.name} {level} seed={seed}', flush=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    stem = 'validation_quick' if args.quick else 'validation'
    if args.matched:
        stem += '_matched'
    if args.selected:
        stem += '_selected'
    if args.grasp_check:
        stem += '_grasp_check'
    if args.frame_check:
        stem += '_frame_check'
    if args.guard_check:
        stem += '_guard_check'
    if args.centered:
        stem += '_centered'
    if args.local_rotation:
        stem += '_local'
    pd.DataFrame(rows).to_csv(EVIDENCE / f'{stem}.csv', index=False)
    grouped = pd.DataFrame(rows).groupby(['level', 'method'])[['AUC', 'SP-SR', 'MP-SR', 'MPPE', 'human_joint_rmse_cm']].mean()
    grouped.to_csv(EVIDENCE / f'{stem}_summary.csv')
    pd.DataFrame(noise_rows).to_csv(EVIDENCE / f'{stem}_noise.csv', index=False)
    print(grouped.to_string())
    (EVIDENCE / f'{stem}_config.json').write_text(json.dumps({'levels': levels, 'seeds': list(range(args.seeds)),
        'variants': {k: v.__dict__ for k, v in configs.items()}, 'metric_mesh_points': 4000,
        'centered': args.centered, 'local_rotation': args.local_rotation,
        'matched_step_std': {'object_m': 0.00075, 'object_deg': 0.71, 'root_m': 0.0004, 'root_deg': 0.2} if args.matched else None,
        'notes': 'Official score() in all four participant metric files, no initial alignment. Same frames and deterministic mesh samples for every method.'}, indent=2))


if __name__ == '__main__':
    main()
