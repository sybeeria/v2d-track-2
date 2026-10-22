"""Tier-2-only physical residuals and matched diagnostics on three upstream examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from contacts import _runs
from denoise import Params, plane, rest_mask, smooth, support_clearance
from hoi import CV_TO_SOMA, FEET, FPS, Sequence
from noise_stats import rotation_residual_deg
from robotic_grounding.retarget.read_soma import SOMA
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / 'external/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/soma'
EVIDENCE = ROOT / 'data/denoise_eval'


def load_cached(folder: Path, soma: SOMA) -> Sequence:
    files = [folder / name for name in ('poses.npy', 'soma_params.npz', 'reconstructed_mesh/output_aligned.glb')]
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    cache = EVIDENCE / 'cache' / f'{digest}.npz'
    if cache.exists():
        with np.load(cache) as a:
            return Sequence(folder, a['poses'], a['joints'], a['mesh'])
    seq = Sequence.load(folder, soma)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, poses=seq.object_poses_cv, joints=seq.joints, mesh=seq.mesh_points)
    return seq


def stats(seq: Sequence) -> dict:
    n, d = plane(seq.folder)
    feet = seq.joints[:, FEET] @ CV_TO_SOMA
    foot_h = (feet @ n + d).min(axis=1)
    rest = rest_mask(seq, Params())
    clearance, support_h = support_clearance(seq)
    pos = seq.object_poses_cv[:, :3, 3]
    lowest_id = np.argmin(feet @ n + d, axis=1)
    selected = feet[np.arange(len(feet)), lowest_id]
    slide = np.linalg.norm(np.diff(selected, axis=0), axis=1) * FPS
    stable = (lowest_id[1:] == lowest_id[:-1]) & (slide < 0.12)
    with np.load(seq.folder / 'soma_params.npz') as a:
        quats = Rotation.from_rotvec(a['poses'].reshape(-1, 3)).as_quat().reshape(len(pos), -1, 4)
    hf = np.sqrt(np.mean(rotation_residual_deg(quats) ** 2, axis=0))
    result = {
        'sequence': seq.folder.name, 'frames': len(pos), 'rest_frames': int(rest.sum()),
        'object_initial_clearance_cm': float(clearance[0] * 100),
        'object_rest_clearance_cm': np.percentile(clearance[rest], [10, 50, 90]).tolist() if rest.any() else [],
        'object_support_frames': int((support_h > 0).sum()),
        'object_initial_support_height_cm': float(support_h[0] * 100),
        'object_quarter_rest_rms_from_first_cm': [float(np.sqrt(np.mean((clearance[idx][rest[idx]] - clearance[0]) ** 2)) * 100) if rest[idx].any() else None
            for idx in np.array_split(np.arange(len(pos)), 4)],
        'foot_initial_cm': float(foot_h[0] * 100),
        'foot_height_cm': (np.percentile(foot_h, [10, 50, 90]) * 100).tolist(),
        'foot_quarter_rms_from_first_cm': [float(np.sqrt(np.mean((v - np.median(foot_h[:15])) ** 2)) * 100) for v in np.array_split(foot_h, 4)],
        'stance_speed_median_mm_s': float(np.median(slide[stable]) * 1000),
        'human_root_rotation_hf_deg': float(hf[0]), 'human_nonroot_rotation_hf_median_deg': float(np.median(hf[1:])),
        'rest_variogram_position_mm2': {}, 'rest_variogram_rotation_deg2': {}, 'stance_variogram_mm2': {},
        'rest_variogram_pairs': {},
    }
    result['object_rest_clearance_cm'] = [v * 100 for v in result['object_rest_clearance_cm']]
    same = rest[1:] & rest[:-1]
    increments_r = seq.object_poses_cv[1:, :3, :3] @ seq.object_poses_cv[:-1, :3, :3].transpose(0, 2, 1)
    origin_rotation_step = np.einsum('tij,tj->ti', increments_r, pos[:-1]) - pos[:-1]
    result['rest_rotation_about_world_origin_implied_step_mm'] = float(np.sqrt(np.mean(np.sum(origin_rotation_step[same] ** 2, axis=1))) * 1000) if same.any() else None
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 and np.std(a) > 1e-8 and np.std(b) > 1e-8 else None
    result['rest_object_foot_height_corr'] = corr(clearance[rest], foot_h[rest])
    result['rest_object_foot_increment_corr'] = corr(np.diff(clearance)[same], np.diff(foot_h)[same])
    for lag in (1, 3, 10, 30, 60):
        pairs = np.zeros(len(pos) - lag, dtype=bool)
        for a, b in _runs(rest, True):
            if b - a > lag:
                pairs[a:b-lag] = True
        dp = pos[lag:] - pos[:-lag]
        result['rest_variogram_pairs'][str(lag)] = int(pairs.sum())
        dr = Rotation.from_matrix(seq.object_poses_cv[lag:, :3, :3] @ seq.object_poses_cv[:-lag, :3, :3].transpose(0, 2, 1)).magnitude()
        result['rest_variogram_position_mm2'][str(lag)] = float(np.mean(np.sum(dp[pairs] ** 2, axis=1)) * 1e6) if pairs.any() else None
        result['rest_variogram_rotation_deg2'][str(lag)] = float(np.mean(np.degrees(dr[pairs]) ** 2)) if pairs.any() else None
        fs = []
        for j in range(len(FEET)):
            h = feet[:, j] @ n + d
            v = np.linalg.norm(np.gradient(smooth(feet[:, j], 15), axis=0), axis=1) * FPS
            stance = (h < foot_h + 0.02) & (v < 0.1)
            for a, b in _runs(stance, True):
                if b - a > lag:
                    fs.extend(np.sum((feet[a+lag:b, j] - feet[a:b-lag, j]) ** 2, axis=1))
        result['stance_variogram_mm2'][str(lag)] = float(np.mean(fs) * 1e6) if fs else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier2', type=Path, default=ROOT / 'data/work/tier2/sequences')
    args = parser.parse_args()
    if args.tier2.resolve() != (ROOT / 'data/work/tier2/sequences').resolve():
        parser.error('Tier 2 input must be the allowed converted Tier 2 directory')
    soma = SOMA(identity_model_type='mhr')
    rows = []
    for group, folders in [('tier2', sorted(args.tier2.iterdir())), ('upstream', sorted(CLEAN.glob('sequence_id=*')))]:
        for folder in folders:
            row = stats(load_cached(folder, soma))
            row['group'] = group
            rows.append(row)
            print(json.dumps(row), flush=True)
    (EVIDENCE / 'physical_residuals.json').write_text(json.dumps(rows, indent=2))
    (EVIDENCE / 'physical_config.json').write_text(json.dumps({
        'params': Params().__dict__,
        'rest_rule': 'hands >25cm, feet >15cm, max smoothed surface speed <8cm/s, runs >=15 frames',
        'limits': 'Rest/stance are inferred masks. Support matching uses released G1 disks and the upstream retargeter frame chain; unmatched elevated supports remain unknown.'
    }, indent=2))


if __name__ == '__main__':
    main()
