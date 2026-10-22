#!/usr/bin/env python3
"""Pack Track 2 Tier 2/3 rollouts using only the public row sample and frozen input NPZs.

No kit packer, dataset reader, clean reference, metric computation, or network access.
The frame conversion extends SOL-T2's research prototype, using the kit's pose math.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'external/kit/v2d_submission_kit'
sys.path.insert(0, str(KIT))
from v2dlb import object_pose as op, report_schema as rs
from v2dlb.submission_parquet import read_trajectory, provenance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_frame(achieved, evaluator, target):
    if (achieved.shape != evaluator.shape or achieved.shape != target.shape
            or achieved.ndim != 2 or achieved.shape[1] != 7 or not len(achieved)):
        raise ValueError('pose arrays must be nonempty, matching [frames,7]')
    if not all(np.isfinite(a).all() for a in (achieved, evaluator, target)):
        raise ValueError('non-finite pose')
    achieved, evaluator, target = [op._normalise_quaternions(a, 'pose', ValueError)
                                    for a in (achieved, evaluator, target)]
    rotation_quat = op._quaternion_multiply(
        target[:1, 3:], evaluator[:1, 3:] * [-1., -1., -1., 1.])[0]
    rotation_quat /= np.linalg.norm(rotation_quat)
    rotation = op._rotation_matrix(rotation_quat[None])[0]
    offset = target[0, :3] - rotation @ evaluator[0, :3]
    converted_reference = evaluator[:, :3] @ rotation.T + offset
    converted_quat = op._quaternion_multiply(rotation_quat, evaluator[:, 3:])
    position_gap = float(np.linalg.norm(converted_reference-target[:, :3], axis=1).max())
    angle_gap = float(op._quaternion_geodesic_angle(converted_quat, target[:, 3:]).max())
    if position_gap > 1e-3 or angle_gap > 1e-3:
        raise ValueError(f'input/evaluator reference is not one rigid transform: {position_gap} m, {angle_gap} rad')
    return np.c_[achieved[:, :3] @ rotation.T + offset,
                 op._quaternion_multiply(rotation_quat, achieved[:, 3:])]


def pack(results, references, out, track, commit):
    if track not in {'track_2_tier2', 'track_2_tier3'}:
        raise ValueError('only Track 2 Tier 2/3 use this builder')
    if not re.fullmatch(rs.CODE_COMMIT_PATTERN, commit):
        raise ValueError('a pinned GitHub commit URL is required')
    sample = pd.read_csv(KIT / 'data/track_2_sample_submission.csv', dtype=str)
    expected_columns = [rs.ROW_ID_COLUMN, *op._POSE_COLUMNS, *rs.PROVENANCE_COLUMNS]
    if list(sample.columns) != expected_columns or sample.row_id.duplicated().any():
        raise ValueError('unexpected public sample schema')
    keys = sample.row_id.str.extract(r'^track_2/e(\d{6})/w(\d{3})/f(\d{6})/o(\d+)$')
    if keys.isna().any().any():
        raise ValueError('invalid public sample row id')
    keys.columns = ['episode', 'world', 'frame', 'slot']
    keys = keys.astype(int)
    if (keys[['world', 'slot']] != 0).any().any():
        raise ValueError('expected one world and one object per episode')
    trajectories = {}
    for path in sorted(results.rglob('*.parquet')):
        table = pq.ParquetFile(path).read()
        md = table.schema.metadata or {}
        for key, value in {b'flash_chord.schema':b'flash_chord_object_trajectories_v1',
                           b'flash_chord.position_unit':b'meter',
                           b'flash_chord.quaternion_order':b'xyzw'}.items():
            if md.get(key) != value:
                raise ValueError(f'{path}: require evaluator v1, metres and xyzw')
        for key in ['environment_id', 'step', 'reference_frame', 'object_id', 'object_body_id']:
            if table[key].null_count or not pd.api.types.is_integer_dtype(table[key].to_pandas()):
                raise ValueError(f'{path}: null/noninteger {key}')
        if any(table[c].null_count for c in table.column_names):
            raise ValueError(f'{path}: null rollout value')
        t = read_trajectory(path)
        if t.episode_index in trajectories:
            raise ValueError('duplicate episode rollout; freeze one evaluation per policy')
        trajectories[t.episode_index] = (t, table)
    if set(trajectories) != set(keys.episode):
        raise ValueError('rollouts must cover exactly the public sample episode roster')
    blocks, uniform = [], {}
    for episode, grid in keys.groupby('episode', sort=True):
        t, table = trajectories[episode]
        with np.load(references / f'episode_{episode:06d}.npz', allow_pickle=False) as archive:
            if (str(archive['coordinate_frame'].item()) != 'released_track2_world'
                    or int(archive['episode_index'].item()) != episode):
                raise ValueError('wrong reference episode or coordinate frame')
            frames, target = archive['frame_index'], archive['pose_xyzw']
        if (frames.dtype.kind not in 'iu' or frames.ndim != 1 or not len(frames)
                or not np.array_equal(frames, np.arange(len(frames)))
                or target.shape != (len(frames), 1, 7)):
            raise ValueError('input NPZ must contain the complete contiguous single-object frame grid')
        order = np.argsort(t.frame_index)
        if (not np.array_equal(t.frame_index[order], frames)
                or not np.array_equal(table['step'].to_numpy()[order], frames)
                or np.any(t.world_index != 0) or np.any(t.object_slot != 0)
                or np.any(t.object_id != 0) or not t.world_valid.all()
                or t.reference_xyzw is None):
            raise ValueError('rollout must have one valid world/body and every input frame once from start=0')
        if not set(grid.frame).issubset(set(frames)):
            raise ValueError('input NPZ misses public scored frames')
        converted = convert_frame(t.pose_xyzw[order], t.reference_xyzw[order], target[:, 0])
        recorded = provenance(t)
        hashes = {key:recorded.get(key, '') for key in
                  ['eval_script_sha256','eval_code_sha256','checkpoint_sha256']}
        if any(not re.fullmatch('[0-9a-fA-F]{64}', value) for value in hashes.values()):
            raise ValueError('every episode must record the three evaluator/checkpoint SHA256 digests')
        for key in ['eval_script_sha256','eval_code_sha256']:
            if key in uniform and uniform[key] != hashes[key]:
                raise ValueError('mixed evaluator builds')
            uniform[key] = hashes[key]
        block = pd.DataFrame(converted[grid.frame], columns=op._POSE_COLUMNS)
        block.insert(0, 'row_id', sample.loc[grid.index, 'row_id'].to_numpy())
        for key,value in hashes.items():
            block[key] = value
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True).set_index('row_id').loc[sample.row_id].reset_index()
    result['packer_script_sha256'] = digest(__file__)
    result['uploader_script_sha256'] = digest(KIT / 'submit/v2d_submit.py')
    result['code_commit_url'] = commit
    result = result[expected_columns]
    if result.isna().any().any():
        raise ValueError('null submission cell')
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, float_format='%.17g')
    print(f'{len(result)} rows -> {out}; no clean scoring performed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=['track_2_tier2','track_2_tier3'], required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--frame-references', type=Path, required=True)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('-o','--out',type=Path,required=True)
    args = parser.parse_args()
    try:
        pack(args.results, args.frame_references, args.out, args.track, args.code_commit)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
