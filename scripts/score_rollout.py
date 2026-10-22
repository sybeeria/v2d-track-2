#!/usr/bin/env python3
"""Score full, single-world upstream rollouts against explicitly supplied clean motion_v1 files.

Repeat --rollout and --reference in matching order. --mesh overrides the clean
parquet's object_mesh_paths; otherwise those paths are re-rooted at its assets/.
MPPE is in cm. No alignment, time resampling, or embedded reference poses are used.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "external/kit/v2d_submission_kit"
METRICS = ("AUC", "SP-SR", "MP-SR", "MPPE")
POSE_SUFFIXES = ("position_x", "position_y", "position_z", "quaternion_x",
                 "quaternion_y", "quaternion_z", "quaternion_w")


@cache
def metric_modules():
    modules = {}
    for name in METRICS:
        spec = importlib.util.spec_from_file_location(name, KIT / "metric_code/track_2" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def score(poses, predicted, vertices):
    # Retained public-score adapter from tier3_recon.md's BEGIN_RECON_PROBE.
    mods = metric_modules()
    columns = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    n = len(poses)
    solution = pd.DataFrame(poses, columns=['ref_' + c for c in columns])
    solution['id'] = np.arange(n).astype(str)
    for key in ('episode_index', 'world_index', 'object_slot'):
        solution[key] = 0
    solution['frame_index'] = np.arange(n)
    solution['object_name'] = 'probe_object'
    buf = io.BytesIO()
    np.savez(buf, probe_object=vertices)
    solution['auc_vertex_samples_b64'] = None
    solution.loc[0, 'auc_vertex_samples_b64'] = base64.b64encode(buf.getvalue()).decode()
    submission = pd.DataFrame(predicted, columns=columns)
    submission['id'] = solution.id
    for key in mods['AUC']._PROVENANCE_COLUMNS:
        submission[key] = '0' * 64
    submission['code_commit_url'] = 'https://github.com/fixture/fixture/commit/0000000'
    return {key: mod.score(solution.copy(), submission.copy(), 'id') for key, mod in mods.items()}


def load_reference(path: Path, mesh: Path | None = None):
    columns = ['schema_version', 'object_body_position', 'object_body_wxyz',
               'object_body_names', 'object_mesh_paths']
    rows = pq.ParquetFile(path).read(columns=columns).to_pylist()
    if len(rows) != 1 or rows[0]['schema_version'] != 'motion_v1':
        raise ValueError(f"{path}: expected one motion_v1 row")
    row = rows[0]
    position = np.asarray(row['object_body_position'], dtype=np.float64)
    quaternion = np.asarray(row['object_body_wxyz'], dtype=np.float64)
    if (position.ndim != 3 or position.shape[1:] != (1, 3) or len(position) == 0
            or quaternion.shape != (len(position), 1, 4)
            or len(row['object_body_names']) != 1 or len(row['object_mesh_paths']) != 1):
        raise ValueError(f"{path}: expected one rigid object with a pose for every frame")
    poses = np.c_[position[:, 0], quaternion[:, 0, [1, 2, 3, 0]]]
    validate_poses(poses, str(path))
    if mesh is None:
        mesh = Path(row['object_mesh_paths'][0])
        if not mesh.is_file():
            # Match FlashCHORD's resolve_asset_path: the final assets/ boundary is authoritative.
            if 'assets' not in mesh.parts:
                raise ValueError(f"{path}: cannot resolve {mesh}; pass --mesh explicitly")
            boundary = max(i for i, part in enumerate(mesh.parts) if part == 'assets')
            suffix = mesh.parts[boundary + 1:]
            assets = next((p for p in path.resolve().parents if p.name == 'assets'), None)
            if assets is None:
                raise ValueError(f"{path}: no assets/ root; pass --mesh explicitly")
            mesh = assets.joinpath(*suffix)
    mesh = mesh.resolve(strict=True)
    sys.path.insert(0, str(KIT))
    from v2dlb.vertices import sample_vertices
    vertices = sample_vertices(mesh)
    if not np.isfinite(vertices).all():
        raise ValueError(f"{mesh}: non-finite mesh vertices")
    return poses, vertices, mesh, row['object_body_names'][0]


def validate_poses(poses, source):
    if not np.isfinite(poses).all() or np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-12):
        raise ValueError(f"{source}: non-finite pose or zero quaternion")


def load_rollout(path: Path, frames: int, body_name: str):
    columns = ['environment_id', 'step', 'reference_frame', 'time_s', 'object_id',
               'object_body_id', 'object_body_name', 'environment_valid']
    columns += ['achieved_' + name for name in POSE_SUFFIXES]
    # Do not even read reference_*: those can be corrupted training inputs.
    table = pq.ParquetFile(path).read(columns=columns)
    metadata = table.schema.metadata or {}
    expected = {b'flash_chord.schema': b'flash_chord_object_trajectories_v1',
                b'flash_chord.quaternion_order': b'xyzw', b'flash_chord.position_unit': b'meter'}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{path}: expected evaluator v1 schema, xyzw quaternions, and meter positions")
    if table.num_rows != frames or any(table[c].null_count for c in columns):
        raise ValueError(f"{path}: expected {frames} non-null rows (one per clean reference frame)")
    df = table.to_pandas().sort_values('reference_frame')
    for name in ['environment_id', 'step', 'reference_frame', 'object_id', 'object_body_id']:
        if not pd.api.types.is_integer_dtype(df[name]):
            raise ValueError(f"{path}: {name} must be integer")
    if not np.array_equal(df.reference_frame, np.arange(frames)):
        raise ValueError(f"{path}: reference_frame must cover 0..{frames - 1} exactly once")
    if not np.array_equal(df.step, df.reference_frame):
        raise ValueError(f"{path}: expected full-episode step == reference_frame (start_frame=0)")
    if any(not (df[c] == 0).all() for c in ['environment_id', 'object_id', 'object_body_id']):
        raise ValueError(f"{path}: expected one world and one object/body, all with id 0")
    if not (df.environment_valid == True).all():
        raise ValueError(f"{path}: environment_valid=false; evaluator sanitised invalid poses")
    if not (df.object_body_name == body_name).all() or not np.isfinite(df.time_s).all():
        raise ValueError(f"{path}: object body name mismatch or non-finite time_s")
    poses = df[['achieved_' + name for name in POSE_SUFFIXES]].to_numpy(dtype=np.float64)
    validate_poses(poses, str(path))
    return poses


def score_rollout(rollout: Path, reference: Path, mesh: Path | None = None):
    clean, vertices, mesh, body_name = load_reference(reference, mesh)
    achieved = load_rollout(rollout, len(clean), body_name)
    return {'rollout': str(rollout.resolve()), 'reference': str(reference.resolve()),
            'rollout_sha256': hashlib.sha256(rollout.read_bytes()).hexdigest(),
            'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
            'mesh': str(mesh), 'frames': len(clean), 'metrics': score(clean, achieved, vertices)}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
            f.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, action='append', required=True)
    parser.add_argument('--reference', type=Path, action='append', required=True,
                        help='explicit CLEAN motion_v1 parquet in the rollout world frame')
    parser.add_argument('--mesh', type=Path, action='append', help='explicit object-local mesh, one per rollout')
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.reference) != len(args.rollout) or (args.mesh and len(args.mesh) != len(args.rollout)):
        parser.error('supply one --reference (and, if used, one --mesh) per --rollout')
    try:
        episodes = [score_rollout(r, ref, mesh) for r, ref, mesh in
                    zip(args.rollout, args.reference, args.mesh or [None] * len(args.rollout))]
        mean = {m: float(np.mean([e['metrics'][m] for e in episodes])) for m in METRICS}
        write_json(args.output, {'episodes': episodes, 'mean': mean, 'mppe_unit': 'cm',
                                'provenance': 'offline diagnostic adapter; synthetic provenance, not a submission',
                                'auc_vertices': {'count': 500, 'seed': 0, 'sampler': 'v2dlb.vertices.sample_vertices'}})
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')
    print('episode\tAUC\tSP-SR\tMP-SR\tMPPE(cm)')
    for e in episodes:
        print(str(e['reference']) + '\t' + '\t'.join(f"{e['metrics'][m]:.8g}" for m in METRICS))
    print('mean\t' + '\t'.join(f'{mean[m]:.8g}' for m in METRICS))


if __name__ == '__main__':
    main()
