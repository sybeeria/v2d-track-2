"""Rerunnable CPU artifact checks: loader/world-frame score controls and queue blocker.

Run with .venv-flash-cpu/bin/python. Recorder fixtures are synthetic, not policies.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from score_rollout import load_reference, score_rollout, write_json
from tier3_quality import continuity, mesh_plane_bounds


def score_controls(path):
    from flash_chord.data.motion_v1 import load_motion_v1
    from flash_chord.evaluation.rollout import (CompletionOutcome, ObjectPoseRollout,
        reference_object_pose, write_object_trajectories_parquet)
    poses, vertices, mesh, body = load_reference(path)
    loaded = load_motion_v1(str(path), control_fps=50., source_frame_playback=True)
    actual = reference_object_pose(loaded, 0, loaded.num_frames)
    np.testing.assert_allclose(actual[:, 0], poses, atol=2e-6, rtol=0)
    full_mesh = trimesh.load(mesh, force='mesh', process=False)
    matrices = np.tile(np.eye(4), (len(poses), 1, 1))
    matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    matrices[:, :3, 3] = poses[:, :3]
    bottom = mesh_plane_bounds(full_mesh.vertices, matrices, [0,0,1,0])[:, 0]
    row = pq.ParquetFile(path).read(columns=['robot_root_position', 'robot_joint_positions']).to_pylist()[0]
    playback = {'object_translation': continuity(actual[:, 0, :3], 50.),
                'G1_root_translation': continuity(np.asarray(row['robot_root_position']), 50.),
                'G1_joint_angles_rad': continuity(np.asarray(row['robot_joint_positions'])[..., None], 50.,
                                                  speed_limit=20., acceleration_limit=200.)}
    fixture = ObjectPoseRollout(
        achieved_pose_w=actual[:, None], reference_pose_w=actual + np.array([9.,0.,0.,0.,0.,0.,0.]),
        tracking_error=np.zeros((len(poses),1,0)), tracking_error_names=(), object_vertices_o=vertices,
        body_object_ids=np.array([0]), object_body_names=(body,),
        completion=CompletionOutcome(np.array([-1]), np.array([False]), np.array([False]), np.array([0.])),
        control_fps=50., start_frame=0, non_finite_worlds=())
    with tempfile.TemporaryDirectory(prefix='tier3-score-control-') as temp:
        temp = Path(temp)
        result = {}
        for label, offset in [('identity', 0.), ('offset_20cm', .2)]:
            achieved = fixture.achieved_pose_w.copy(); achieved[..., 0] += offset
            output = write_object_trajectories_parquet(temp / f'{label}.parquet', replace(fixture, achieved_pose_w=achieved))
            metrics = score_rollout(output, path)['metrics']
            result[label] = metrics
        assert result['identity']['MPPE'] < .001, result
        assert result['identity']['SP-SR'] == result['identity']['MP-SR'] == 1., result
        assert abs(result['offset_20cm']['MPPE'] - 20.) < .001, result
        assert result['offset_20cm']['SP-SR'] == result['offset_20cm']['MP-SR'] == 0., result
    return {'evidence': 'measured CPU loader + actual evaluator writer + score_rollout/official metrics; synthetic achieved poses',
            'limitation': 'No simulator, physics, policy or CUDA execution. Embedded reference deliberately offset 9m and ignored.',
            'frames': loaded.num_frames, 'loader_max_pose_element_delta': float(np.abs(actual[:,0]-poses).max()),
            'complete_runtime_mesh_vertices': len(full_mesh.vertices),
            'runtime_bottom_min_m': float(bottom.min()), 'runtime_below_z0_frames': np.flatnonzero(bottom < 0).tolist(),
            'G1_playback_continuity_50Hz': playback,
            'controls': result}


def verify(bridge):
    manifest = bridge / 'manifest.json'
    entry = json.loads(manifest.read_text())['sequences'][0]
    result = {'scoring': {}}
    for label in ('tier3_raw', 'tier3_repaired'):
        arm = entry['arms'][label]
        assert arm['training_reference'] == arm['clean_scoring_reference']
        result['scoring'][label] = score_controls(Path(arm['training_reference']['path']))
    assert result['scoring']['tier3_repaired']['runtime_below_z0_frames'] == []
    command = ['bash', 'scripts/pilot_queue.sh', '--arms', 'scripts/pilot_arms.txt', '--arm', 'T2C-contact',
               '--manifest', str(manifest), '--inputs', 'tier3_raw', 'tier3_repaired', '--out-root', '/tmp/x',
               '--gpu', '0', '--steps', '40001536', '49999872', '--dry-run']
    out = Path('/tmp/x')
    before = sorted((str(p.relative_to(out)), p.stat().st_size, p.stat().st_mtime_ns) for p in out.rglob('*')) if out.exists() else None
    completed = subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'}, capture_output=True, text=True)
    after = sorted((str(p.relative_to(out)), p.stat().st_size, p.stat().st_mtime_ns) for p in out.rglob('*')) if out.exists() else None
    assert before == after, 'dry-run changed /tmp/x'
    result['queue'] = {'command': command, 'returncode': completed.returncode,
                       'stdout': completed.stdout, 'stderr': completed.stderr,
                       'out_root_before': before, 'out_root_after': after,
                       'ready': completed.returncode == 0}
    write_json(bridge / 'evidence/unit2_verification.json', result)
    (bridge / 'evidence/unit2_pilot_dry_run.log').write_text(completed.stdout + completed.stderr)
    print(json.dumps({'verification': str(bridge / 'evidence/unit2_verification.json'),
                      'queue_ready': result['queue']['ready'],
                      'runtime_bottom_min_m': {k: v['runtime_bottom_min_m'] for k, v in result['scoring'].items()}}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bridge', type=Path, default=ROOT / 'data/tier3_bridge')
    a = p.parse_args()
    verify(a.bridge.resolve())


if __name__ == '__main__':
    main()
