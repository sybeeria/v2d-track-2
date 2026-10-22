"""Verify denoised input integrity, source-frame invariants, and the actual saved SOMA correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from analyze_denoise import ROOT, EVIDENCE, load_cached
from denoise import DEFAULT_PARAMS, denoise
from hoi import CV_TO_SOMA, Sequence
from robotic_grounding.retarget.read_soma import SOMA
from scipy.spatial.transform import Rotation


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/work/tier2')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/work/tier2_denoised')
    args = parser.parse_args()
    if args.source.resolve() != ROOT / 'data/work/tier2' or args.out.resolve() != ROOT / 'data/work/tier2_denoised':
        parser.error('verification is restricted to Tier 2 and tier2_denoised')
    soma = SOMA(identity_model_type='mhr')
    rows = []
    inputs = sorted((args.source / 'sequences').iterdir())
    metadata = [json.loads(line) for line in (ROOT / 'data/challenge/track_2/tier_2_synthetic_noise/meta/episodes_metadata.jsonl').read_text().splitlines()]
    episodes = {r['sequence_id']: r['episode_index'] for r in metadata}
    assert len(inputs) == len(episodes)
    assert len(list((EVIDENCE / 'frame_references').glob('episode_*.npz'))) == len(episodes)
    assert [f.name for f in inputs] == [f.name for f in sorted((args.out / 'sequences').iterdir())]
    for i, folder in enumerate(inputs):
        out = args.out / 'sequences' / folder.name
        original = np.load(folder / 'poses.npy')
        poses = np.load(out / 'poses.npy')
        assert poses.shape == original.shape and np.isfinite(poses).all()
        np.testing.assert_array_equal(poses[:, 3], np.broadcast_to([0, 0, 0, 1], (len(poses), 4)))
        np.testing.assert_allclose(poses[:, :3, :3] @ poses[:, :3, :3].transpose(0, 2, 1), np.eye(3)[None] * np.ones((len(poses), 1, 1)), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-10)
        for name in ('ground_plane.json', 'filled_frames.json', 'reconstructed_mesh/output_aligned.glb'):
            assert sha(folder / name) == sha(out / name), (folder, name)
        with np.load(folder / 'soma_params.npz') as a, np.load(out / 'soma_params.npz') as b:
            assert set(a.files) == set(b.files)
            for name in a.files:
                if name != 'transl':
                    np.testing.assert_array_equal(a[name], b[name], err_msg=name)
            shift_soma = b['transl'] - a['transl']
        source = load_cached(folder, soma)
        expected, shift_cv, _ = denoise(source, DEFAULT_PARAMS)
        np.testing.assert_allclose(poses, expected.object_poses_cv, atol=1e-12)
        np.testing.assert_allclose(shift_soma, shift_cv @ CV_TO_SOMA, atol=1e-10)
        # Reconstruct saved parameters through the same public SOMA seam the retargeter uses.
        if i in (0, len(inputs)//2, len(inputs)-1):
            reloaded = Sequence.load(out, soma)
            np.testing.assert_allclose(reloaded.joints, expected.joints, atol=2e-6)
        stage = f'motion/whole_body/reconstructed_stage/{folder.name}_support.usda'
        assert sha(args.source / stage) == sha(args.out / stage)
        parquet = args.out / 'motion/whole_body/soma' / f'sequence_id={folder.name}/robot_name=g1/data.parquet'
        assert sha(folder / 'object/textured_mesh.obj') == sha(parquet.parent.parent / 'object/textured_mesh.obj')
        table = pq.read_table(parquet, columns=['object_root_position', 'object_root_axis_angle'])
        robot_t = np.asarray(table.column(0)[0].as_py())
        robot_r = Rotation.from_rotvec(np.asarray(table.column(1)[0].as_py())).as_matrix()
        assert robot_t.shape == (len(poses), 3)
        mapping_r = robot_r[0] @ poses[0, :3, :3].T
        mapping_t = robot_t[0] - mapping_r @ poses[0, :3, 3]
        position_residual = np.linalg.norm(robot_t - (poses[:, :3, 3] @ mapping_r.T + mapping_t), axis=1)
        rotation_residual = Rotation.from_matrix(robot_r @ (mapping_r @ poses[:, :3, :3]).transpose(0, 2, 1)).magnitude()
        assert position_residual.max() < 1e-3
        assert rotation_residual.max() < 1e-3
        episode = episodes[folder.name]
        with np.load(EVIDENCE / 'frame_references' / f'episode_{episode:06d}.npz') as a:
            assert a['episode_index'].item() == episode
            assert a['coordinate_frame'].item() == 'released_track2_world'
            assert a['pose_xyzw'].shape == (len(poses), 1, 7)
            assert np.isfinite(a['pose_xyzw']).all()
            np.testing.assert_array_equal(a['frame_index'], np.arange(len(poses)))
            np.testing.assert_array_equal(a['pose_xyzw'][:, 0, :3], poses[:, :3, 3])
            np.testing.assert_allclose(Rotation.from_quat(a['pose_xyzw'][:, 0, 3:]).as_matrix(), poses[:, :3, :3], atol=1e-12)
        row = dict(sequence=folder.name, frames=len(poses),
            source_poses_sha256=sha(folder / 'poses.npy'), output_poses_sha256=sha(out / 'poses.npy'),
            max_object_translation_change_cm=float(np.linalg.norm(poses[:, :3, 3]-original[:, :3, 3], axis=1).max()*100),
            max_object_rotation_change_deg=float(np.degrees(Rotation.from_matrix(poses[:, :3, :3] @ original[:, :3, :3].transpose(0,2,1)).magnitude()).max()),
            frame_reference_to_g1_max_position_residual_mm=float(position_residual.max()*1000),
            frame_reference_to_g1_max_rotation_residual_rad=float(rotation_residual.max()),
            initial_translation_change_mm=float(np.linalg.norm(poses[0, :3, 3]-original[0, :3, 3])*1000))
        rows.append(row)
        print(json.dumps(row), flush=True)
    (EVIDENCE / 'integrity.json').write_text(json.dumps(rows, indent=2))
    print(f'PASS: {len(rows)} sequences, all frames, rotations, sidecars, meshes, supports, NPZs and G1 frame mappings; 3 saved-SOMA reloads')


if __name__ == '__main__':
    main()
