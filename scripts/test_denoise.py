"""Behavior checks for static, foot-driven, and rotating objects through denoise()."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from denoise import Params, denoise, ground_heights
from hoi import CV_TO_SOMA, FEET, Sequence
from scipy.spatial.transform import Rotation


class DenoiseBehavior(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / 'data/denoise_eval'
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.folder = Path(self.temporary.name)
        (self.folder / 'ground_plane.json').write_text(json.dumps({'plane': [0, -1, 0, 0]}))
        self.frames = 91
        poses = np.tile(np.eye(4), (self.frames, 1, 1))
        poses[:, :3, 3] = [0, -0.5, 0]
        mesh = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (-0.02, 0.02)])
        self.seq = Sequence(self.folder, poses, np.full((self.frames, 77, 3), 3.0), mesh)
        self.params = Params(root_strength=0, propagate_rest=True)

    def tearDown(self):
        self.temporary.cleanup()

    def test_static_walk_is_removed_without_mutating_input(self):
        rng = np.random.default_rng(4)
        poses = self.seq.object_poses_cv.copy()
        poses[:, :3, 3] += np.cumsum(rng.normal(0, 0.0003, (self.frames, 3)), axis=0)
        original = poses.copy()
        noisy = replace(self.seq, object_poses_cv=poses)
        output, _, _ = denoise(noisy, self.params)
        self.assertEqual(output.object_poses_cv.shape, poses.shape)
        self.assertLess(np.linalg.norm(np.std(output.object_poses_cv[10:-10, :3, 3], axis=0)), 1e-12)
        self.assertGreater(np.linalg.norm(np.std(poses[10:-10, :3, 3], axis=0)), 0.0001)
        np.testing.assert_array_equal(noisy.object_poses_cv, original)

    def test_slow_foot_push_is_preserved(self):
        poses = self.seq.object_poses_cv.copy()
        poses[:, 0, 3] = np.arange(self.frames) * 0.001
        joints = self.seq.joints.copy()
        joints[:, FEET] = ((poses[:, None, :3, 3] + self.seq.mesh_points[0]) @ CV_TO_SOMA)
        seq = replace(self.seq, object_poses_cv=poses, joints=joints)
        output, _, _ = denoise(seq, self.params)
        np.testing.assert_allclose(output.object_poses_cv, poses, atol=1e-12)
        self.assertAlmostEqual(output.object_poses_cv[-1, 0, 3], 0.09)

    def test_rotation_with_stationary_origin_is_preserved(self):
        poses = self.seq.object_poses_cv.copy()
        poses[:, :3, :3] = Rotation.from_rotvec(np.column_stack([np.linspace(0, 1, self.frames),
            np.zeros((self.frames, 2))])).as_matrix()
        seq = replace(self.seq, object_poses_cv=poses, mesh_points=self.seq.mesh_points * 40)
        output, _, _ = denoise(seq, self.params)
        angle = Rotation.from_matrix(output.object_poses_cv[-1, :3, :3] @ output.object_poses_cv[0, :3, :3].T).magnitude()
        self.assertGreater(angle, 0.999)
        np.testing.assert_allclose(output.object_poses_cv[:, :3, 3], poses[:, :3, 3], atol=1e-12)

    def test_ground_height_matches_a_rotated_cube(self):
        poses = self.seq.object_poses_cv.copy()
        poses[:, :3, :3] = Rotation.from_euler('z', 45, degrees=True).as_matrix()
        np.testing.assert_allclose(ground_heights(self.seq, poses), 0.5 - 0.02*np.sqrt(2), atol=1e-12)

    def test_support_guard_preserves_a_slow_tipping_box(self):
        frames = 181
        theta = np.linspace(0, np.pi/2, frames)
        poses = np.tile(np.eye(4), (frames, 1, 1))
        poses[:, :3, :3] = Rotation.from_euler('z', theta[:,None]).as_matrix()
        poses[:, 1, 3] = -0.1*(np.sin(theta)+np.cos(theta))
        seq = replace(self.seq, object_poses_cv=poses, mesh_points=self.seq.mesh_points*5,
                      joints=np.full((frames, 77, 3),3.0))
        p = Params(root_strength=0, rest_strength=0.5, propagate_rest=True, rotation_frame='local')
        unsafe, _, _ = denoise(seq, p)
        guarded, _, row = denoise(seq, replace(p, support_guard=True))
        self.assertGreater(np.mean(np.abs(ground_heights(seq, unsafe.object_poses_cv))),0.005)
        self.assertLess(np.max(np.abs(ground_heights(seq, guarded.object_poses_cv))),0.001)
        self.assertTrue(row['support_rejected'])

    def test_local_bias_follows_a_carried_object_rotation(self):
        frames = 121
        clean = np.tile(np.eye(4), (frames, 1, 1))
        clean[:, :3, 3] = [0,-0.5,0]
        clean[60:, :3, :3] = Rotation.from_euler('z',np.linspace(0,np.pi/2,frames-60)[:,None]).as_matrix()
        noise = Rotation.from_euler('x',np.r_[np.linspace(0,0.2,60),np.full(frames-60,0.2)][:,None]).as_matrix()
        poses = clean.copy();poses[:, :3, :3] = clean[:, :3, :3] @ noise
        joints = np.full((frames,77,3),3.0)
        near = np.einsum('tij,j->ti',poses[60:,:3,:3],self.seq.mesh_points[0])+poses[60:,:3,3]
        joints[60:] = (near @ CV_TO_SOMA)[:,None]
        seq = replace(self.seq, object_poses_cv=poses, joints=joints)
        p = Params(window=1,root_strength=0,propagate_rest=True,rotation_frame='local')
        local = denoise(seq,p)[0].object_poses_cv
        world = denoise(seq,replace(p,rotation_frame='world'))[0].object_poses_cv
        def error(output):
            return Rotation.from_matrix(output[-1,:3,:3] @ clean[-1,:3,:3].T).magnitude()
        self.assertLess(error(local),0.02)
        self.assertGreater(error(world),0.1)


if __name__ == '__main__':
    unittest.main()
