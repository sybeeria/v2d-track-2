"""Observable repair, manifest, asset audit and shared gap contracts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import trimesh
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tier3_align import CV_TO_SOMA, load_soma
from tier3_quality import contact_gaps, mesh_plane_bounds
from tier3_scene_assets import ASSETS, compare_assets
from tier3_scene_assets import SEQUENCE
from hoi import FINGERTIPS


class RepairTest(unittest.TestCase):
    def fixture(self, source, count=41):
        (source / 'reconstructed_mesh').mkdir(parents=True)
        data = dict(poses=np.zeros((count, 77, 3)), transl=np.zeros((count, 3)),
                    bone_length_flexibles=np.zeros((count, 6)), joint_names=np.array([str(i) for i in range(77)]),
                    identity_model_type=np.array('mhr'), identity_coeffs=np.ones((1, 45)),
                    scale_params=np.ones((1, 68)), unit=np.array('meters'), keep_root=np.array(False))
        data['transl'][20, 0] = 1.
        np.savez(source / 'soma_params.npz', **data)
        poses = np.tile(np.eye(4), (count, 1, 1)); poses[:, 2, 3] = 2.; poses[20, 0, 3] = 1.
        np.save(source / 'poses.npy', poses)
        (source / 'ground_plane.json').write_text('{"plane":[0,0,1,0]}')
        trimesh.creation.box(extents=[1., 1., 1.]).export(source / 'reconstructed_mesh/output_aligned.glb')
        return data, poses

    def cli(self, source, out):
        return subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/tier3_repair.py'),
                               '--source', str(source), '--out', str(out)], capture_output=True, text=True)

    def test_cli_smooths_both_motions_exactly_and_preserves_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'input', Path(tmp) / 'repaired'
            data, poses = self.fixture(source)
            r = self.cli(source, out); self.assertEqual(r.returncode, 0, r.stderr)
            saved = load_soma(out / 'soma_params.npz'); obj = np.load(out / 'poses.npy')
            expected = np.zeros(41)
            expected[13:28] = np.array([-78,-13,42,87,122,147,162,167,162,147,122,87,42,-13,-78]) / 1105
            assert_allclose(saved['transl'][:, 0], expected, atol=1e-13)
            assert_allclose(obj[:, 0, 3], expected, atol=1e-13)
            for name in ('poses', 'identity_coeffs', 'scale_params', 'joint_names', 'bone_length_flexibles'):
                assert_array_equal(saved[name], data[name])
            assert_allclose(obj[:, 2, 3], poses[:, 2, 3], atol=2e-14, rtol=0)
            before = (out / 'poses.npy').read_bytes()
            self.assertNotEqual(self.cli(source, out).returncode, 0)
            self.assertEqual((out / 'poses.npy').read_bytes(), before)

    def test_cli_ground_shift_is_minimal_full_mesh_and_shared_in_both_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'input', Path(tmp) / 'repaired'
            data, poses = self.fixture(source)
            poses[:, :3, 3] = [0., 0., .25]
            data['transl'][:] = [1., 2., 3.]
            np.save(source / 'poses.npy', poses); np.savez(source / 'soma_params.npz', **data)
            (source / 'ground_plane.json').write_text('{"plane":[0,0,2,0]}')
            r = self.cli(source, out); self.assertEqual(r.returncode, 0, r.stderr)
            saved = load_soma(out / 'soma_params.npz'); obj = np.load(out / 'poses.npy')
            assert_allclose(obj[:, 2, 3], .5001, atol=1e-14)
            assert_allclose(saved['transl'], np.tile([1., 2., 2.7499], (41, 1)), atol=1e-14)
            assert_allclose(obj[:, :3, 3] - saved['transl'] @ CV_TO_SOMA,
                            poses[:, :3, 3] - data['transl'] @ CV_TO_SOMA, atol=1e-14)
            vertices = trimesh.load(out / 'reconstructed_mesh/output_aligned.glb', force='mesh', process=False).vertices
            assert_allclose(mesh_plane_bounds(vertices, obj, [0,0,2,0])[:, 0], .0001, atol=1e-14)

    def test_cli_rejects_nonrigid_object_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'input', Path(tmp) / 'repaired'
            _, poses = self.fixture(source); poses[5, 0, 0] = 2.
            np.save(source / 'poses.npy', poses)
            self.assertNotEqual(self.cli(source, out).returncode, 0)
            self.assertFalse(out.exists())

    def test_cli_smooths_object_and_local_joint_rotations_on_so3(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'input', Path(tmp) / 'repaired'
            data, poses = self.fixture(source)
            angles = np.full(41, .3); angles[20] = .6
            data['poses'][:, 7, 2] = angles
            poses[:, :3, :3] = Rotation.from_rotvec(np.c_[np.zeros((41, 2)), angles]).as_matrix()
            np.save(source / 'poses.npy', poses); np.savez(source / 'soma_params.npz', **data)
            r = self.cli(source, out); self.assertEqual(r.returncode, 0, r.stderr)
            saved = load_soma(out / 'soma_params.npz'); obj = np.load(out / 'poses.npy')
            c = 167 / 1105
            expected = 2 * np.arctan2(c*np.sin(.3) + (1-c)*np.sin(.15), c*np.cos(.3) + (1-c)*np.cos(.15))
            self.assertAlmostEqual(saved['poses'][20, 7, 2], expected, places=13)
            self.assertAlmostEqual(Rotation.from_matrix(obj[20,:3,:3]).as_rotvec()[2], expected, places=13)
            assert_array_equal(saved['poses'][0], data['poses'][0])
            assert_allclose(np.linalg.det(obj[:,:3,:3]), np.ones(41), atol=1e-14)


class GapAndSceneTest(unittest.TestCase):
    def test_gaps_use_labelled_frames_and_object_local_coordinates(self):
        frames = 3
        poses = np.tile(np.eye(4), (frames, 1, 1)); poses[:, 0, 3] = [1., 2., 3.]
        joints = np.zeros((frames, 77, 3))
        joints[:, FINGERTIPS['Left']] = (poses[:, None, :3, 3] + np.array([0., .03, 0.])) @ CV_TO_SOMA
        row = dict(object_body_position=poses[:, None, :3, 3], object_body_wxyz=np.tile([1.,0.,0.,0.], (frames,1,1)),
                   ee_pose_w=np.c_[poses[:, :3, 3] + [[0.,.05,0.],[0.,.10,0.],[0.,.15,0.]], np.zeros((frames,4))][:, None],
                   hand_sides=['left'], ee_link_names=['left_hand_palm_link'], hand_contact_active=[[1.,0.,1.]])
        gap = contact_gaps(np.array([[0.,0.,0.]]), poses, joints, row)['left']
        self.assertEqual(gap['labelled_frames'], [0,2])
        self.assertEqual(gap['G1_palm_nearest_vertex_m_on_labels'],
                         {'count':2,'min':.05,'median':.1,'p95':.145,'max':.15})
        self.assertEqual(gap['human_fingertip_nearest_vertex_m_on_labels']['median'], .03)

    def test_asset_audit_is_exact_and_rejects_different_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for tier in ('tier_1_multiview_caption', 'tier_2_synthetic_noise'):
                for name in ASSETS:
                    p = root / tier / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'fixture')
            result = compare_assets(root)
            self.assertEqual(len(result['assets']), 3)
            self.assertEqual([r['sha256'] for r in result['assets']], [hashlib.sha256(b'fixture').hexdigest()] * 3)
            (root / 'tier_2_synthetic_noise' / ASSETS[0]).write_bytes(b'changed')
            with self.assertRaises(ValueError): compare_assets(root)


class ManifestTest(unittest.TestCase):
    def test_cli_regenerates_absolute_self_references_after_host_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = Path(tmp) / 'host1/bridge'
            for base, z in [(bridge, 1.), (bridge / 'repaired', 2.)]:
                motion = base / 'assets/human_motion_data/whole_body'
                path = motion / f'soma/sequence_id={SEQUENCE}/robot_name=g1/data.parquet'
                path.parent.mkdir(parents=True)
                pq.write_table(pa.Table.from_pylist([{'coord_frame':'robot_base_z_up', 'object_body_position':[[[0.,0.,z]]]}]), path)
                stage = motion / f'reconstructed_stage/{SEQUENCE}_support.usda'
                stage.parent.mkdir(parents=True)
                stage.write_text('#usda 1.0\ndef Xform "support_surfaces" {}\n')
            for host in ('host1', 'host2'):
                if host == 'host2':
                    moved = Path(tmp) / 'host2/bridge'; moved.parent.mkdir(); bridge.rename(moved); bridge = moved
                output = bridge / 'manifest.json'
                r = subprocess.run([sys.executable, '-I', str(ROOT/'scripts/tier3_manifest.py'),
                                    '--bridge', str(bridge), '--out', str(output)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                manifest = json.loads(output.read_text())
                self.assertEqual(manifest['scoring_mode'], 'self')
                arms = manifest['sequences'][0]['arms']
                self.assertEqual(set(arms), {'tier3_raw','tier3_repaired'})
                for name in ('tier3_raw','tier3_repaired'):
                    arm = arms[name]; ref = arm['training_reference']
                    self.assertEqual(arm['clean_scoring_reference'], ref)
                    self.assertTrue(Path(ref['path']).is_relative_to(bridge.resolve()))
                    self.assertEqual(ref['sha256'], hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest())
                    self.assertIs(arm['support']['enabled'], False)
                self.assertNotEqual(arms['tier3_raw']['training_reference'], arms['tier3_repaired']['training_reference'])


if __name__ == '__main__':
    unittest.main()
