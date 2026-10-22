"""CPU checks using upstream examples, never challenge trajectories."""

from __future__ import annotations

import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lerobot_to_soma import Episode
from tier3_align import (ALIGN_JOINTS, CV_TO_SOMA, FRAME0_COLUMNS, Similarity, align_sequence,
                        estimate_alignment, load_soma, load_tier1_frame0, soma_frame0_joints)


class Tier3AlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from robotic_grounding.retarget.read_soma import SOMA
        torch.set_num_threads(2)
        cls.soma = SOMA(device=torch.device("cpu"))
        cls.addClassCleanup(cls.release_model)
        cls.example = sorted((ROOT / "external/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/soma").glob("sequence_id=*"))[0]
        cls.params = load_soma(cls.example / "soma_params.npz")
        cls.objects = np.load(cls.example / "poses.npy")
        cls.temporary = tempfile.TemporaryDirectory(prefix="v2d-alignment-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.metrics = {}

    @classmethod
    def release_model(cls):
        # Later bridge tests start another body-model process.
        del cls.soma
        gc.collect()

    def test_similarity_noise_and_lever_arm(self):
        rng = np.random.default_rng(20261021)
        with tempfile.TemporaryDirectory(dir=self.work) as temp:
            joints = soma_frame0_joints(self.params, self.soma, Path(temp) / "first.npz")
        names = list(self.params["joint_names"])
        points = joints[[names.index(n) for n in ALIGN_JOINTS]]
        known = Similarity(1.37, Rotation.random(random_state=rng).as_matrix(), rng.normal(size=3))
        inverse = Similarity(1 / known.scale, known.rotation.T, -known.rotation.T @ known.translation / known.scale)
        corrupted = inverse.points(points)
        recovered = estimate_alignment(corrupted, points, similarity=True)
        assert_allclose(recovered.scale, known.scale, atol=1e-12)
        assert_allclose(recovered.rotation, known.rotation, atol=1e-12)
        assert_allclose(recovered.translation, known.translation, atol=1e-12)
        trajectory = inverse.points(self.objects[:, :3, 3])
        assert_allclose(recovered.points(trajectory), self.objects[:, :3, 3], atol=1e-12)
        assert_allclose(recovered.poses(inverse.poses(self.objects)), self.objects, atol=1e-12)
        noisy = estimate_alignment(corrupted + rng.normal(0, .01, corrupted.shape), points, similarity=True)
        errors = np.linalg.norm(noisy.points(trajectory) - self.objects[:, :3, 3], axis=-1)
        rotation_error = np.degrees(Rotation.from_matrix(noisy.rotation @ known.rotation.T).magnitude())
        sensitivity = []
        pivot = points.mean(0)
        path = self.objects[:, :3, 3] - pivot
        for angle in (.5, 1., 3., 5.):
            r = Rotation.from_euler("y", angle, degrees=True).as_matrix()
            error = np.linalg.norm(path @ r.T - path, axis=1)
            sensitivity.append({"rotation_error_deg": angle, "trajectory_rmse_m": float(np.sqrt(np.mean(error**2))),
                                "trajectory_max_m": float(error.max())})
            lever = np.array([[1., 0., 0.], [3., 0., 0.], [5., 0., 0.]])
            assert_allclose(np.linalg.norm(lever @ r.T-lever, axis=1),
                            2*np.array([1., 3., 5.])*np.sin(np.radians(angle)/2), atol=1e-12)
        self.metrics.update(example=self.example.name, frames=len(trajectory), random_similarity_scale=known.scale,
                            exact_sim3_max_translation_error_m=float(np.abs(recovered.points(trajectory)-self.objects[:, :3, 3]).max()),
                            noise_sigma_source_m=.01, noisy_sim3_scale=noisy.scale,
                            noisy_rotation_error_deg=float(rotation_error), noisy_trajectory_rmse_m=float(np.sqrt(np.mean(errors**2))),
                            noisy_trajectory_max_m=float(errors.max()), sensitivity=sensitivity,
                            lever_arm_1deg_m={str(d): float(2*d*np.sin(np.radians(1)/2)) for d in (1,3,5)})

    def test_folder_export_through_real_soma_reader(self):
        data = self.params
        with tempfile.TemporaryDirectory(dir=self.work) as temp_dir:
            temp = Path(temp_dir)
            source = temp / "raw"
            (source / "reconstructed_mesh").mkdir(parents=True)
            perturb = Similarity(1., Rotation.from_euler("xyz", [.27, -.51, 1.1]).as_matrix(), np.array([.7,-.2,.4]))
            native_r = CV_TO_SOMA @ perturb.rotation @ CV_TO_SOMA
            native_t = CV_TO_SOMA @ perturb.translation
            altered = {key: value.copy() for key, value in data.items()}
            altered["poses"][:, 0] = Rotation.from_matrix(native_r @ Rotation.from_rotvec(data["poses"][:, 0]).as_matrix()).as_rotvec()
            altered["transl"] = data["transl"] @ native_r.T + native_t
            np.savez(source / "soma_params.npz", **altered)
            # Also perturb the object-local mesh basis and origin, independently.
            local = np.eye(4)
            local[:3, :3] = Rotation.from_euler("xyz", [.3,.1,-.4]).as_matrix()
            local[:3, 3] = [.04,-.02,.03]
            np.save(source / "poses.npy", perturb.poses(self.objects) @ np.linalg.inv(local))
            import trimesh
            mesh = trimesh.load(self.example / "reconstructed_mesh/output_aligned.glb", force="scene", process=False)
            mesh.apply_transform(local)
            mesh.export(source / "reconstructed_mesh/output_aligned.glb")
            (source / "filled_frames.json").write_text('{"frames": [2]}')
            ground = self.example / "ground_plane.json"
            support = temp / "support.usda"
            support.write_text("#usda 1.0\n")
            ref_object = self.objects[:1]
            reference = Episode(0, "synthetic", "example", list(data["joint_names"]), data["poses"][:1], data["transl"][:1],
                                data["identity_coeffs"][0], data["scale_params"][0], data["bone_length_flexibles"][:1],
                                np.c_[ref_object[:, :3, 3], Rotation.from_matrix(ref_object[:, :3, :3]).as_quat(scalar_first=True)],
                                np.ones(1,bool), Path(), Path(), Path())
            folder = align_sequence(source, reference, temp / "sequences", temp / "motion", ground, support, self.soma)
            result = load_soma(folder / "soma_params.npz")
            actual = np.load(folder / "poses.npy")
            assert_allclose(actual, self.objects, atol=3e-6)
            assert_allclose(result["transl"], data["transl"], atol=3e-6)
            assert_allclose(Rotation.from_rotvec(result["poses"][:,0]).as_matrix(),
                            Rotation.from_rotvec(data["poses"][:,0]).as_matrix(), atol=3e-6)
            for key in ("identity_coeffs", "scale_params"):
                assert_allclose(result[key], data[key][:1], atol=0)
            self.assertEqual(result["poses"].shape, data["poses"].shape)
            self.assertEqual(json.loads((folder / "filled_frames.json").read_text())["frames"], [2])
            self.assertEqual((temp / "motion/whole_body/reconstructed_stage/synthetic_support.usda").read_bytes(), support.read_bytes())
            self.assertEqual((folder / "ground_plane.json").read_bytes(), ground.read_bytes())
            saved_mesh = trimesh.load(folder / "reconstructed_mesh/output_aligned.glb", force="mesh", process=False)
            original_mesh = trimesh.load(self.example / "reconstructed_mesh/output_aligned.glb", force="mesh", process=False)
            assert_allclose(saved_mesh.vertices, original_mesh.vertices, atol=3e-6)
            # Decode representative saved frames through the retargeter's reader.
            indices = [0, len(actual)//2, len(actual)-1]
            for params, filename in ((result,"actual.npz"), (data,"expected.npz")):
                np.savez(temp/filename, **{k: v[indices] if k in ("poses","transl","bone_length_flexibles") else v[:1] if k in ("identity_coeffs","scale_params") else v for k,v in params.items()})
            got = self.soma.load_motion(temp/"actual.npz")
            expected = self.soma.load_motion(temp/"expected.npz")
            assert_allclose(got["joints"], expected["joints"], atol=2e-5)
            self.metrics["folder_object_max_error_m"] = float(np.abs(actual[:,:3,3]-self.objects[:,:3,3]).max())
            self.metrics["folder_decoded_joint_max_error_m"] = float(np.abs(got["joints"]-expected["joints"]).max())

    def test_frame0_loader_never_requests_a_second_batch(self):
        data = self.params
        row = {"frame_index": 0, "observation.human.pose": data["poses"][0].flatten().tolist(),
               "observation.human.translation": data["transl"][0].tolist(),
               "observation.human.identity_coeffs": data["identity_coeffs"][0].tolist(),
               "observation.human.scale_params": data["scale_params"][0].tolist(),
               "observation.human.bone_length_flexibles": data["bone_length_flexibles"][0].tolist(),
               "observation.object.pose": [0.,0.,1.,1.,0.,0.,0.], "observation.object.visible": True}
        with tempfile.TemporaryDirectory(dir=self.work) as temp_dir:
            root = Path(temp_dir)
            (root/"meta").mkdir()
            (root/"meta/info.json").write_text(json.dumps({"data_path":"episode_{episode_index}.parquet", "chunks_size":1000,
                "features":{"observation.human.pose":{"names":[f"{n}.{a}" for n in data["joint_names"] for a in ("rx","ry","rz")]}}}))
            (root/"meta/episodes_metadata.jsonl").write_text(json.dumps({"episode_index":0,"sequence_id":"synthetic","object":"test"})+"\n")
            bad = {**row, "frame_index":1, "observation.human.translation":[999.,999.,999.]}
            pq.write_table(pa.Table.from_pylist([row,bad]), root/"episode_0.parquet")
            real = pq.ParquetFile
            calls=[]
            class FirstBatchOnly:
                def __init__(self,*args,**kwargs): self.pf=real(*args,**kwargs)
                def __enter__(self): return self
                def __exit__(self,*args): self.pf.close()
                def iter_batches(self,**kwargs):
                    calls.append(kwargs)
                    yield next(self.pf.iter_batches(**kwargs))
                    raise AssertionError("attempted to read a later Tier 1 batch")
            with patch("tier3_align.pq.ParquetFile", FirstBatchOnly), patch("tier3_align.pq.read_table", side_effect=AssertionError("full read forbidden")):
                episode = load_tier1_frame0(root, 0)
            assert_allclose(episode.transl, data["transl"][:1])
            self.assertEqual(episode.poses_rotvec.shape, (1,77,3))
            self.assertEqual(calls, [{"batch_size":1,"columns":list(FRAME0_COLUMNS),"use_threads":False}])

    def test_degenerate_alignment_rejected(self):
        with self.assertRaisesRegex(ValueError, "collinear"):
            estimate_alignment(np.zeros((4,3)),np.zeros((4,3)),similarity=True)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            estimate_alignment(np.full((4,3),np.nan),np.zeros((4,3)))

    @classmethod
    def tearDownClass(cls):
        (cls.work / "alignment_test_results.json").write_text(json.dumps(cls.metrics, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
