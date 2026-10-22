"""Behavioral regression tests. PIPELINE_COPY selects a disposable project copy."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO
sys.path.insert(0, str(ROOT / "scripts"))

class MotionStatsTests(unittest.TestCase):
    def test_rotating_object_with_fixed_origin_counts_as_moving(self):
        from noise_stats import sequence_stats
        with tempfile.TemporaryDirectory(prefix="BUG-PIPELINE-stats-", dir="/tmp") as tmp:
            folder=Path(tmp)
            (folder/"reconstructed_mesh").mkdir()
            frames=91
            poses=np.tile(np.eye(4),(frames,1,1))
            poses[:,:3,:3]=Rotation.from_euler("z",np.linspace(0,1.5,frames)[:,None]).as_matrix()
            poses[:,:3,3]=[0.,-1.5,0.]
            np.save(folder/"poses.npy",poses)
            np.savez(folder/"soma_params.npz",poses=np.zeros((frames,77,3)),transl=np.zeros((frames,3)))
            (folder/"ground_plane.json").write_text(json.dumps({"plane":[0,-1,0,0]}))
            trimesh.creation.box(extents=[2.,2.,2.]).export(folder/"reconstructed_mesh/output_aligned.glb")
            class Body:
                def load_motion(self,path):
                    return {"joints":np.full((frames,77,3),3.)}
            actual=sequence_stats(folder,Body())
            # 0.5 rad/s about z moves every cube corner above the 0.2 m/s threshold.
            self.assertEqual(actual["moving_frac"],1.)
            self.assertAlmostEqual(actual["away_speed_mm_s"], 0., places=8)
            self.assertTrue(np.isfinite(actual["gap_p50_cm"]))
            self.assertGreater(actual["gap_p50_cm"],0.)
            poses[:,:3,:3]=np.eye(3)
            np.save(folder/"poses.npy",poses)
            self.assertEqual(sequence_stats(folder,Body())["moving_frac"],0.)
            poses[:,0,3]=np.arange(frames)/30.
            np.save(folder/"poses.npy",poses)
            translated = sequence_stats(folder,Body())
            self.assertEqual(translated["moving_frac"],1.)
            self.assertAlmostEqual(translated["away_speed_mm_s"], 1000., places=8)


class MeshTests(unittest.TestCase):
    def test_retarget_load_uses_changed_mesh_scale_and_origin(self):
        source = ROOT / "external/video_to_data/robotic_grounding/scripts/retarget/soma_to_g1.py"
        code = tempfile.TemporaryDirectory(prefix="INTEGRATE-retarget-code-", dir="/tmp")
        self.addCleanup(code.cleanup)
        upstream = Path(code.name)/"video_to_data"
        copied = upstream/"robotic_grounding/scripts/retarget/soma_to_g1.py"
        copied.parent.mkdir(parents=True)
        shutil.copy2(source, copied)
        helper = ROOT/"scripts/apply_retarget_patch.sh"
        if helper.is_file():
            for _ in range(2):
                subprocess.run(["bash", str(helper), str(upstream/"robotic_grounding")], check=True,
                               capture_output=True, text=True)
        spec = importlib.util.spec_from_file_location("bug_pipeline_retarget", copied)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="BUG-PIPELINE-mesh-", dir="/tmp") as temp:
            folder = Path(temp)
            (folder / "reconstructed_mesh").mkdir()
            np.savez(folder / "soma_params.npz", fixture=np.array(0))
            np.save(folder / "poses.npy", np.tile(np.eye(4), (2,1,1)))
            glb = folder / "reconstructed_mesh/output_aligned.glb"
            trimesh.creation.box(extents=[.2,.3,.4]).export(glb)
            module.load_data(str(folder))
            changed = trimesh.creation.box(extents=[.4,.6,.8])
            changed.apply_translation([.1,-.2,.3])
            changed.export(glb)
            _, obj, poses = module.load_data(str(folder))
            np.testing.assert_allclose(trimesh.load(obj, force="mesh").bounds, changed.bounds, atol=1e-7)
            np.testing.assert_array_equal(poses, np.tile(np.diag([1.,-1.,-1.,1.]), (2,1,1)))
            before = Path(obj).read_bytes()
            module.load_data(str(folder))
            self.assertEqual(Path(obj).read_bytes(), before)

class CompletenessTests(unittest.TestCase):
    def cli(self, *args):
        env = {**os.environ, "PYTHONPATH":str(ROOT / "external/video_to_data/robotic_grounding/flash_chord/src")}
        return subprocess.run([sys.executable, "-B", str(ROOT/"scripts/check_references.py"), *map(str,args)],
                              env=env, capture_output=True, text=True)

    def test_empty_work_dir_rejected(self):
        with tempfile.TemporaryDirectory(prefix="BUG-PIPELINE-empty-", dir="/tmp") as tmp:
            result = self.cli(tmp)
            self.assertNotEqual(result.returncode,0,result.stdout)

    def test_no_arguments_rejected(self):
        result = self.cli()
        self.assertNotEqual(result.returncode,0,result.stdout)

    def test_valid_reference_accepted_then_missing_episode_rejected(self):
        work = REPO / "data/work/tier2"
        src = sorted(work.glob("motion/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet"))[0]
        name = src.parent.parent.name.removeprefix("sequence_id=")
        with tempfile.TemporaryDirectory(prefix="BUG-PIPELINE-count-", dir="/tmp") as tmp:
            root=Path(tmp)
            ref=root/src.relative_to(work)
            ref.parent.mkdir(parents=True)
            shutil.copyfile(src,ref)
            folder=root/"sequences"/name
            folder.mkdir(parents=True)
            shutil.copyfile(work/"sequences"/name/"poses.npy",folder/"poses.npy")
            result=self.cli(root)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            missing=root/"sequences"/"missing_episode"
            missing.mkdir()
            np.save(missing/"poses.npy",np.tile(np.eye(4),(2,1,1)))
            result=self.cli(root)
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertIn("missing_episode",result.stdout)

if __name__=="__main__":
    unittest.main(verbosity=2)
