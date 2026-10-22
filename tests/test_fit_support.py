"""Exercise the fitter CLI and the actual FlashCHORD scene builder on synthetic clean motion."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh

ROOT = Path(__file__).resolve().parents[1]


class FitSupportTest(unittest.TestCase):
    def test_rest_carried_and_ground_through_cli_and_scene_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            mesh = work / 'box.obj'
            trimesh.creation.box(extents=[.2, .2, .2]).export(mesh)
            reference = work / 'motion/whole_body/soma/sequence_id=synthetic/robot_name=g1/data.parquet'
            reference.parent.mkdir(parents=True)
            positions = np.tile([1., 2., 1.1], (90, 1))
            positions[30:60] = [3, 2, 1.5]
            positions[60:] = [1, 2, 1.1]
            contacts = np.zeros((2, 90))
            contacts[0, 30:60] = 1
            def write(p, c):
                pq.write_table(pa.Table.from_pylist([dict(object_root_position=p.tolist(),
                    object_root_axis_angle=np.zeros_like(p).tolist(), hand_contact_active=c.tolist(),
                    object_mesh_paths=[str(mesh)])]), reference)
            write(positions, contacts)
            command = [sys.executable, str(ROOT / 'scripts/fit_support.py'), str(reference)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(len(report['disks']), 1)
            disk = report['disks'][0]
            np.testing.assert_allclose(disk['center_xy'], [1, 2], atol=1e-12)
            self.assertAlmostEqual(disk['top_z_m'], 1.)
            self.assertAlmostEqual(disk['radius_m'], np.sqrt(.02) + .01)
            self.assertEqual(disk['intervals'], [[0, 26], [64, 90]])
            self.assertEqual([r['xy_overlap_frames'] for r in disk['rest_intervals']], [26, 26])
            self.assertEqual(disk['carried_intersection_candidates'], 0)
            self.assertGreater(disk['carried_min_clearance_m'], 1)
            stage = Path(report['output'])
            before = stage.read_bytes()
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(stage.read_bytes(), before)
            flash = ROOT / 'external/video_to_data/robotic_grounding/flash_chord'
            code = '''import sys, newton
from flash_chord.assets.registry import support_usda_for_parquet
from flash_chord.scene.support import add_support_surfaces
from pxr import Usd
path = support_usda_for_parquet(sys.argv[1], robot_name="g1")
stage = Usd.Stage.Open(str(path))
assert any(p.GetTypeName() == "Cylinder" for p in stage.Traverse())
builder = newton.ModelBuilder()
binding = add_support_surfaces(builder, path)
assert len(binding.shapes) == 1
assert abs(builder.shape_transform[0].p[2] - .98) < 1e-6
print("PASS: detector and scene builder accepted fitted disk")
'''
            result = subprocess.run([str(ROOT / '.venv-flash-cpu/bin/python'), '-c', code, str(reference)],
                                    capture_output=True, text=True,
                                    env={**os.environ, 'PYTHONPATH': str(flash / 'src')})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            write(positions, np.ones_like(contacts))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['disks'], [])
            write(np.tile([1., 2., .1], (90, 1)), np.zeros_like(contacts))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['disks'], [])
            moving = positions.copy()
            moving[:, 0] = np.arange(90) / 30
            write(moving, np.zeros_like(contacts))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['disks'], [])
            moving = np.tile([1., 2., 1.1], (90, 1))
            moving[:, 0] += np.maximum(np.arange(90) - 5, 0) * .01
            write(moving, np.zeros_like(contacts))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['disks'][0]['intervals'], [[0, 2]])
            write(np.tile([1., 2., 1.1], (90, 1)), np.zeros_like(contacts))
            row = pq.read_table(reference).to_pylist()[0]
            row['object_root_axis_angle'] = np.c_[np.zeros((90, 2)), np.arange(90) * .03].tolist()
            pq.write_table(pa.Table.from_pylist([row]), reference)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['disks'], [])


if __name__ == '__main__':
    unittest.main()
