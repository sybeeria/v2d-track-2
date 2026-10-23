"""Public rest-repair CLI: exact elevated support, preserved motion and input refusal."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


class VideoRestTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.source = self.root / 'source'; self.source.mkdir()
        (self.source / 'reconstructed_mesh').mkdir()
        self.poses = np.tile(np.eye(4), (3,1,1)); self.poses[:,2,3] = 3
        np.save(self.source / 'poses.npy', self.poses)
        np.savez(self.source / 'soma_params.npz', poses=np.zeros((3,77,3)))
        (self.source / 'ground_plane.json').write_text('{"plane":[0,-1,0,2]}')
        (self.source / 'filled_frames.json').write_text('{"frames": []}')
        (self.source / 'tier3_alignment.json').write_text(json.dumps(dict(reference_frames_used=[0], scale=1,
            rotation=np.eye(3).tolist(), translation=[0,0,0])))
        trimesh.creation.box(extents=[1,1,1]).export(self.source / 'reconstructed_mesh/output_aligned.glb')
        self.camera = self.root / 'camera.json'
        self.camera.write_text(json.dumps(dict(intrinsics=[[800,0,768],[0,800,576],[0,0,1]])))
        self.evidence = self.root / 'evidence.json'; self.out = self.root / 'output'
        self.config = dict(frame_count=3,image_width=1536,image_height=1152, protected_motion=[[1,3]],
            polygons={'0':[[608,416],[928,416],[928,736],[608,736]]},
            rests=[dict(rest=[0,1],blend_extent=[0,1],fit_frames=[0],support='table',support_plane=[0,-2,0,2])])
        self.evidence.write_text(json.dumps(self.config))

    def invoke(self, *extra):
        return subprocess.run([sys.executable,'-B',str(ROOT / 'scripts/tier3_rest_repair.py'),
            '--source',str(self.source),'--camera',str(self.camera),'--evidence',str(self.evidence),
            '--episode','0','--frames','3','--video-sha256','a'*64,'--legacy-video-sha256','a'*64,
            '--out',str(self.out),*extra],capture_output=True,text=True,timeout=20)

    def test_exact_elevated_anchor_preserves_protected_motion_and_source_bytes(self):
        result = self.invoke(); self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        expected = self.poses.copy(); expected[0,:3,3] = [0,.4999,3]
        np.testing.assert_array_equal(np.load(self.out / 'poses.npy'),expected)
        for name in ('soma_params.npz','ground_plane.json','filled_frames.json','tier3_alignment.json','reconstructed_mesh/output_aligned.glb'):
            self.assertEqual((self.out/name).read_bytes(),(self.source/name).read_bytes())
        report = json.loads((self.out / 'video_rest_repair.json').read_text())
        self.assertEqual(report['phases'][0]['selected_source_frame'],0)
        self.assertEqual(report['evidence_sha256'],hashlib.sha256(self.evidence.read_bytes()).hexdigest())
        before = (self.out / 'poses.npy').read_bytes()
        result = self.invoke(); self.assertNotEqual(result.returncode,0)
        self.assertEqual((self.out / 'poses.npy').read_bytes(),before)

    def test_missing_support_plane_is_hold_without_output(self):
        self.config['rests'][0]['support_plane'] = None
        self.evidence.write_text(json.dumps(self.config))
        result = self.invoke(); self.assertEqual(result.returncode,2,result.stderr)
        self.assertEqual(json.loads(result.stdout),dict(episode=0,status='HOLD',
            reasons=['explicit aligned plane missing for table rest 0:1']))
        self.assertFalse(self.out.exists())

    def test_changed_evidence_hash_is_refused_without_output(self):
        result = self.invoke('--evidence-sha256','0'*64)
        self.assertNotEqual(result.returncode,0); self.assertIn('rest evidence bytes changed',result.stderr)
        self.assertFalse(self.out.exists())

    def test_nonrigid_source_and_overlapping_motion_are_refused_without_output(self):
        self.poses[1,0,0] = 2; np.save(self.source / 'poses.npy',self.poses)
        result = self.invoke(); self.assertNotEqual(result.returncode,0)
        self.assertIn('rigid SE3 required',result.stderr); self.assertFalse(self.out.exists())
        self.config['protected_motion'] = [[0,3]]; self.evidence.write_text(json.dumps(self.config))
        result = self.invoke(); self.assertNotEqual(result.returncode,0)
        self.assertIn('protected real/unknown motion overlaps a repair',result.stderr); self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
