"""Public coordinator regressions; model boundaries use the existing tiny fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(os.environ.get('B9_REPO', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from tests import test_tier3_bridge_batch as fixtures
from tier3_align import FRAME0_COLUMNS, load_tier1_frame0


class BridgeResumeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.BridgeCoordinatorTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def changed_dependency(self, relative):
        c = self.case
        c.prepare_fresh()
        first = c.invoke(fresh=True)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        path = c.project / relative
        path.write_text(path.read_text() + '\n# changed retarget/repair implementation\n')
        second = c.invoke(fresh=True)
        self.assertNotEqual(second.returncode, 0, 'silently reused completed stages: ' + second.stdout)
        self.assertIn('binding changed', (c.out / 'status.tsv').read_text())
        self.assertFalse((c.out / 'coverage_manifest.json').exists())

    def test_changed_retarget_shell_cannot_reuse_a_finished_reference(self):
        self.changed_dependency('scripts/retarget_tier.sh')

    def test_changed_denoiser_cannot_reuse_a_finished_reference(self):
        self.changed_dependency('scripts/denoise.py')

    def test_engineering_only_flag_is_required_before_any_output(self):
        c = self.case
        result = subprocess.run([sys.executable, '-B', str(c.project / 'scripts/tier3_bridge_batch.py'),
            '--plan', str(c.plan), '--native-root', str(c.native.parent), '--out', str(c.out),
            '--tier1-root', str(c.base / 'never-read'), '--protocol', str(fixtures.fixtures.PROTOCOL)],
            text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn('--released-scene-assets-engineering-only', result.stderr)
        self.assertFalse(c.out.exists())


class FrameZeroTests(unittest.TestCase):
    def test_only_projected_first_batch_is_exposed_and_bad_frame_zero_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'meta').mkdir()
            names = [f'J{i}' for i in range(77)]
            (root / 'meta/info.json').write_text(json.dumps({'data_path': 'ep{episode_index}.parquet',
                'chunks_size': 1000, 'features': {'observation.human.pose': {
                    'names': [f'{n}.{a}' for n in names for a in ('rx', 'ry', 'rz')]}}}))
            (root / 'meta/episodes_metadata.jsonl').write_text(json.dumps({
                'episode_index': 0, 'sequence_id': 'fixture', 'object': 'box'}) + '\n')
            row = {'frame_index': 0, 'observation.human.pose': [0.] * 231,
                'observation.human.translation': [1., 2., 3.],
                'observation.human.identity_coeffs': [0.] * 45,
                'observation.human.scale_params': [0.] * 68,
                'observation.human.bone_length_flexibles': [0.] * 6,
                'observation.object.pose': [0., 0., 1., 1., 0., 0., 0.],
                'observation.object.visible': True, 'caption': 'must not be projected'}
            pq.write_table(pa.Table.from_pylist([row, {**row, 'frame_index': 1,
                'observation.human.translation': [999., 999., 999.]}]), root / 'ep0.parquet')
            actual = pq.ParquetFile
            calls = []

            class OneBatch:
                def __init__(self, *args, **kwargs): self.reader = actual(*args, **kwargs)
                def __enter__(self): return self
                def __exit__(self, *args): self.reader.close()
                def iter_batches(self, **kwargs):
                    calls.append(kwargs)
                    yield next(self.reader.iter_batches(**kwargs))
                    raise AssertionError('later Tier 1 batch requested')

            with patch('tier3_align.pq.ParquetFile', OneBatch):
                frame = load_tier1_frame0(root, 0)
            np.testing.assert_array_equal(frame.transl, [[1., 2., 3.]])
            self.assertEqual(frame.poses_rotvec.shape, (1, 77, 3))
            self.assertEqual(calls, [{'batch_size': 1, 'columns': list(FRAME0_COLUMNS), 'use_threads': False}])
            for change in ({'frame_index': 1}, {'observation.object.visible': False}):
                pq.write_table(pa.Table.from_pylist([{**row, **change}]), root / 'ep0.parquet')
                with self.assertRaises(ValueError): load_tier1_frame0(root, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
