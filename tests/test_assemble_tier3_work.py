"""Assemble/transfer verification through the public CLI and actual packer check."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(os.environ.get('B9_REPO', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / 'scripts'))
from ep18_pilot import SEQUENCE, LABEL

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


class ExportWorkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bridge = self.root / 'bridge'
        self.bridge.mkdir()
        self.candidate = self.root / 'candidate_v1'
        self.tier = self.root / 'tier2'
        (self.tier / 'meta').mkdir(parents=True)
        self.work = self.root / 'export-work'
        self.manifest = self.root / 'manifest.json'
        entries, covered, metadata = [], [], []
        self.sources = {}
        for ep in range(30):
            sequence = SEQUENCE if ep == 18 else f'fixture{ep:02d}'
            label = LABEL if ep == 18 else 'tier3_rest_repaired' if ep % 2 else 'tier3_repaired'
            source_root = self.candidate if ep == 18 else self.bridge / f'ep{ep:02d}'
            source = source_root / f'sequences/{sequence}/poses.npy'
            source.parent.mkdir(parents=True)
            poses = np.tile(np.eye(4), (4, 1, 1))
            poses[:, 0, 3] = np.arange(4) * .01 + ep
            np.save(source, poses)
            self.sources[sequence] = source
            reference = source_root / f'assets/human_motion_data/whole_body/soma/sequence_id={sequence}/robot_name=g1/data.parquet'
            reference.parent.mkdir(parents=True)
            # A different fixed world frame is legal; the trajectory must remain exact.
            pos = poses[:, :3, 3].copy() + [10., 20., 30.]
            pq.write_table(pa.Table.from_pylist([{'object_body_position': pos[:, None].tolist(),
                'object_body_wxyz': np.tile([1., 0., 0., 0.], (4, 1, 1)).tolist()}]), reference)
            record = {'path': str(reference), 'sha256': sha(reference)}
            entry = {'sequence': sequence, 'frames': 4, 'arms': {label: {
                'training_reference': record, 'clean_scoring_reference': record}}}
            entries.append(entry)
            covered.append({**entry, 'episode': ep, 'gate': {'status': 'HOLD'}})
            if ep != 18:
                stage = 'video_rest_repair' if ep % 2 else 'repair'
                (source_root / f'.{stage}.done.json').write_text(json.dumps({'artifacts': [
                    [str(source), source.stat().st_size, sha(source)]]}))
            metadata.append({'episode_index': ep, 'sequence_id': sequence, 'frames': 4})
            pq.write_table(pa.table({'frame_index': np.arange(4)}), self.tier / f'ep{ep}.parquet')
        # Deliberately wrong generic ep18 output: the assembler must use candidate_v1.
        generic = self.bridge / f'ep18/sequences/{SEQUENCE}/poses.npy'
        generic.parent.mkdir(parents=True)
        np.save(generic, np.zeros((4, 4, 4)))
        self.manifest.write_text(json.dumps({'tier': 3, 'scoring_mode': 'self', 'sequences': entries}))
        (self.bridge / 'coverage_manifest.json').write_text(json.dumps({'tier': 3, 'scoring_mode': 'self', 'sequences': covered}))
        (self.tier / 'meta/info.json').write_text(json.dumps({'total_episodes': 30, 'chunks_size': 1000, 'data_path': 'ep{episode_index}.parquet'}))
        (self.tier / 'meta/episodes_metadata.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metadata))

    def invoke(self, action='assemble', *extra):
        command = [sys.executable, '-B', str(ROOT / 'scripts/assemble_tier3_work.py'), action,
            '--manifest', str(self.manifest), '--tier-root', str(self.tier), '--work', str(self.work)]
        if action == 'assemble': command += ['--bridge', str(self.bridge), '--ep18-work', str(self.candidate)]
        return subprocess.run([*command, *extra], text=True, capture_output=True, timeout=30)

    def test_exact_30_files_candidate18_idempotency_preflight_and_converter(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('30 exact converter files', result.stdout)
        for sequence, source in self.sources.items():
            self.assertEqual((self.work / f'sequences/{sequence}/poses.npy').read_bytes(), source.read_bytes())
        before = (self.work / 'work_manifest.json').read_bytes()
        self.assertEqual(self.invoke().returncode, 0)
        self.assertEqual((self.work / 'work_manifest.json').read_bytes(), before)
        check = subprocess.run([sys.executable, '-B', str(ROOT / 'data/brainstorm7/REVIEW/frame_preflight.py'),
            str(self.manifest), str(self.work)], text=True, capture_output=True, timeout=30)
        self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
        self.assertIn('30/30 sequences pass', check.stdout)
        converted = self.root / 'converted'
        check = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/frame_references.py'),
            '--work-dir', str(self.work), '--tier-root', str(self.tier), '--out', str(converted)],
            text=True, capture_output=True, timeout=30)
        self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
        with np.load(converted / 'episode_000018.npz') as packet:
            np.testing.assert_array_equal(packet['pose_xyzw'][:, 0, :3], [[18., 0., 0.], [18.01, 0., 0.], [18.02, 0., 0.], [18.03, 0., 0.]])
            np.testing.assert_array_equal(packet['frame_index'], [0, 1, 2, 3])

    def test_changed_source_bytes_are_rejected_before_publishing_work(self):
        source = self.sources['fixture00']
        poses = np.load(source)
        poses[2, 0, 3] += .2
        np.save(source, poses)
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('completed bridge stage', result.stderr)
        self.assertFalse(self.work.exists())

    def test_wrong_poses_reject_even_with_current_stage_hash(self):
        source = self.sources['fixture00']
        poses = np.load(source)
        poses[2, 0, 3] += .2
        np.save(source, poses)
        (self.bridge / 'ep00/.repair.done.json').write_text(json.dumps({'artifacts': [
            [str(source), source.stat().st_size, sha(source)]]}))
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not one rigid transform', result.stderr)
        self.assertFalse(self.work.exists())

    def test_wrong_candidate_and_changed_release_reference_refuse(self):
        result = self.invoke('assemble', '--ep18-work', str(self.bridge / 'ep18'))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.work.exists())
        manifest = json.loads(self.manifest.read_text())
        reference = next(iter(manifest['sequences'][0]['arms'].values()))['training_reference']['path']
        Path(reference).write_bytes(b'changed reference')
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('reference changed', result.stderr)
        self.assertFalse(self.work.exists())

    def test_transferred_work_verifies_without_converter_sources_then_rejects_tampering(self):
        self.assertEqual(self.invoke().returncode, 0)
        # The host retains training references, but not the Mac converter source tree.
        for source in self.sources.values():
            source.unlink()
        result = self.invoke('verify')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        target = self.work / 'sequences/fixture00/poses.npy'
        target.write_bytes(target.read_bytes() + b'tampered')
        result = self.invoke('verify')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('sealed converter poses/reference changed', result.stderr)

    def test_incomplete_or_duplicate_release_is_rejected_without_output(self):
        original = json.loads(self.manifest.read_text())
        for sequences in (original['sequences'][:-1], original['sequences'][:-1] + [original['sequences'][0]]):
            self.manifest.write_text(json.dumps({**original, 'sequences': sequences}))
            result = self.invoke()
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.work.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
