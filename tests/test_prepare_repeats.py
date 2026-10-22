"""Public runner test: the same checkpoint yields k fresh evaluator calls and zero retrainings."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ORIGINAL_ROOT = HERE.parent
ROOT = Path(os.environ.get('EVAL_NOISE_PROJECT_ROOT', ORIGINAL_ROOT))
sys.path.insert(0, str(ORIGINAL_ROOT / 'tests'))
import test_score_rollout as base


class RepeatJobsTest(unittest.TestCase):
    manifest = base.PilotQueueTest.manifest
    fake_python = base.PilotQueueTest.fake_python

    def test_evaluate_only_missing_checkpoint_never_trains(self):
        with tempfile.TemporaryDirectory(prefix='EVAL-NOISE-') as temporary:
            work = Path(temporary).resolve()
            manifest = self.manifest(work)
            events, env = self.fake_python(work)
            command = [str(ROOT / 'scripts/pilot_queue.sh'), '--arms', str(ROOT / 'scripts/pilot_arms.txt'),
                       '--manifest', str(manifest), '--arm', 'E1-control', '--inputs', 'clean',
                       '--sequences', base.SEQUENCES[0], '--steps', '49999872',
                       '--out-root', str(work / 'no-checkpoint'), '--evaluate-only']
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing requested checkpoint', result.stderr)
            self.assertFalse(events.exists())
            # test_day can select the same guarantee through its inherited environment.
            result = subprocess.run(command[:-1], env={**env, 'PILOT_EVALUATE_ONLY': '1'}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing requested checkpoint', result.stderr)
            self.assertFalse(events.exists())

    def test_existing_cache_and_three_fresh_roots_through_test_day(self):
        with tempfile.TemporaryDirectory(prefix='EVAL-NOISE-') as temporary:
            work = Path(temporary).resolve()
            manifest = self.manifest(work)
            events, env = self.fake_python(work)
            source = work / 'source'
            command = [str(ROOT / 'scripts/pilot_queue.sh'), '--arms', str(ROOT / 'scripts/pilot_arms.txt'),
                       '--manifest', str(manifest), '--arm', 'E1-control', '--inputs', 'clean',
                       '--sequences', base.SEQUENCES[0], '--steps', '49999872', '--out-root', str(source / 'fixture')]
            for _ in range(2):
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(events.read_text().split(), ['train', 'evaluate'])
            checkpoint = next(source.rglob('policy_*.safetensors'))
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            plan = dict(repeats=3, candidates=[dict(name='fixture', source='fixture/E1-control', group='screen',
                arm='E1-control', arms_file='pilot_arms.txt', input='clean', sequence=base.SEQUENCES[0],
                steps=[49999872], final_step=49999872)])
            plan_path = work / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            output = work / 'repeats'
            prep = [sys.executable, str(ROOT / 'scripts/prepare_repeats.py'), '--plan', str(plan_path),
                    '--source-root', str(source), '--out-root', str(output)]
            for _ in range(2):
                result = subprocess.run(prep, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(['bash', str(ROOT / 'scripts/test_day.sh'), '1', str(output / 'screen')],
                env={**env, 'PILOT_EVALUATE_ONLY':'1', 'JOBS_FILE': str(output / 'jobs-screen.txt'), 'MANIFEST': str(manifest)},
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(events.read_text().split(), ['train', 'evaluate', 'evaluate', 'evaluate', 'evaluate'])
            with (output / 'screen/summary.tsv').open() as file:
                rows = list(csv.DictReader(file, delimiter='\t'))
            self.assertEqual(len(rows), 3)
            self.assertEqual([float(row['AUC']) for row in rows], [1., 1., 1.])
            trajectories = list(output.rglob('trajectory.parquet'))
            self.assertEqual(len(trajectories), 3)
            self.assertEqual(len({str(p) for p in trajectories}), 3)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)
            self.assertEqual({json.loads(p.read_text())['checkpoint_sha256'] for p in output.rglob('evaluation.done.json')}, {before})
            # Missing source checkpoints fail before producing any runnable job root.
            checkpoint.unlink()
            bad = prep.copy()
            missing_output = work / 'missing'
            bad[-1] = str(missing_output)
            result = subprocess.run(bad, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing checkpoint; no training is permitted', result.stderr)
            self.assertFalse(missing_output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
