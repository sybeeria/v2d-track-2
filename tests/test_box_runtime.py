"""Positive controls for GPU worker exclusivity and hook failure propagation; no real GPU/provider."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import unittest

ROOT = Path(os.environ.setdefault('BOX_REPO', str(Path(__file__).resolve().parents[1]))).resolve()
os.environ['BUG_RUNNER_REPO'] = str(ROOT)
spec = importlib.util.spec_from_file_location('runner_fixture', ROOT / 'tests/test_runner_regressions.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class BoxRuntime(unittest.TestCase):
    def test_one_and_two_gpus_use_exactly_one_worker_per_gpu(self):
        for gpus in (1, 2):
            with self.subTest(gpus=gpus):
                fixture = base.QueueTests()
                fixture.setUp()
                try:
                    pilot = fixture.root / 'scripts/pilot_queue.sh'
                    text = pilot.read_text()
                    text = text.replace('out.mkdir(parents=True,exist_ok=True)', '''out.mkdir(parents=True,exist_ok=True)
import fcntl
_gpu_guard=open(str(Path(os.environ['EVENTS']).parent/('gpu-'+args[args.index('--gpu')+1]+'.lock')),'w')
try: fcntl.flock(_gpu_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: sys.exit(93)''')
                    pilot.write_text(text)
                    names = [f'job{i}' for i in range(6)]
                    fixture.plan(names)
                    result = fixture.execute(['bash', 'scripts/test_day.sh', str(gpus), str(fixture.out)], JOB_DELAY='.05')
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    calls = [json.loads(x) for x in fixture.events.read_text().splitlines()]
                    self.assertEqual({x['job']: x['gpu'] for x in calls},
                                     {name: str(i % gpus) for i, name in enumerate(names)})
                    self.assertEqual(len(calls), 6)
                finally:
                    fixture.doCleanups()

    def run_box(self, fixture, box):
        process = subprocess.Popen(['bash', str(box)], cwd=fixture.root, env=fixture.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=15)
            return process.returncode, stdout, stderr
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def test_nonzero_post_hook_marks_box_failed(self):
        fixture = base.TrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        box = fixture.make_box()
        (fixture.root / 'scripts/test_day.sh').write_text('#!/bin/sh\nexit 0\n')
        after = fixture.work / 'should-not-run'
        (fixture.work / 'workspace/batch.post.sh').write_text(f'set -euo pipefail\nfalse\ntouch "{after}"\n')
        status, stdout, stderr = self.run_box(fixture, box)
        self.assertEqual(status, 1, stdout + stderr)
        self.assertIn('box failed ', stdout)
        self.assertNotIn('box done ', stdout)
        self.assertFalse(after.exists())

    def test_nonzero_pre_hook_never_starts_the_queue(self):
        fixture = base.TrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        box = fixture.make_box()
        marker = fixture.work / 'queue-was-started'
        (fixture.root / 'scripts/test_day.sh').write_text(f'#!/bin/sh\ntouch "{marker}"\n')
        (fixture.work / 'workspace/batch.pre.sh').write_text('set -euo pipefail\nfalse\n')
        status, stdout, stderr = self.run_box(fixture, box)
        self.assertEqual(status, 1, stdout + stderr)
        self.assertIn('box failed ', stdout)
        self.assertNotIn('box done ', stdout)
        self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
