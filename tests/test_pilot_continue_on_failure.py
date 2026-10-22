"""Proposed test for pilot_queue_continue_on_failure.patch: one failed training must not cancel the
arm's other sequences. Drop into tests/ next to test_score_rollout.py."""

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_score_rollout as base
from test_score_rollout import ROOT, SEQUENCES


class ContinueOnFailureTest(unittest.TestCase):
    # Reuse the existing fixtures without re-running every inherited test.
    manifest = base.PilotQueueTest.manifest
    fake_python = base.PilotQueueTest.fake_python

    def test_failed_training_still_runs_and_summarises_other_sequences(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp).resolve()
            manifest = self.manifest(work)
            events, env = self.fake_python(work)
            # Wrap the shared fake so that only the first sequence's training crashes.
            flaky = work / 'flaky_flash'
            flaky.write_text(f'#!/bin/sh\ncase "$*" in *train_flash_sac.py*{SEQUENCES[0]}*) exit 3;; esac\n'
                             f'exec {env["PYTHON"]} "$@"\n')
            flaky.chmod(0o755)
            env['PYTHON'] = str(flaky)
            command = [str(ROOT / 'scripts/pilot_queue.sh'), '--arms', str(ROOT / 'scripts/pilot_arms.txt'),
                       '--manifest', str(manifest), '--arm', 'E1-control',
                       '--out-root', str(work / 'runs'), '--steps', '49999872']
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(SEQUENCES[0], result.stderr)
            self.assertEqual(events.read_text().split(), ['train', 'evaluate', 'train', 'evaluate'])
            with (work / 'runs/summary.tsv').open() as f:
                rows = list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual([r['sequence'] for r in rows], list(SEQUENCES[1:]))


if __name__ == '__main__':
    unittest.main()
