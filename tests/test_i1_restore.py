"""I1 replacement input-custody regression, using real stage/queue CLIs and fake CUDA.

Native screen regeneration is a boundary fixture: build B differs only in valid parquet metadata.
The test does not claim a measured rate of native IK nondeterminism.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(os.environ.setdefault('BOX_REPO', str(Path(__file__).resolve().parents[1]))).resolve()
sys.path.insert(0, str(ROOT / 'tests'))
import test_score_rollout as base

BUILDER = '''import hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['BOX_REPO']+'/tests')
from test_score_rollout import PilotQueueTest
import pyarrow.parquet as pq
root=Path('data/screen/tier2');root.mkdir(parents=True,exist_ok=True)
path=PilotQueueTest().manifest(root.resolve())
manifest=json.loads(path.read_text())
for entry in manifest['sequences']:
 for arm in entry['arms'].values():
  record=arm['training_reference'];p=Path(record['path'])
  table=pq.read_table(p);pq.write_table(table.replace_schema_metadata({b'fixture_build':os.environ['SCREEN_BUILD_ID'].encode()}),p)
  record['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 # Rebind every clean-scoring record after rewriting clean parquet metadata.
 clean=entry['arms']['clean']['training_reference']
 entry['clean_scoring_reference']=clean
 for arm in entry['arms'].values():arm['clean_scoring_reference']=clean
path.write_text(json.dumps(manifest))
'''


class I1Restore(unittest.TestCase):
    def test_replacement_uses_first_screen_bytes_before_resuming_the_same_recipe(self):
        with tempfile.TemporaryDirectory(prefix='i1-', dir=os.environ.get('TMPDIR', '/tmp')) as temporary:
            work = Path(temporary).resolve()
            stage = work / 'staged'
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'data/brainstorm7/stage_i1.py'), str(stage)],
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            staged = stage / 'i1screen'
            box = work / 'project'

            def provision():
                shutil.copytree(ROOT / 'scripts', box / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
                (box / 'scripts/make_tier2_screen.py').write_text(BUILDER)
                shutil.copytree(ROOT / 'external/kit', box / 'external/kit')
                (box / '.venv/bin').mkdir(parents=True)
                interpreter = box / '.venv/bin/python'
                interpreter.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' \"$@\"\n')
                interpreter.chmod(0o755)
                shutil.copytree(staged, box / 'runs/i1screen')

            def boot(build):
                # Execute box.sh's env -> optional screen build -> sourced pre hook prefix.
                command = '''set -euo pipefail
[ ! -f runs/i1screen/env.sh ] || source runs/i1screen/env.sh
if [ "${BUILD_SCREEN:-1}" = 1 ]; then .venv/bin/python scripts/make_tier2_screen.py --jobs 6; fi
[ ! -f runs/i1screen/pre.sh ] || source runs/i1screen/pre.sh
'''
                return subprocess.run(['bash', '-c', command], cwd=box,
                    env={**os.environ, 'SCREEN_BUILD_ID': build}, text=True, capture_output=True)

            provision()
            result = boot('first')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = box / 'data/screen/tier2/manifest.json'
            original = manifest.read_bytes()
            first_line = (staged / 'jobs.txt').read_text().splitlines()[0]
            name, af, arm, label, sequence, steps = first_line.split('|')
            helper = base.PilotQueueTest()
            events, env = helper.fake_python(work)
            # Native runtime is not invoked. These are the existing suite's fake train/evaluate entry points.
            env.update(FLASH_ROOT=str(ROOT / 'external/video_to_data/robotic_grounding/flash_chord'))
            pilot_command = [str(box / 'scripts/pilot_queue.sh'), '--arms', str(box / 'scripts' / af),
                '--arm', arm, '--manifest', str(manifest), '--inputs', label, '--sequences', sequence,
                '--steps', steps, '--out-root', str(box / 'runs/i1screen' / name)]
            result = subprocess.run(pilot_command, cwd=box, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(events.read_text().split(), ['train', 'evaluate'])
            # launch.sh restores only the run tree. Keep the exact same /project path, as on real replacements.
            old = work / 'retired-first'
            box.rename(old)
            provision()
            shutil.copytree(old / 'runs/i1screen', box / 'runs/i1screen', dirs_exist_ok=True)
            result = boot('replacement')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run(['bash', str(box / 'scripts/test_day.sh'), '1', 'runs/i1screen', '--dry-run'],
                cwd=box, env={**env, 'MANIFEST': str(manifest), 'JOBS_FILE': str(staged / 'jobs.txt')},
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(manifest.read_bytes(), original)
            result = subprocess.run(pilot_command, cwd=box, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('# skip training:', result.stdout)
            self.assertIn('# skip evaluation:', result.stdout)
            self.assertEqual(events.read_text().split(), ['train', 'evaluate'])


if __name__ == '__main__':
    unittest.main()
