"""Drive the export shell and real kit coordinator with a fake GPU evaluator."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class ExportCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='INTEGRATE-export-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name in ('scripts', 'patches'):
            shutil.copytree(PROJECT / name, self.root / name, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(PROJECT / 'external/kit/v2d_submission_kit', self.root / 'external/kit/v2d_submission_kit',
                        ignore=shutil.ignore_patterns('__pycache__'))
        (self.root / 'external/video_to_data/robotic_grounding/flash_chord').mkdir(parents=True)
        for tier in ('tier_1_multiview_caption', 'tier_2_synthetic_noise'):
            meta = self.root / 'data/challenge/track_2' / tier / 'meta'
            meta.mkdir(parents=True)
            (meta / 'info.json').write_text(json.dumps({'total_episodes': 2}))
            (meta / 'episodes_metadata.jsonl').write_text('\n'.join(json.dumps(
                {'episode_index': i, 'sequence_id': f'seq{i}'}) for i in range(2)) + '\n')
        self.evaluator = self.root / 'fake_evaluator.py'
        self.evaluator.write_text('''import hashlib,json,sys
from pathlib import Path
args=dict(a.split('=',1) for a in sys.argv[1:])
checkpoint=Path(args['evaluation.checkpoint'])
Path(args['evaluation.metrics_output']).write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))
''')
        self.python = self.root / 'python_boundary'
        self.python.write_text('#!' + sys.executable + '\n' + '''import os,runpy,sys
from pathlib import Path
args=sys.argv[1:]
# Packing and validation are outside this test; the GPU boundary remains a genuine subprocess.
if Path(args[0]).name in ('pack_submission.py','pack_no_clean.py','frame_references.py','validate_offline.py'): sys.exit(0)
if Path(args[0]).name=='run_policy_evaluation.py':
    sys.argv=args+['--eval-script',os.environ['FAKE_EVALUATOR']]
    runpy.run_path(args[0],run_name='__main__')
else: os.execv(sys.executable,[sys.executable,*args])
''')
        self.python.chmod(0o755)
        self.env = {**os.environ, 'PYTHON': str(self.python), 'PACK_PYTHON': str(self.python),
                    'FAKE_EVALUATOR': str(self.evaluator)}
        self.env.pop('POLICY_STEPS', None)
        self.env.pop('CHECKPOINT_MAP', None)
        self.paths = []
        for episode, step in ((0, 330012672), (1, 349999104)):
            path = self.root / f'selected/seq{episode}/policy_{step}.safetensors'
            path.parent.mkdir(parents=True)
            path.write_bytes(f'selected episode {episode} step {step}'.encode())
            self.paths.append(path)
            for default in (349999104, 49999872):
                decoy = self.root / f'runs/seq{episode}/policy_{default}.safetensors'
                decoy.parent.mkdir(parents=True, exist_ok=True)
                decoy.write_bytes(f'default episode {episode} step {default}'.encode())

    def export(self, tier='tier1', mapping=None, name='out'):
        env = dict(self.env)
        if mapping is not None:
            path = self.root / 'selected.json'
            path.write_text(json.dumps(mapping))
            env['CHECKPOINT_MAP'] = str(path)
        out = self.root / name
        result = subprocess.run(['bash', str(self.root / 'scripts/eval_and_pack.sh'), tier,
                                 str(self.root / 'runs'), str(out),
                                 'https://github.com/fixture/fixture/commit/0123456789abcdef',
                                 str(self.root / 'frozen')], env=env, capture_output=True, text=True)
        return result, out

    def assert_selected(self, mapping, name):
        result, out = self.export(mapping=mapping, name=name)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = {str(i): str(path) for i, path in enumerate(self.paths)}
        evaluated = {str(i): json.loads((out / f'evaluation/episode_{i:06d}.json').read_text())['checkpoint_sha256']
                     for i in range(2)}
        self.assertEqual(evaluated, {str(i): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for i, path in enumerate(self.paths)})
        self.assertEqual(json.loads((out / 'checkpoints.json').read_text()), expected)

    def test_decimal_map_evaluates_exact_selected_checkpoint_digests(self):
        self.assert_selected({str(i): str(p) for i, p in enumerate(self.paths)}, 'decimal')

    def test_tournament_map_converts_episode_keys_and_evaluates_exact_digests(self):
        self.assert_selected({'checkpoint_map': {f'episode_{i:06d}': str(p) for i, p in enumerate(self.paths)}}, 'tournament')

    def test_invalid_maps_fail_before_evaluation(self):
        for index, mapping in enumerate(({'0': str(self.paths[0])},
                {'0': str(self.paths[0]), 'episode_000000': str(self.paths[1])},
                {'0': str(self.paths[0]), '30': str(self.paths[1])},
                {'0': str(self.paths[0]), '1': str(self.root / 'missing.safetensors')})):
            with self.subTest(mapping=mapping):
                result, out = self.export(mapping=mapping, name=f'bad{index}')
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse((out / 'evaluation').exists())

    def test_hard_tiers_evaluate_default_49999872_checkpoints(self):
        for tier in ('tier2', 'tier3'):
            result, out = self.export(tier=tier, name=tier)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            got = {str(i): json.loads((out / f'evaluation/episode_{i:06d}.json').read_text())['checkpoint_sha256']
                   for i in range(2)}
            self.assertEqual(got, {str(i): hashlib.sha256(f'default episode {i} step 49999872'.encode()).hexdigest()
                                   for i in range(2)})
