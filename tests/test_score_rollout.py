"""Public CLI checks with evaluator-schema fixtures; no challenge data or GPU runtime."""

from __future__ import annotations

import json
import hashlib
import shlex
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
sys.path.insert(0, str(ROOT / 'scripts'))
SCREEN = json.loads((ROOT / 'data/screen/tier2/manifest.json').read_text())
SEQUENCES = tuple(s['sequence'] for s in SCREEN['sequences'])


def write_rollout(path, poses, embedded=None, frames=None, valid=True):
    n = len(poses)
    frames = np.arange(n, dtype=np.int32) if frames is None else frames
    data = {name: pa.array(values, type=pa.int32()) for name, values in {
        'environment_id': np.zeros(n), 'step': frames, 'reference_frame': frames,
        'object_id': np.zeros(n), 'object_body_id': np.zeros(n)}.items()}
    data.update(time_s=pa.array(np.arange(n) / 50, type=pa.float64()),
                object_body_name=pa.array(['object'] * n).dictionary_encode(),
                environment_valid=pa.array([valid] * n))
    suffixes = ['position_x', 'position_y', 'position_z', 'quaternion_x',
                'quaternion_y', 'quaternion_z', 'quaternion_w']
    for prefix, values in [('achieved', poses), ('reference', embedded if embedded is not None else poses)]:
        for i, suffix in enumerate(suffixes):
            data[f'{prefix}_{suffix}'] = pa.array(values[:, i], type=pa.float32())
    table = pa.table(data)
    schema = pa.schema([pa.field(f.name, f.type, nullable=False) for f in table.schema], metadata={
        b'flash_chord.schema': b'flash_chord_object_trajectories_v1',
        b'flash_chord.position_unit': b'meter', b'flash_chord.quaternion_order': b'xyzw',
        b'flash_chord.row_order': b'step,environment_id,object_body_id',
        b'flash_chord.control_fps': b'50.0', b'flash_chord.start_frame': b'0',
        b'flash_chord.step_count': str(n).encode(), b'flash_chord.world_count': b'1',
        b'flash_chord.object_body_count': b'1', b'flash_chord.pose_width': b'7'})
    pq.write_table(table.cast(schema), path)


def write_reference(path, poses, mesh):
    pq.write_table(pa.Table.from_pylist([{
        'schema_version': 'motion_v1', 'object_body_names': ['object'],
        'object_mesh_paths': [str(mesh)], 'object_body_position': poses[:, None, :3].tolist(),
        'object_body_wxyz': poses[:, None, [6, 3, 4, 5]].tolist(),
    }]), path)


class ScoreRolloutTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.mesh = self.work / 'object.ply'
        trimesh.Trimesh(vertices=np.r_[np.eye(3), -np.eye(3)] * .1,
                        faces=[[0, 1, 2], [3, 4, 5]], process=False).export(self.mesh)
        self.identity = np.tile([0., 0., 0., 0., 0., 0., 1.], (10, 1))
        self.reference = self.work / 'clean.parquet'
        write_reference(self.reference, self.identity, self.mesh)

    def cli(self, pairs, *extra, success=True):
        output = self.work / 'scores.json'
        command = [sys.executable, str(ROOT / 'scripts/score_rollout.py'), '-o', str(output)]
        for rollout, reference in pairs:
            command += ['--rollout', str(rollout), '--reference', str(reference)]
        result = subprocess.run([*command, *extra], capture_output=True, text=True, cwd=ROOT)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('MPPE(cm)', result.stdout)
            self.assertIn('mean\t', result.stdout)
            return json.loads(output.read_text())
        self.assertNotEqual(result.returncode, 0)
        return result

    def test_identity_ignores_corrupted_embedded_reference(self):
        # Vary frame poses so dropping, shifting, or sorting by the wrong field cannot pass.
        poses = self.identity.copy().astype(np.float32)
        poses[:, :3] = np.arange(30).reshape(10, 3) / 32
        poses[1::2, 3:] = [0, 0, 1, 0]
        write_reference(self.reference, poses, self.mesh)
        rollout = self.work / 'perfect.parquet'
        write_rollout(rollout, poses, embedded=poses + 99)
        table = pq.ParquetFile(rollout).read()
        pq.write_table(table.take(pa.array(np.arange(9, -1, -1))), rollout)
        report = self.cli([(rollout, self.reference)])
        expected = {'AUC': 1., 'SP-SR': 1., 'MP-SR': 1., 'MPPE': 0.}
        self.assertEqual(report['episodes'][0]['metrics'], expected)
        self.assertEqual(report['mean'], expected)

    def test_twenty_cm_translation(self):
        poses = self.identity.copy()
        poses[:, 0] += .2
        rollout = self.work / 'offset.parquet'
        write_rollout(rollout, poses)
        metrics = self.cli([(rollout, self.reference)])['mean']
        self.assertEqual([metrics[m] for m in ['AUC', 'SP-SR', 'MP-SR']], [0., 0., 0.])
        # float32 evaluator poses cannot encode decimal 0.2 exactly; keep the kit's raw answer.
        self.assertAlmostEqual(metrics['MPPE'], 20., delta=1e-6)

    def test_halfturn_and_quaternion_sign(self):
        poses = self.identity.copy()
        poses[:, 3:] = [0, 0, 1, 0]
        rollout = self.work / 'halfturn.parquet'
        write_rollout(rollout, poses)
        metrics = self.cli([(rollout, self.reference)])['mean']
        self.assertEqual([metrics[m] for m in ['AUC', 'SP-SR', 'MP-SR']], [0., 0., 0.])
        self.assertAlmostEqual(metrics['MPPE'], 20 / 3, delta=1e-10)
        write_rollout(rollout, self.identity * [1, 1, 1, -1, -1, -1, -1])
        self.assertEqual(self.cli([(rollout, self.reference)])['mean'],
                         {'AUC': 1., 'SP-SR': 1., 'MP-SR': 1., 'MPPE': 0.})

    def test_equal_episode_weight_with_different_lengths(self):
        perfect = self.work / 'perfect.parquet'
        write_rollout(perfect, self.identity)
        short_ref = self.work / 'short.parquet'
        write_reference(short_ref, self.identity[:2], self.mesh)
        bad = self.work / 'bad.parquet'
        write_rollout(bad, self.identity[:2] + [.2, 0, 0, 0, 0, 0, 0])
        mean = self.cli([(perfect, self.reference), (bad, short_ref)])['mean']
        self.assertEqual([mean[m] for m in ['AUC', 'SP-SR', 'MP-SR']], [.5, .5, .5])
        self.assertAlmostEqual(mean['MPPE'], 10., delta=1e-6)

    def test_missing_duplicate_and_invalid_frames_fail(self):
        rollout = self.work / 'invalid.parquet'
        for poses, frames, valid in [
            (self.identity[:-1], None, True),
            (self.identity, np.zeros(10, dtype=np.int32), True),
            (self.identity, np.arange(1, 11, dtype=np.int32), True),
            (self.identity, None, False),
            (np.zeros_like(self.identity), None, True),
        ]:
            with self.subTest(frames=frames, valid=valid):
                write_rollout(rollout, poses, frames=frames, valid=valid)
                self.cli([(rollout, self.reference)], success=False)

    def test_reference_is_required_and_schema_units_are_checked(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/score_rollout.py'),
                                 '--rollout', 'unused', '-o', str(self.work / 'out.json')], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'--reference', result.stderr)
        rollout = self.work / 'units.parquet'
        write_rollout(rollout, self.identity)
        table = pq.ParquetFile(rollout).read()
        pq.write_table(table.replace_schema_metadata({**table.schema.metadata,
                       b'flash_chord.position_unit': b'centimeter'}), rollout)
        self.assertIn('meter', self.cli([(rollout, self.reference)], success=False).stderr)

    def test_three_fresh_clean_references_through_cli(self):
        pairs = []
        for sequence in SEQUENCES:
            reference = Path(next(s for s in SCREEN['sequences'] if s['sequence'] == sequence)['clean_scoring_reference']['path'])
            row = pq.ParquetFile(reference).read(columns=['object_body_position', 'object_body_wxyz']).to_pylist()[0]
            poses = np.c_[np.asarray(row['object_body_position'])[:, 0],
                          np.asarray(row['object_body_wxyz'])[:, 0, [1, 2, 3, 0]]]
            rollout = self.work / f'{sequence}.parquet'
            write_rollout(rollout, poses, embedded=poses + 99)
            pairs.append((rollout, reference))
        report = self.cli(pairs)
        self.assertEqual(len(report['episodes']), 3)
        self.assertEqual(report['mean'], {'AUC': 1., 'SP-SR': 1., 'MP-SR': 1., 'MPPE': 0.})


class PilotQueueTest(unittest.TestCase):
    def manifest(self, work):
        def record(path):
            return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        mesh = work / 'mesh.ply'
        trimesh.creation.box(extents=[.1, .1, .1]).export(mesh)
        sequences = []
        for sequence in SEQUENCES:
            inputs = {}
            for label, offset in [('clean', 0), ('raw_repaired', .2)]:
                path = work / label / 'motion/whole_body/soma' / f'sequence_id={sequence}/robot_name=g1/data.parquet'
                path.parent.mkdir(parents=True, exist_ok=True)
                poses = np.tile([offset, 0., 0., 0., 0., 0., 1.], (10, 1))
                write_reference(path, poses, mesh)
                support = work / label / 'motion/whole_body/reconstructed_stage' / f'{sequence}_support.usda'
                support.parent.mkdir(parents=True, exist_ok=True)
                support.write_text('#usda 1.0\ndef Xform "support_surfaces" {}\n')
                inputs[label] = dict(training_reference=record(path), support=dict(**record(support), enabled=False))
            clean = inputs['clean']['training_reference']
            for arm in inputs.values():
                arm['clean_scoring_reference'] = clean
            sequences.append(dict(sequence=sequence, clean_scoring_reference=clean, arms=inputs))
        path = work / 'manifest.json'
        path.write_text(json.dumps(dict(sequences=sequences)))
        return path

    def test_dry_run_all_arms_and_bad_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'untouched'
            manifest = self.manifest(Path(temp).resolve())
            command = [str(ROOT / 'scripts/pilot_queue.sh'), '--arms', str(ROOT / 'scripts/pilot_arms.txt'),
                       '--manifest', str(manifest), '--out-root', str(out), '--steps', '10000384', '49999872', '--dry-run']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(out.exists())
            self.assertEqual(result.stdout.count('scripts/train_queue.sh'), 27)
            self.assertEqual(result.stdout.count('scripts/evaluate_policy.py'), 54)
            self.assertEqual(result.stdout.count('scripts/score_rollout.py'), 54)
            result = subprocess.run([*command, '--steps', '50000000'], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requested steps', result.stderr)

    def test_manifest_train_eval_score_triples_and_noisy_scoring_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp).resolve()
            manifest = self.manifest(work)
            command = [str(ROOT / 'scripts/pilot_queue.sh'), '--arms', str(ROOT / 'scripts/pilot_arms.txt'),
                       '--manifest', str(manifest), '--arm', 'E1-control', '--arm', 'E4-orientation',
                       '--inputs', 'clean', 'raw_repaired', '--sequences', SEQUENCES[1],
                       '--out-root', str(work / 'runs'), '--steps', '49999872', '--dry-run']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertEqual(len(lines), 13)
            clean = json.loads(manifest.read_text())['sequences'][1]['clean_scoring_reference']['path']
            for i, (arm, label) in enumerate((a, l) for a in ['E1-control', 'E4-orientation']
                                            for l in ['clean', 'raw_repaired']):
                train, evaluate, score = lines[i * 3:i * 3 + 3]
                base = work / 'runs' / arm / label
                training = work / label / 'motion/whole_body/soma' / f'sequence_id={SEQUENCES[1]}/robot_name=g1/data.parquet'
                self.assertIn(shlex.quote(str(training)), train)
                self.assertIn(str(base / 'train' / SEQUENCES[1] / 'policy_49999872.safetensors'), evaluate)
                self.assertIn(str(base / 'evaluation' / SEQUENCES[1] / '49999872/trajectory.parquet'), score)
                self.assertIn(f'--reference {clean} --output', score)
            self.assertFalse((work / 'runs').exists())
            data = json.loads(manifest.read_text())
            entry = data['sequences'][1]
            entry['arms']['raw_repaired']['clean_scoring_reference'] = entry['arms']['raw_repaired']['training_reference']
            manifest.write_text(json.dumps(data))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('scoring reference', result.stderr)
            self.assertNotIn('scripts/train_queue.sh', result.stdout)
            no_manifest = command.copy()
            j = no_manifest.index('--manifest')
            del no_manifest[j:j+2]
            result = subprocess.run(no_manifest, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--manifest', result.stderr)

    def test_self_scoring_mode_scores_each_input_against_itself(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp).resolve()
            manifest = self.manifest(work)
            data = json.loads(manifest.read_text())
            data['scoring_mode'] = 'self'
            for entry in data['sequences']:
                for arm in entry['arms'].values():
                    arm['clean_scoring_reference'] = arm['training_reference']
            manifest.write_text(json.dumps(data))
            command = [str(ROOT / 'scripts/pilot_queue.sh'), '--arms', str(ROOT / 'scripts/pilot_arms.txt'),
                       '--manifest', str(manifest), '--arm', 'T2C-contact', '--inputs', 'raw_repaired',
                       '--sequences', SEQUENCES[1], '--out-root', str(work / 'runs'), '--steps', '49999872', '--dry-run']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            own = data['sequences'][1]['arms']['raw_repaired']['training_reference']['path']
            score = result.stdout.splitlines()[2]
            self.assertIn(f'--reference {own} --output', score)
            self.assertFalse((work / 'runs').exists())
            # Self mode still refuses an input that names any other scoring reference.
            data['sequences'][1]['arms']['raw_repaired']['clean_scoring_reference'] = data['sequences'][1]['arms']['clean']['training_reference']
            manifest.write_text(json.dumps(data))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('must name its own training reference', result.stderr)
            data['scoring_mode'] = 'selfish'
            manifest.write_text(json.dumps(data))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown scoring_mode 'selfish'", result.stderr)

    def fake_python(self, work):
        fake = work / 'fake_flash'
        fake.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(os.environ['PILOT_TEST_ROOT']) / 'tests'))
from test_score_rollout import write_rollout
if sys.argv[1] == '-':
    print('true')
    sys.exit(0)
args = dict(a.split('=', 1) for a in sys.argv[2:])
event = 'train' if sys.argv[1].endswith('train_flash_sac.py') else 'evaluate'
with open(os.environ['PILOT_TEST_EVENTS'], 'a') as f:
    f.write(event + '\\n')
if event == 'train':
    out = Path(args['output_dir'].strip("'"))
    (out / ('policy_' + args['training.total_environment_steps'] + '.safetensors')).write_text(args['task.parquet'].strip("'"))
else:
    ref = Path(args['evaluation.checkpoint']).read_text().strip()
    row = pq.ParquetFile(ref).read(columns=['object_body_position','object_body_wxyz']).to_pylist()[0]
    poses = np.c_[np.asarray(row['object_body_position'])[:,0], np.asarray(row['object_body_wxyz'])[:,0,[1,2,3,0]]]
    write_rollout(Path(args['evaluation.object_trajectories_output']), poses)
    if os.environ.get('PILOT_TEST_FAIL'):
        sys.exit(3)
    Path(args['evaluation.metrics_output']).write_text('{}')
''')
        fake.chmod(0o755)
        events = work / 'events'
        return events, {**os.environ, 'PYTHON': str(fake), 'PACK_PYTHON': sys.executable,
                        'PILOT_TEST_ROOT': str(ROOT), 'PILOT_TEST_EVENTS': str(events)}

    def test_evaluator_code_change_reevaluates_cached_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp).resolve()
            manifest = self.manifest(work)
            flash = work / 'flash'
            subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout',
                            str(ROOT / 'external/video_to_data'), str(flash)], check=True)
            (flash / 'objective.py').write_text('weight = 1\n')
            subprocess.run(['git', '-C', str(flash), 'add', 'objective.py'], check=True)
            events, env = self.fake_python(work)
            env['FLASH_ROOT'] = str(flash)
            command = [str(ROOT / 'scripts/pilot_queue.sh'), '--arms', str(ROOT / 'scripts/pilot_arms.txt'),
                       '--manifest', str(manifest), '--arm', 'E1-control', '--sequences', SEQUENCES[1],
                       '--out-root', str(work / 'runs'), '--steps', '49999872']
            for change in [None, None, lambda: (flash / 'objective.py').write_text('weight = 2\n'),
                           lambda: (flash / 'new_term.py').write_text('extra = 1\n')]:
                if change:
                    change()
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(events.read_text().splitlines(), ['train', 'evaluate', 'evaluate', 'evaluate'])

    def test_partial_evaluation_retries_then_skips_and_recipe_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp).resolve()
            manifest = self.manifest(work)
            events, env = self.fake_python(work)
            command = [str(ROOT / 'scripts/pilot_queue.sh'), '--arms', str(ROOT / 'scripts/pilot_arms.txt'),
                       '--manifest', str(manifest), '--arm', 'E1-control', '--sequences', SEQUENCES[1],
                       '--out-root', str(work / 'runs'), '--steps', '49999872']
            failed = subprocess.run(command, env={**env, 'PILOT_TEST_FAIL': '1'}, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            for attempt in range(2):
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(events.read_text().splitlines(), ['train', 'evaluate', 'evaluate'])
            report = json.loads((work / 'runs/summary.json').read_text())
            self.assertEqual(report['candidates'], [dict(arm='E1-control', input='clean', sequence=SEQUENCES[1], step=49999872,
                              **{'AUC': 1., 'SP-SR': 1., 'MP-SR': 1., 'MPPE': 0.})])
            checkpoint = next((work / 'runs').rglob('policy_*.safetensors'))
            checkpoint.write_text(checkpoint.read_text() + '\n')
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(events.read_text().splitlines(), ['train', 'evaluate', 'evaluate', 'evaluate'])
            arms = work / 'changed.txt'
            arms.write_text((ROOT / 'scripts/pilot_arms.txt').read_text().replace('49999872', '50001920'))
            changed = command.copy()
            changed[changed.index('--arms') + 1] = str(arms)
            changed[changed.index('--steps') + 1] = '50001920'
            result = subprocess.run(changed, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('fresh --out-root', result.stderr)
            self.assertEqual(events.read_text().splitlines(), ['train', 'evaluate', 'evaluate', 'evaluate'])
            result = subprocess.run([*command, '--inputs', 'raw_repaired'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((work / 'runs/summary.json').read_text())
            self.assertEqual(report['candidates'][0]['input'], 'raw_repaired')
            self.assertAlmostEqual(report['candidates'][0]['MPPE'], 20., delta=1e-6)
            self.assertEqual(len(list((work / 'runs').rglob('policy_*.safetensors'))), 2)
            data = json.loads(manifest.read_text())
            support = data['sequences'][1]['arms']['clean']['support']
            Path(support['path']).write_text('#usda 1.0\ndef Xform "changed" {}\n')
            support['sha256'] = hashlib.sha256(Path(support['path']).read_bytes()).hexdigest()
            manifest.write_text(json.dumps(data))
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('support changed', result.stderr)
            self.assertEqual(events.read_text().splitlines(), ['train', 'evaluate', 'evaluate', 'evaluate', 'train', 'evaluate'])


if __name__ == '__main__':
    unittest.main()
