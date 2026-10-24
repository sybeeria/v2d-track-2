"""Public CLI regressions for Tier 3 admission and multi-host release; no GPU/provider calls.

Copy to tests/ in an isolated checkout. Metadata is real; reference/policy bytes are fixtures.
The existing prod_ready tests cover training/evaluation/qualification with fake CUDA boundaries.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / 'data/brainstorm6/PROD-READY'
sys.path.insert(0, str(ROOT / 'scripts'))
from ep18_pilot import PAIR, SEQUENCE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class Tier3LaunchGaps(unittest.TestCase):
    def setUp(self):
        (ROOT / 'data/test-tmp').mkdir(parents=True, exist_ok=True)  # under ROOT: staging uses ROOT-relative paths
        self.temp = tempfile.TemporaryDirectory(prefix='b8-launch-', dir=ROOT / 'data/test-tmp')
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.tag = self.work.name.replace('-', '')
        self.roster = [json.loads(line) for line in
                       (ROOT / 'data/challenge/track_2/tier_2_synthetic_noise/meta/episodes_metadata.jsonl').read_text().splitlines()]
        self.assertEqual(len(self.roster), 30)
        self.coverage = self.work / 'coverage.json'
        entries = []
        for row in self.roster:
            ep, seq, frames = row['episode_index'], row['sequence_id'], row['frames']
            assets = self.work / f'bridge/ep{ep:02d}/assets'
            prefix = f'human_motion_data/whole_body/soma/sequence_id={seq}'
            mesh = assets / prefix / 'object/textured_mesh.obj'
            mesh.parent.mkdir(parents=True)
            mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
            urdf = mesh.with_suffix('.urdf'); urdf.write_text('<robot name="fixture"/>')
            ref = assets / prefix / 'robot_name=g1/data.parquet'; ref.parent.mkdir()
            pq.write_table(pa.Table.from_pylist([{
                'schema_version': 'motion_v1', 'object_body_names': ['object'],
                'object_mesh_paths': ['/assets/' + str(mesh.relative_to(assets))],
                'object_urdf_paths': ['/assets/' + str(urdf.relative_to(assets))],
                'object_body_position': [[[0., 0., 0.]]] * frames,
                'object_body_wxyz': [[[1., 0., 0., 0.]]] * frames,
            }]), ref)
            support = assets / f'human_motion_data/whole_body/reconstructed_stage/{seq}_support.usda'
            support.parent.mkdir(parents=True, exist_ok=True); support.write_text('#usda 1.0\ndef Xform "support_surfaces" {}\n')
            record = {'path': str(ref), 'sha256': sha(ref)}
            entries.append({'episode': ep, 'sequence': seq, 'frames': frames, 'arms': {'tier3_repaired': {
                'training_reference': record, 'clean_scoring_reference': record,
                'support': {'path': str(support), 'sha256': sha(support), 'enabled': False}}},
                'gate': {'status': 'HOLD', 'reasons': ['fixture; no quality claim']}})
        self.content = {'schema_version': 1, 'tier': 3, 'scoring_mode': 'self', 'sequences': entries}
        put(self.coverage, self.content)
        # Never the real READY/runs/ep18-pilot-d1 pull: tonight's verdict lives there.
        self.pilot_root = self.work / 'pulled/ep18-pilot-d1'
        self.decision('retain 50M')

    def decision(self, value):
        passed = {'ep18-qualified50': value == 'retain 50M',
                  'ep18-qualified350': value in ('retain 50M', 'selective rescue')}
        q = self.pilot_root / 'qualification/qualification.json'
        put(q, {'candidates': [{'name': name, 'go': ok, 'members': [{'go': ok}] * 3}
                               for name, ok in passed.items()]})
        result = self.call('scripts/ep18_pilot.py', 'decide', '--qualification', q,
                           '--output', self.pilot_root / 'decision.json', unlink=self.pilot_root / 'decision.json')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def call(self, script, *args, unlink=None):
        if unlink is not None: unlink.unlink(missing_ok=True)
        return subprocess.run([sys.executable, '-B', str(ROOT / script), *map(str, args)], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)

    def jobs(self):
        return self.call('scripts/tier3_coverage.py', 'jobs', '--coverage', self.coverage,
                         '--pilot-decision', self.pilot_root / 'decision.json')

    def stage(self, lines, suffix):
        jobs = self.work / f'{suffix}.txt'; jobs.write_text('\n'.join(lines) + '\n')
        name = self.tag + suffix
        controller = READY / 'runs' / name
        host = ROOT / 'runs' / name
        self.addCleanup(shutil.rmtree, controller, True)
        self.addCleanup(shutil.rmtree, host, True)
        result = self.call('data/brainstorm6/PROD-READY/make_run.py', '--tier', 3,
                           '--coverage', self.coverage, '--jobs', jobs, '--name', name,
                           '--pilot-decision', self.pilot_root / 'decision.json')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.call('data/brainstorm6/PROD-READY/verify_payload.py', '--pilot', controller)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        shutil.copytree(controller / 'payload', ROOT, dirs_exist_ok=True)
        result = subprocess.run(['bash', str(controller / 'pre.sh')], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return controller, host

    def test_all30_shards_merge_and_changed_input_identity_is_rejected(self):
        result = self.jobs(); self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines(); self.assertEqual(len(lines), 30)
        cohorts = []; expected = {}
        for index in range(5):
            _, host = self.stage(lines[index::5], f'h{index}')
            manifest = host / 'manifest.json'
            cohort = host / 'qualification'; cohort.mkdir()
            mapping = {}; candidates = []; vetoes = []
            for line in lines[index::5]:
                name, _, arm, label, seq, final = line.split('|')
                ep = next(r['episode_index'] for r in self.roster if r['sequence_id'] == seq)
                checkpoint = host / name / arm / label / 'train' / seq / f'policy_{final}.safetensors'
                checkpoint.parent.mkdir(parents=True); checkpoint.write_text('fixture-policy-' + str(ep))
                mapping[str(ep)] = str(checkpoint)
                candidates.append({'episode': ep, 'sequence': seq, 'input': label})
                vetoes.append({'episode': ep, 'status': 'pass', 'checkpoint_sha256': sha(checkpoint)})
            expected.update(mapping)
            put(cohort / 'plan.json', {'tier': 3, 'manifest': str(manifest), 'manifest_sha256': sha(manifest),
                'repeats': 3, 'evaluator': 'fixed-runtime-fixture', 'ep18': False, 'source_root': str(host),
                'candidates': candidates})
            put(cohort / 'selected.json', mapping)
            put(cohort / 'veto.json', {'candidates': vetoes})
            put(cohort / 'export-binding.json', {'selected_sha256': sha(cohort / 'selected.json'),
                'veto_sha256': sha(cohort / 'veto.json'), 'checkpoints': {p: sha(Path(p)) for p in mapping.values()}})
            cohorts.append(cohort)
        out = self.work / 'release'
        args = [arg for c in cohorts for arg in ('--cohort', c)]
        result = self.call('scripts/production_repeats.py', 'merge', *args,
                           '--tier-root', ROOT / 'data/challenge/track_2/tier_2_synthetic_noise', '--out', out)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((out / 'selected.json').read_text()), expected)
        plan = json.loads((out / 'plan.json').read_text())
        manifest = Path(plan['manifest'])
        self.assertEqual(sha(manifest), plan['manifest_sha256'])
        self.assertEqual({s['sequence'] for s in json.loads(manifest.read_text())['sequences']},
                         {r['sequence_id'] for r in self.roster})
        # A different declared reconstruction must still fail, even with disjoint episode keys.
        changed = Path(json.loads((cohorts[-1] / 'plan.json').read_text())['manifest'])
        content = json.loads(changed.read_text()); content['tier3_coverage']['coverage_sha256'] = '0' * 64
        put(changed, content)
        plan = json.loads((cohorts[-1] / 'plan.json').read_text()); plan['manifest_sha256'] = sha(changed)
        put(cohorts[-1] / 'plan.json', plan)
        result = self.call('scripts/production_repeats.py', 'merge', *args,
                           '--tier-root', ROOT / 'data/challenge/track_2/tier_2_synthetic_noise', '--out', self.work / 'bad-release')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('protocol/input/runtime differs', result.stderr)
        self.assertFalse((self.work / 'bad-release').exists())

    def test_missing_non18_episode_is_refused_before_any_jobs(self):
        self.content['sequences'] = [r for r in self.content['sequences'] if r['episode'] != 7]
        put(self.coverage, self.content)
        result = self.jobs()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, '')
        self.assertIn('missing', result.stderr)

    def test_fail_or_wrong_roster_entry_is_never_covered(self):
        for change in ('FAIL', 'frames'):
            row = self.content['sequences'][1]
            if change == 'FAIL': row['gate']['status'] = 'FAIL'
            else: row['gate']['status'] = 'HOLD'; row['frames'] -= 1
            put(self.coverage, self.content)
            result = self.jobs()
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout, '')

    def test_missing_pilot_verdict_is_refused_before_any_jobs(self):
        (self.pilot_root / 'decision.json').unlink()
        result = self.jobs()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, '')

    def test_diagnose_retains_50M_coverage_under_D2(self):
        self.decision('diagnose')
        result = self.jobs(); self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 30)
        self.assertTrue(all(line.endswith('|49999872') for line in result.stdout.splitlines()))

    def test_missing_bridge_ep18_uses_the_registered_candidate(self):
        self.content['sequences'] = [r for r in self.content['sequences'] if r['episode'] != 18]
        put(self.coverage, self.content)
        result = self.jobs(); self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 30)
        self.assertEqual([line.split('|')[3] for line in result.stdout.splitlines() if line.startswith('t3-ep18|')],
                         ['tier3_support_static'])

    def test_selective_rescue_uses_registered_350M_only_for_episode18(self):
        self.decision('selective rescue')
        result = self.jobs(); self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines(); self.assertEqual(len(lines), 30)
        desired = f't3-ep18|../data/brainstorm6/PROD-READY/generated/arms/T3P-contact350.txt|T3P-contact350|tier3_support_static|{SEQUENCE}|349999104'
        self.assertEqual([line for line in lines if line.startswith('t3-ep18|')], [desired])
        self.assertTrue(all(line.endswith('|49999872') for line in lines if not line.startswith('t3-ep18|')))
        controller, host = self.stage([desired], 'rescue')
        name, _, arm, label, seq, final = desired.split('|')
        source = host / name / arm
        put(source / 'recipe.json', {})
        checkpoint = source / label / 'train' / seq / f'policy_{final}.safetensors'
        checkpoint.parent.mkdir(parents=True); checkpoint.write_text('fixture-rescue-policy')
        put(source / label / 'evaluation' / seq / final / 'evaluation.done.json', {'checkpoint_sha256': sha(checkpoint)})
        result = self.call('scripts/production_repeats.py', 'prepare', '--jobs', controller / 'jobs.txt',
                           '--manifest', host / 'manifest.json', '--source', host, '--tier', 3, '--k', 3,
                           '--tier-root', host / 'input-assets/roster', '--out', host / 'qualification')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        candidates = json.loads((host / 'qualification/plan.json').read_text())['candidates']
        self.assertEqual([(c['episode'], c['final_step']) for c in candidates], [(18, 349999104)])
        staged_arm = controller / 'payload/data/brainstorm6/PROD-READY/generated/arms/T3P-contact350.txt'
        staged_arm.write_text(staged_arm.read_text() + '\n# changed after staging\n')
        result = self.call('data/brainstorm6/PROD-READY/verify_payload.py', '--pilot', controller)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('arm SHA-256 mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
