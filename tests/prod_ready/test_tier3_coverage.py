"""Offline CLI rehearsal of Tier 3 coverage staging (decision D2): bridge coverage manifest -> trained, qualified cohort."""
import collections
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from .fixture import Fixture, BASE, sha

READY = Path('data/brainstorm6/PROD-READY')
CANDIDATE = Path('data/brainstorm5/T3-REPAIR/candidate_v1/assets')
ROSTER = Path('data/challenge/track_2/tier_2_synthetic_noise/meta')
EP18 = '2026-05-06_10-00-19_blue_trash_can_drag_sideways_08'


class Tier3Coverage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='t3-coverage-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name)); self.addCleanup(self.f.stop_box)
        for name in ['make_run.py', 'verify_payload.py']:
            target = self.f.controller/READY/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BASE/READY/name, target)
        shutil.copytree(BASE/READY/'generated/arms', self.f.controller/READY/'generated/arms')
        shutil.copytree(BASE/CANDIDATE, self.f.controller/CANDIDATE)
        (self.f.controller/ROSTER).mkdir(parents=True)
        for name in ['episodes_metadata.jsonl', 'info.json']:
            shutil.copyfile(BASE/ROSTER/name, self.f.controller/ROSTER/name)
        roster = [json.loads(line) for line in (BASE/ROSTER/'episodes_metadata.jsonl').read_text().splitlines()]
        self.ep05 = next(r['sequence_id'] for r in roster if r['episode_index'] == 5)
        # A bridge output: ep05 materialized (candidate files under ep05's name) and its own ep18 variant.
        self.bridge = Path(self.temp.name)/'bridge'
        entries = [self.materialize(5, self.ep05, 'tier3_rest_repaired', 'HOLD'),
                   self.materialize(18, EP18, 'tier3_repaired', 'HOLD', shift=True)]
        self.coverage = self.bridge/'coverage_manifest.json'
        self.coverage.write_text(json.dumps({'schema_version': 1, 'tier': 3, 'scoring_mode': 'self', 'sequences': entries}))

    def materialize(self, episode, sequence, label, status, shift=False):
        root = self.bridge/f'ep{episode:02d}/assets'
        for path in (BASE/CANDIDATE).rglob('*'):
            if path.is_file():
                target = root/path.relative_to(BASE/CANDIDATE).as_posix().replace(EP18, sequence)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        reference = root/f'human_motion_data/whole_body/soma/sequence_id={sequence}/robot_name=g1/data.parquet'
        import pyarrow as pa, pyarrow.parquet as pq
        table = pq.read_table(reference)
        for column in ['object_mesh_paths', 'object_urdf_paths']:  # keep the copy self-consistent, as materialize does
            values = [[p.replace(EP18, sequence) for p in row] for row in table[column].to_pylist()]
            table = table.set_column(table.schema.get_field_index(column), column, pa.array(values, type=table.schema.field(column).type))
        pq.write_table(table, reference)
        if shift:  # the bridge's ep18 differs from the registered pilot candidate
            with reference.open('ab') as stream: stream.write(b'\0')
        record = {'path': str(reference), 'sha256': sha(reference)}
        return {'episode': episode, 'sequence': sequence, 'frames': 858, 'arms': {label: {
            'training_reference': record, 'clean_scoring_reference': record, 'support': {'enabled': False}}},
            'gate': {'status': status, 'reasons': ['fixture']}}

    def command(self, *args, ok=True):
        result = self.f.command([sys.executable, '-B', *map(str, args)])
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def jobs(self):
        lines = self.command(self.f.controller/'scripts/tier3_coverage.py', 'jobs', '--coverage', self.coverage).stdout
        path = Path(self.temp.name)/'jobs.txt'; path.write_text(lines)
        return path, lines.splitlines()

    def test_coverage_cohort_trains_every_episode_and_ep18_is_the_pilot_candidate(self):
        jobs, lines = self.jobs()
        arms = '../data/brainstorm6/PROD-READY/generated/arms/T2C-contact.txt'
        self.assertEqual(lines, [f't3-ep05|{arms}|T2C-contact|tier3_rest_repaired|{self.ep05}|49999872',
                                 f't3-ep18|{arms}|T2C-contact|tier3_support_static|{EP18}|49999872'])
        self.command(self.f.controller/READY/'make_run.py', '--tier', '3', '--coverage', self.coverage,
                     '--jobs', jobs, '--name', 't3cov')
        run = self.f.controller/READY/'runs/t3cov'
        self.assertEqual(self.command(self.f.controller/READY/'verify_payload.py', '--pilot', run).stdout.strip(),
            'COVERAGE PAYLOAD PASS: 2 episodes, 1 arm, roster, 1 binding; no clean/Tier 1 files')
        self.f.launch(run); self.f.finished(run)
        source = self.f.remote/'runs/t3cov'
        manifest = json.loads((source/'manifest.json').read_text())
        references = {s['sequence']: next(iter(s['arms'].values()))['training_reference']['sha256'] for s in manifest['sequences']}
        candidate = BASE/CANDIDATE/f'human_motion_data/whole_body/soma/sequence_id={EP18}/robot_name=g1/data.parquet'
        self.assertEqual(references[EP18], sha(candidate))
        events = self.f.calls()
        self.assertEqual(len([e for e in events if e['kind'] == 'train']), 2)
        self.assertEqual(collections.Counter(e['step'] for e in events if e['kind'] == 'evaluate'), {49999872: 2 + 6})  # one training-time evaluation per episode, then k=3 fresh
        selected = json.loads((source/'qualification/selected.json').read_text())
        self.assertEqual(set(selected), {'5', '18'})
        veto = json.loads((source/'qualification/veto.json').read_text())
        self.assertEqual(sorted(v['episode'] for v in veto['candidates']), [5, 18])
        self.assertIn('self-MPPE >10 cm', veto['rule'])

    def test_unlisted_job_changed_reference_and_extra_payload_file_are_refused(self):
        jobs, lines = self.jobs()
        forged = Path(self.temp.name)/'forged.txt'
        forged.write_text(lines[0].replace('tier3_rest_repaired', 'tier3_raw') + '\n')
        self.command(self.f.controller/READY/'make_run.py', '--tier', '3', '--coverage', self.coverage,
                     '--jobs', forged, '--name', 'forged', ok=False)
        self.assertFalse((self.f.controller/READY/'runs/forged/payload').exists())
        self.command(self.f.controller/READY/'make_run.py', '--tier', '3', '--coverage', self.coverage,
                     '--jobs', jobs, '--name', 'tampered')
        run = self.f.controller/READY/'runs/tampered'
        (run/'payload/runs/tampered/input-assets/extra.txt').write_text('x')
        self.assertIn('unexpected files', self.command(self.f.controller/READY/'verify_payload.py', '--pilot', run, ok=False).stderr)
        import pyarrow as pa, pyarrow.parquet as pq
        reference = next((self.bridge/'ep05').rglob('data.parquet'))
        table = pq.read_table(reference)
        values = [[p.replace(self.ep05, EP18) for p in row] for row in table['object_mesh_paths'].to_pylist()]
        pq.write_table(table.set_column(table.schema.get_field_index('object_mesh_paths'), 'object_mesh_paths',
                                        pa.array(values, type=table.schema.field('object_mesh_paths').type)), reference)
        entries = json.loads(self.coverage.read_text())
        entries['sequences'][0]['arms']['tier3_rest_repaired']['training_reference']['sha256'] = sha(reference)
        entries['sequences'][0]['arms']['tier3_rest_repaired']['clean_scoring_reference']['sha256'] = sha(reference)
        self.coverage.write_text(json.dumps(entries))
        self.command(self.f.controller/READY/'make_run.py', '--tier', '3', '--coverage', self.coverage,
                     '--jobs', jobs, '--name', 'outside', ok=False)
        with reference.open('ab') as stream: stream.write(b'\0')
        self.assertIn('reference changed', self.command(self.f.controller/'scripts/tier3_coverage.py', 'jobs',
                                                        '--coverage', self.coverage, ok=False).stderr)


if __name__ == '__main__':
    unittest.main()
