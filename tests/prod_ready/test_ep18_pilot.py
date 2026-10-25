"""Offline CLI rehearsal of the registered ep18 production payload and frozen decisions."""
import collections
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from .fixture import Fixture, BASE, put, sha

READY = Path('data/brainstorm6/PROD-READY')
CANDIDATE = Path('data/brainstorm5/T3-REPAIR/candidate_v1/assets')


class Ep18Pilot(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ep18-pilot-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name)); self.addCleanup(self.f.stop_box)
        for name in ['make_run.py', 'verify_payload.py']:
            target = self.f.controller/READY/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BASE/READY/name, target)
        shutil.copytree(BASE/READY/'generated/arms', self.f.controller/READY/'generated/arms')
        tier3 = self.f.controller/READY/'generated/tier3'; tier3.mkdir()
        shutil.copyfile(BASE/READY/'generated/tier3/jobs.GATED.txt', tier3/'jobs.GATED.txt')
        shutil.copytree(BASE/CANDIDATE, self.f.controller/CANDIDATE)

    def stage(self, name, extra=(), ok=True, jobs=None):
        result = self.f.command([sys.executable, '-B', str(self.f.controller/READY/'make_run.py'),
            '--tier', '3', '--name', name, '--jobs', str(jobs or self.f.controller/READY/'generated/tier3/jobs.GATED.txt'), *extra])
        self.assertEqual(result.returncode == 0, ok, result.stdout+result.stderr)
        return self.f.controller/READY/'runs'/name, result

    def verify(self, run, ok=True):
        result = self.f.command([sys.executable, '-B', str(self.f.controller/READY/'verify_payload.py'), '--pilot', str(run)])
        self.assertEqual(result.returncode == 0, ok, result.stdout+result.stderr)
        return result

    def test_passing_pair_and_missing_member_fail_closed(self):
        run, _ = self.stage('ep18-pass')
        self.assertEqual(self.verify(run).stdout.strip(),
            'PILOT PAYLOAD PASS: 6 candidate assets, 2 arms, 1 binding; no clean/Tier 1 files')
        gate = json.loads((self.f.controller/READY/'generated/tier3/GATE.json').read_text())
        self.assertEqual((gate['state'],gate['run'],gate['jobs'],gate['k']),
            ('PILOT_STAGED',str(READY/'runs/ep18-pass'),2,3))
        self.f.launch(run); self.f.finished(run)
        source = self.f.remote/'runs'/run.name
        qualification = source/'qualification/qualification.json'
        decision = json.loads((source/'decision.json').read_text())
        self.assertEqual(decision, {'decision':'retain 50M',
            'passes':{'ep18-qualified350':True,'ep18-qualified50':True},
            'broad_tier3_policy_spend':False,'qualification_sha256':sha(qualification)})
        self.assert_pair(source, True)
        no_export=self.f.command([sys.executable,'-B',str(self.f.remote/'scripts/production_repeats.py'),
            'export','--out',str(source/'qualification'),'--work',str(source),'--commit','https://github.com/fixture/fixture/commit/0123456789abcdef'])
        self.assertEqual(no_export.returncode,1)
        self.assertIn('ep18 pilot has no submission export',no_export.stderr)
        # A retained, incomplete copy must fail mechanically, without another GPU process.
        broken = self.f.work/'missing-cohort'
        shutil.copytree(source/'qualification',broken,symlinks=True)
        (broken/'qualification.json').unlink()
        missing = next(broken.glob('cohort/ep18-qualified50-r02/**/trajectory.parquet')); missing.unlink()
        result = self.f.command([sys.executable,'-B',str(self.f.remote/'scripts/production_repeats.py'),
            'qualify','--out',str(broken)])
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        rows = {r['name']:r for r in json.loads((broken/'qualification.json').read_text())['candidates']}
        self.assertFalse(rows['ep18-qualified50']['go']); self.assertTrue(rows['ep18-qualified350']['go'])
        self.assertFalse(rows['ep18-qualified50']['members'][1]['go'])
        self.assertIn('error',rows['ep18-qualified50']['members'][1])
        self.assertEqual(len([e for e in self.f.calls() if e['kind']=='evaluate']),6)
        # Isolate the other frozen gates on retained fake outputs, resealing only the fixture digest.
        import pyarrow as pa
        import pyarrow.parquet as pq
        for fault in ('motion','position450','invalid'):
            target=self.f.work/f'{fault}-cohort'
            shutil.copytree(source/'qualification',target,symlinks=True)
            (target/'qualification.json').unlink()
            trajectory=next(target.glob('cohort/ep18-qualified50-r02/**/trajectory.parquet'))
            table=pq.ParquetFile(trajectory).read()
            column={'motion':'achieved_position_x','position450':'achieved_position_y',
                    'invalid':'environment_valid'}[fault]
            values=table[column].to_pylist()
            if fault=='motion':values[300]=values[450]-.59
            elif fault=='position450':values[450]+=.21
            else:values[450]=False
            table=table.set_column(table.schema.get_field_index(column),column,
                pa.array(values,type=table.schema.field(column).type))
            pq.write_table(table,trajectory)
            seal=trajectory.with_name('evaluation.done.json')
            identity=json.loads(seal.read_text());identity['trajectory_sha256']=sha(trajectory)
            seal.write_text(json.dumps(identity))
            result=self.f.command([sys.executable,'-B',str(self.f.remote/'scripts/production_repeats.py'),
                'qualify','--out',str(target)],timeout=90)
            self.assertEqual(result.returncode,1,result.stdout+result.stderr)
            rows={r['name']:r for r in json.loads((target/'qualification.json').read_text())['candidates']}
            self.assertTrue(rows['ep18-qualified350']['go']); self.assertFalse(rows['ep18-qualified50']['go'])
            members=rows['ep18-qualified50']['members']
            self.assertEqual([m['go'] for m in members],[True,False,True])
            if fault=='invalid':self.assertIn('environment_valid=false',members[1]['error'])
            else:
                self.assertLessEqual(members[1]['mppe_cm'],10)
                self.assertFalse(members[1]['early_drop'])
                if fault=='motion':
                    self.assertAlmostEqual(members[1]['x300_to450_m'],.59)
                    self.assertLessEqual(members[1]['position_error450_m'],.2)
                else:
                    self.assertGreater(members[1]['position_error450_m'],.2)
                    self.assertGreaterEqual(abs(members[1]['x300_to450_m']),.6)
        print('EP18 FALSIFIERS PASS: 0.59m displacement, >0.20m frame450 error, invalid frame each vetoes one of three')
        print('EP18 REHEARSAL PASS: retain 50M; 2 trainers; 6 distinct evaluator processes; missing member fails')

    def assert_pair(self, source, passes50):
        events = self.f.calls(); evaluations = [e for e in events if e['kind']=='evaluate']
        self.assertEqual(len([e for e in events if e['kind']=='train']),2)
        self.assertEqual(collections.Counter(e['step'] for e in evaluations),{49999872:3,349999104:3})
        self.assertEqual(len({e['pid'] for e in evaluations}),6)
        rows = {r['name']:r for r in json.loads((source/'qualification/qualification.json').read_text())['candidates']}
        self.assertEqual({k:v['go'] for k,v in rows.items()},
            {'ep18-qualified350':True,'ep18-qualified50':passes50})
        self.assertEqual([len(r['members']) for r in rows.values()],[3,3])
        self.assertTrue(all(m['frames']==858 for r in rows.values() for m in r['members']))
        self.assertFalse((source/'qualification/selected.json').exists())
        self.assertFalse((source/'qualification/export-binding.json').exists())

    def test_carry_veto_selective_rescue_despite_low_mppe(self):
        run, _ = self.stage('ep18-rescue')
        self.f.launch(run,{'GPU_EP18_DROP50':'1'}); self.f.finished(run)
        source=self.f.remote/'runs'/run.name
        q=source/'qualification/qualification.json'
        self.assertEqual(json.loads((source/'decision.json').read_text()),{
            'decision':'selective rescue','passes':{'ep18-qualified350':True,'ep18-qualified50':False},
            'broad_tier3_policy_spend':False,'qualification_sha256':sha(q)})
        self.assert_pair(source, False)
        row=next(r for r in json.loads(q.read_text())['candidates'] if r['name']=='ep18-qualified50')
        self.assertEqual([m['early_drop'] for m in row['members']],[False,True,False])
        self.assertTrue(all(m['mppe_cm']<=10 for m in row['members']))
        print('EP18 REHEARSAL PASS: selective rescue; all MPPE <=10cm; one 50M early carry drop vetoes')

    def test_changed_asset_refused_before_training_and_payload_fence(self):
        run, _ = self.stage('ep18-corrupt')
        candidate = next((run/'payload').rglob('data.parquet'))
        original=candidate.read_bytes()
        candidate.write_bytes(original+b'changed candidate')
        self.assertIn('candidate SHA-256 mismatch', self.verify(run,False).stderr)
        candidate.write_bytes(original)
        self.f.launch(run,{'TRANSPORT_CORRUPT_CANDIDATE':'1'})
        log=self.f.workspace/f'{run.name}.log'
        # Fixture completion helper expects success, so wait directly for the failed box marker.
        import time
        end=time.monotonic()+30
        while time.monotonic()<end:
            if log.exists() and 'box failed ' in log.read_text():break
            time.sleep(.05)
        self.assertIn('candidate SHA-256 mismatch',log.read_text())
        self.assertIn('box failed ',log.read_text()); self.assertNotIn('box done ',log.read_text())
        self.assertFalse(self.f.events.exists())
        self.assertFalse((self.f.remote/'runs'/run.name/'manifest.json').exists())
        # Files under any Tier 1 tree or any extra clean payload are rejected, even before upload.
        clean,_=self.stage('ep18-fence')
        for relative in ['data/challenge/track_2/tier_1_multiview_caption/forbidden.parquet','data/clean/score.json']:
            forbidden=clean/'payload'/relative; put(forbidden,'forbidden fixture')
            self.assertIn('no clean/Tier 1 data permitted',self.verify(clean,False).stderr)
            transport=self.f.home/'transport.jsonl'
            before=transport.read_bytes()
            refused=self.f.command(['bash',str(self.f.controller/'scripts/vast/launch.sh'),str(clean),self.f.instance(clean)])
            self.assertEqual(refused.returncode,1,refused.stdout+refused.stderr)
            self.assertIn('no clean/Tier 1 data permitted',refused.stderr)
            self.assertEqual(transport.read_bytes(),before)
            forbidden.unlink()
        self.verify(clean)
        print('EP18 REHEARSAL PASS: changed candidate refuses pre.sh before any trainer; payload clean/Tier 1 injection refused')

    def test_unregistered_tier3_jobs_and_k_refused(self):
        run,result=self.stage('ep18-k5',['--k','5'],False)
        self.assertIn('ep18 requires k=3',result.stderr); self.assertFalse(run.exists())
        wrong=self.f.work/'wrong.txt'
        wrong.write_text((self.f.controller/READY/'generated/tier3/jobs.GATED.txt').read_text().replace('49999872','40001536'))
        run,result=self.stage('ep18-wrong',ok=False,jobs=wrong)
        self.assertIn('only the registered ep18',result.stderr); self.assertFalse(run.exists())

    def test_missing_qualification_diagnoses_without_export(self):
        out=self.f.work/'decision.json'
        result=self.f.command([sys.executable,'-B',str(self.f.controller/'scripts/ep18_pilot.py'),
            'decide','--qualification',str(self.f.work/'absent.json'),'--output',str(out)])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(out.read_text()),{'decision':'diagnose',
            'passes':{'ep18-qualified350':False,'ep18-qualified50':False},
            'broad_tier3_policy_spend':False,'qualification_sha256':None})

    def test_both_failed_results_diagnose(self):
        q=self.f.work/'qualification.json'
        q.write_text(json.dumps({'candidates':[{'name':name,'go':False,
            'members':[{'go':True},{'go':False},{'go':True}]} for name in
            ['ep18-qualified350','ep18-qualified50']]}))
        out=self.f.work/'decision.json'
        result=self.f.command([sys.executable,'-B',str(self.f.controller/'scripts/ep18_pilot.py'),
            'decide','--qualification',str(q),'--output',str(out)])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(out.read_text()),{'decision':'diagnose',
            'passes':{'ep18-qualified350':False,'ep18-qualified50':False},
            'broad_tier3_policy_spend':False,'qualification_sha256':sha(q)})
