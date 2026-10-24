"""Connected production rehearsal (brainstorm 6 PROD-READY): fakes only at the GPU and provider boundaries."""
import hashlib,json,os,shutil,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
import pandas as pd
import numpy as np
from .fixture import Fixture,BASE,HERE,wait,sha,put

class ProductionChain(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='journey-',dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.f=Fixture(Path(self.temp.name));self.addCleanup(self.f.stop_box)
    def tool(self,args,ok=True,env=None):
        command=[sys.executable,'-B',str(self.f.remote/'scripts/production_repeats.py'),*args]
        r=self.f.command(command,env,timeout=90)
        if ok:self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        else:self.assertNotEqual(r.returncode,0,r.stdout)
        return r
    def stage_cohort(self,run,tier,name,k=None):
        out=run/name
        args=['prepare','--jobs',str(run/'jobs.txt'),'--manifest',str(self.f.remote/f'data/fixture/manifest{tier}.json'),
              '--source',str(self.f.remote/'runs'/run.name),'--tier-root',str(self.f.remote/'data/challenge/track_2'/('tier_1_multiview_caption' if tier==1 else 'tier_2_synthetic_noise')),
              '--tier',str(tier),'--out',str(out)]
        if k is not None:args+=['--k',str(k)]
        self.tool(args);return out
    def test_connected_launch_pull_preempt_restore_repeat_and_three_export_routes(self):
        for tier in [1,2,3]:
            run,label,arm=self.f.run_spec(tier,f'tier{tier}')
            if tier==1:
                ready=self.f.home/'checkpoint-ready'
                self.f.launch(run,{'GPU_INTERRUPT':str(ready)})
                try:wait(ready)
                except AssertionError:
                    log=self.f.workspace/f'{run.name}.log'
                    raise AssertionError(log.read_text() if log.exists() else 'box never started')
                watchdog=subprocess.Popen(['bash',str(self.f.controller/'scripts/vast/watchdog.sh'),str(run),'100','1'],cwd=self.f.controller,
                    env=self.f.env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
                try:
                    wait(self.f.home/'periodic-pull')
                    policies={p.relative_to(run/'pulled'/run.name).as_posix():sha(p) for p in (run/'pulled'/run.name).rglob('policy_*.safetensors')}
                    self.assertEqual(len(policies),2)
                    states=list((run/'pulled').rglob('state_*.safetensors'));self.assertEqual(len(states),2)
                    self.f.stop_box();os.killpg(watchdog.pid,signal.SIGTERM);watchdog.wait(timeout=5)
                finally:
                    if watchdog.poll() is None:os.killpg(watchdog.pid,signal.SIGKILL);watchdog.wait()
                # A truly empty backing directory, retaining the same logical remote namespace.
                shutil.rmtree(self.f.workspace);self.f.provision()
                reference=next((run/'payload/data/fixture/tier1').rglob('data.parquet'))
                reference.write_bytes(b'changed regeneration must be replaced by frozen input')
                self.f.launch(run)
                self.f.finished(run)
                restored=self.f.remote/'runs'/run.name
                self.assertEqual({rel:sha(restored/rel) for rel in policies},policies)
                resumed=[event for event in self.f.calls() if event['kind']=='train' and event.get('resume')=='true']
                self.assertEqual(len(resumed),1);self.assertIn('state_330012672.safetensors',resumed[0]['state'])
                print('CHAIN periodic weights/states/frozen-input pull -> empty replacement -> hash-identical input+policy restore -> latest-state resume')
            else:
                self.f.launch(run);self.f.finished(run)
            source=self.f.remote/'runs'/run.name
            before=[e['kind'] for e in self.f.calls()].count('train')
            out=self.stage_cohort(run,tier,'k3-default')
            self.assertEqual(json.loads((out/'plan.json').read_text())['repeats'],3)
            self.tool(['run','--out',str(out)])
            self.tool(['qualify','--out',str(out)])
            selected=json.loads((out/'selected.json').read_text())
            self.assertEqual(set(selected),{'0','1'})
            self.assertTrue(all(p.endswith('policy_'+('349999104' if tier==1 else '49999872')+'.safetensors') for p in selected.values()))
            self.assertEqual(len(list((out/'cohort').rglob('trajectory.parquet'))),12 if tier==1 else 6)
            self.assertEqual([e['kind'] for e in self.f.calls()].count('train'),before)
            cohort_events=[e for e in self.f.calls() if e['kind']=='evaluate' and str(out/'cohort') in e['rollout']]
            self.assertEqual(len(cohort_events),12 if tier==1 else 6)
            if tier==1:
                for row in json.loads((out/'selection.json').read_text()):
                    self.assertEqual(row['auc_means'],{'330012672':.875,'349999104':.9375})
                first=next((out/'cohort').glob('t1-ep0-r01/*/*/evaluation/seq0/330012672/scores.json'))
                self.assertEqual(json.loads(first.read_text())['mean']['AUC'],1.)
                alternate=self.stage_cohort(run,tier,'earlier-wins',k=3)
                self.tool(['run','--out',str(alternate)],env={'GPU_FINAL_WORSE':'1'})
                self.tool(['qualify','--out',str(alternate)])
                self.assertTrue(all(p.endswith('policy_330012672.safetensors') for p in json.loads((alternate/'selected.json').read_text()).values()))
                print('SELECTION first trial favors 330M, k-mean favors final; changed independent cohort selects 330M when its mean wins')
            self.tool(['run','--out',str(out)],ok=False)
            merged=run/'release'
            released='tier_1_multiview_caption' if tier==1 else 'tier_2_synthetic_noise'
            self.tool(['merge','--cohort',str(out),'--cohort',str(out),'--tier-root',str(self.f.remote/'data/challenge/track_2'/released),'--out',str(run/'duplicate-release')],ok=False)
            self.tool(['merge','--cohort',str(out),'--tier-root',str(self.f.remote/'data/challenge/track_2'/released),'--out',str(merged)])
            env={}
            if tier!=1:
                guard=self.f.work/'guard';guard.mkdir(exist_ok=True)
                put(guard/'sitecustomize.py',"import sys\nfrom pathlib import Path\nimport pyarrow.parquet as pq\ndef check(path):\n if isinstance(path,(str,bytes,Path)):\n  p=Path(path.decode() if isinstance(path,bytes) else path)\n  if any('tier_1' in s or s=='baseline' for s in p.parts) or p.name=='pack_submission.py':raise PermissionError('hard-tier clean read')\ndef audit(event,args):\n if event=='open':check(args[0])\nsys.addaudithook(audit)\noriginal_file,original_read=pq.ParquetFile,pq.read_table\ndef parquet_file(source,*a,**k):\n check(source);return original_file(source,*a,**k)\ndef read_table(source,*a,**k):\n check(source);return original_read(source,*a,**k)\npq.ParquetFile,pq.read_table=parquet_file,read_table\nfor callback in [lambda:open('/absent/tier_1_multiview_caption/a.parquet'),lambda:pq.ParquetFile('/absent/tier_1_multiview_caption/a.parquet'),lambda:open('/absent/pack_submission.py')]:\n try:callback()\n except PermissionError:pass\n else:raise AssertionError('clean-read guard ineffective')\n")
                env={'PYTHONPATH':str(guard)}
            export_command=[sys.executable,'-B',str(self.f.remote/'scripts/production_repeats.py'),'export','--out',str(merged),
                            '--work',str(self.f.remote/f'data/fixture/tier{tier}'),'--commit','https://github.com/fixture/fixture/commit/0123456789abcdef']
            attempts=[subprocess.Popen(export_command,cwd=self.f.controller,env={**self.f.env,**env},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
            try:
                outputs=[p.communicate(timeout=90) for p in attempts]
                self.assertEqual(sorted(p.returncode for p in attempts),[0,1],outputs)
            finally:
                for p in attempts:
                    if p.poll() is None:p.kill();p.communicate()
            exported=[e for e in self.f.calls() if e['kind']=='evaluate' and str(merged/'export') in e['rollout']]
            self.assertEqual(len(exported),2,exported)
            csv=merged/f'export/track2_tier{tier}.csv';frame=pd.read_csv(csv)
            self.assertEqual(len(frame),20)
            np.testing.assert_allclose(frame[['pos_x','pos_y','pos_z','quat_x','quat_y','quat_z','quat_w']].to_numpy(),np.tile([.01,0,0,0,0,0,1],(20,1)),rtol=0,atol=1e-12)
            self.assertEqual(frame.row_id.tolist(),[f'track_2/e{e:06d}/w000/f{f:06d}/o0' for e in range(2) for f in range(10)])
            for episode,policy in selected.items():
                rows=frame[frame.row_id.str.contains(f'/e{int(episode):06d}/')]
                self.assertEqual(set(rows.checkpoint_sha256),{sha(Path(policy))})
            kit=self.f.remote/'external/kit/v2d_submission_kit';manifest=json.loads((kit/'KIT_MANIFEST.json').read_text())['files']
            for path in ['tools/pack_submission.py','submit/v2d_submit.py']:self.assertEqual(sha(kit/path),manifest[path])
            self.assertEqual(set(frame.packer_script_sha256),{sha(kit/'tools/pack_submission.py' if tier==1 else self.f.remote/'scripts/pack_no_clean.py')})
            self.tool(['export','--out',str(merged),'--work',str(self.f.remote/f'data/fixture/tier{tier}'),'--commit','unused'],ok=False)
            print(f'CHAIN tier{tier}: exact default k=3 -> frozen map -> one fresh export -> real packer -> offline validator; 20 fixture rows')
            if tier==2:
                # Fixed k=5 is supported; one missing result blocks qualification without new evaluations.
                five=self.stage_cohort(run,tier,'k5',k=5);self.tool(['run','--out',str(five)])
                self.assertEqual(len(list(five.rglob('trajectory.parquet'))),10)
                missing=next(five.rglob('trajectory.parquet'));retained=missing.read_bytes();missing.unlink()
                self.tool(['qualify','--out',str(five)],ok=False);self.assertFalse((five/'selected.json').exists());missing.write_bytes(retained)
                self.tool(['qualify','--out',str(five)])
                veto=self.stage_cohort(run,tier,'veto',k=3);self.tool(['run','--out',str(veto)],env={'GPU_BAD_REPEAT':'02'})
                self.tool(['qualify','--out',str(veto)])
                # A veto keeps coverage: same final map, failure status recorded, catastrophes first in the reseed queue.
                self.assertEqual(json.loads((veto/'selected.json').read_text()),selected)
                report=json.loads((veto/'veto.json').read_text())
                self.assertEqual([(v['episode'],v['status']) for v in report['candidates']],[(0,'catastrophe'),(1,'catastrophe')])
                self.assertEqual([m['catastrophe'] for m in report['candidates'][0]['members']],[False,True,False])
                self.assertEqual(report['reseed_order'],[0,1])
                self.assertEqual({v['status'] for v in json.loads((out/'veto.json').read_text())['candidates']},{'pass'})
                self.assertEqual(len(list(veto.rglob('trajectory.parquet'))),6)
                tier_root=str(self.f.remote/'data/challenge/track_2/tier_2_synthetic_noise')
                # A passing re-evaluation of the vetoed checkpoint is not a rescue; only a newly trained seed is.
                self.tool(['merge','--cohort',str(veto),'--rescue',str(out),'--tier-root',tier_root,'--out',str(run/'reeval-rescue')],ok=False)
                retained=run/'vetoed-release'
                self.tool(['merge','--cohort',str(veto),'--tier-root',tier_root,'--out',str(retained)])
                self.assertEqual(json.loads((retained/'selected.json').read_text()),selected)
                self.assertEqual(json.loads((retained/'veto.json').read_text())['episodes'],
                                 {e:{'original':'catastrophe','final':'catastrophe'} for e in ['0','1']})
                print('GATES k=5 exact count; missing member fails; one bad of k=3 vetoes but keeps coverage; re-evaluation cannot rescue; zero training')

class RepeatInterface(unittest.TestCase):
    def test_hard_tier_refuses_tier1_before_open(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            work=Path(temporary)
            bad={'path':str(work/'tier1/forbidden.parquet'),'sha256':'0'*64}
            manifest=work/'manifest.json'
            manifest.write_text(json.dumps({'tier':2,'scoring_mode':'self','sequences':[{'sequence':'seq','arms':{'raw_repaired':{
                'training_reference':bad,'clean_scoring_reference':bad,'support':{'path':'absent','sha256':'0'*64,'enabled':False}}}}]}))
            jobs=work/'jobs.txt';jobs.write_text('job|pilot_arms.txt|T2C-contact|raw_repaired|seq|49999872\n')
            result=subprocess.run([sys.executable,str(BASE/'scripts/production_repeats.py'),'prepare','--jobs',str(jobs),
                '--manifest',str(manifest),'--source',str(work/'absent'),'--tier-root',str(work/'absent'),
                '--tier','2','--out',str(work/'out')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('hard-tier input path names Tier 1',result.stderr)
            self.assertFalse((work/'out').exists())


    def test_freeze_public_cli_exists(self):
        result=subprocess.run([sys.executable,str(BASE/'scripts/freeze_inputs.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('restore',result.stdout)

    def test_public_cli_exists(self):
        result=subprocess.run([sys.executable,str(BASE/'scripts/production_repeats.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('prepare',result.stdout);self.assertIn('qualify',result.stdout);self.assertIn('export',result.stdout)

class FinancialStops(unittest.TestCase):
    def test_unreachable_credit_floor_and_budget_cap(self):
        sys.path.insert(0,str(BASE/'tests'))
        from test_runner_regressions import WatchdogTests
        for cap,credit,expected in [('0','10','budget cap'),('100','.5','credit')]:
            fixture=WatchdogTests('test_unreachable_at_cap_destroys_and_flags_missing_backup');fixture.setUp()
            try:
                result=fixture.watchdog(cap=cap,STATE='',STOP_FAIL='1',PULL_FAIL='1',CREDIT=credit)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual([e['kind'] for e in fixture.events()].count('destroy'),1)
                self.assertIn(expected,(fixture.run/'STOPPED').read_text());self.assertIn(expected,(fixture.run/'BACKUP_FAILED').read_text())
            finally:fixture.doCleanups()
        print('FINANCIAL unreachable at cap AND floor -> destroy requested, BACKUP_FAILED retained')
