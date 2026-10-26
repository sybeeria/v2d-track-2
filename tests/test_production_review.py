"""Production-path regressions from the brainstorm 6 code review (data/brainstorm6/CODE-REVIEW/REPORT.md)."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_runner_regressions import WatchdogTests, Sandbox, TrainingTests
from test_score_rollout import PilotQueueTest, SEQUENCES


def stop(process):
    try:
        descendants=psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants=[]
    groups={process.pid}
    for child in descendants:
        try: groups.add(os.getpgid(child.pid))
        except ProcessLookupError: pass
    for group in groups:
        if group==os.getpgrp(): continue
        try: os.killpg(group,signal.SIGKILL)
        except ProcessLookupError: pass
    process.communicate(timeout=5)


class MoneyTests(unittest.TestCase):
    def setUp(self):
        self.f = WatchdogTests('runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.env['WATCHDOG_COMMAND_TIMEOUT_SECONDS'] = '1'

    def bounded_watchdog(self, **env):
        f = self.f
        command = ['bash', 'scripts/vast/watchdog.sh', str(f.run), env.pop('cap', '100'), '1']
        p = subprocess.Popen(command, cwd=f.root, env={**f.env, **env},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=True)
        try:
            stdout, stderr = p.communicate(timeout=35)
        except subprocess.TimeoutExpired:
            stop(p)
            self.fail('watchdog never requested destruction: blocked transport prevented hard-limit enforcement')
        self.assertEqual(p.returncode, 0, stdout + stderr)
        self.assertEqual([e['kind'] for e in f.events()].count('destroy'), 1)

    def hang_sync(self):
        sync = self.f.bin / 'fake_sync'
        sync.write_text(sync.read_text().replace("event('sync',args=args)",
                        "event('sync',args=args)\n    if any('/runs/' in a for a in args): time.sleep(180)"))

    def test_cap_destroys_even_when_backup_never_returns(self):
        self.hang_sync()
        self.bounded_watchdog(cap='0', STATE='running')
        self.assertIn('budget cap', (self.f.run / 'STOPPED').read_text())
        self.assertIn('budget cap', (self.f.run / 'BACKUP_FAILED').read_text())

    def test_floor_destroys_even_when_stop_never_returns(self):
        remote = self.f.bin / 'fake_remote'
        remote.write_text(remote.read_text().replace("event('quiesce')",
                          "event('quiesce'); time.sleep(180)"))
        self.bounded_watchdog(STATE='running', CREDIT='.996')
        self.assertIn('credit', (self.f.run / 'STOPPED').read_text())

    def test_soft_backup_timeout_retains_box_until_floor(self):
        self.hang_sync()
        transport = self.f.bin / 'fake_provider'
        transport.write_text(transport.read_text().replace("event('destroy')", "event('destroy',tick=tick)"))
        self.bounded_watchdog(STATE='done', CREDIT_DROP_TICK='1800')
        destroys = [e for e in self.f.events() if e['kind'] == 'destroy']
        self.assertEqual([e['tick'] for e in destroys], [1800])
        self.assertIn('credit', (self.f.run / 'STOPPED').read_text())

    def test_relaunch_cannot_rebind_a_watched_run(self):
        self.hang_sync()
        f = self.f
        p = subprocess.Popen(['bash', 'scripts/vast/watchdog.sh', str(f.run), '100', '1'],
                             cwd=f.root, env=f.env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
        self.addCleanup(stop, p)
        deadline = time.monotonic() + 5
        while not any(e['kind'] == 'sync' for e in f.events()) and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(any(e['kind'] == 'sync' for e in f.events()))
        before = (f.run / 'host').read_bytes()
        result = f.execute(['bash', 'scripts/vast/launch.sh', str(f.run), '456'])
        self.assertNotEqual(result.returncode, 0, 'launch rebound the run while its watchdog still owned instance 123')
        self.assertEqual((f.run / 'host').read_bytes(), before)
        self.assertNotIn('launch', [e['kind'] for e in f.events()])

    def test_second_instance_refused_before_watcher_starts(self):
        before=(self.f.run/'host').read_bytes()
        result=self.f.execute(['bash','scripts/vast/launch.sh',str(self.f.run),'456'])
        self.assertNotEqual(result.returncode,0,'a second launch replaced the instance binding before any watcher could arm')
        self.assertEqual((self.f.run/'host').read_bytes(),before)
        self.assertNotIn('launch',[e['kind'] for e in self.f.events()])
        (self.f.run/'STOPPED').write_text('completed destroy request\n')
        replacement=self.f.execute(['bash','scripts/vast/launch.sh',str(self.f.run),'456'])
        self.assertEqual(replacement.returncode,0,replacement.stderr)
        self.assertTrue((self.f.run/'host').read_text().startswith('456 '))
        self.assertFalse((self.f.run/'STOPPED').exists())
        self.assertEqual([e['kind'] for e in self.f.events()].count('launch'),1)


BOOTSTRAP_BOUNDARY = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
root=Path(os.environ['BOOTSTRAP_FIXTURE']); tool=Path(sys.argv[0]).resolve().name; args=sys.argv[1:]
def event(name):
    with (root/'events').open('a') as f: f.write(name+'\n')
if tool=='b6_gpu':
    if '--query-gpu=driver_version' in args: print('600.0')
    elif '--query-gpu=memory.free' in args: print(os.environ.get('B6_GPU_FREE','50000'))
    else: print('GPU fixture')
elif tool=='b6_ldconfig': print('libEGL.so.1')
elif tool=='b6_nproc': print(os.environ.get('B6_NPROC','2'))
elif tool=='b6_uv':
    if args[0]=='venv':
        directory=Path(args[-1]); (directory/'bin').mkdir(parents=True,exist_ok=True)
        python=directory/'bin/python'; python.symlink_to(root/'bin/b6_python')
    elif 'torch' in args:
        event('base-install')
        attempt=root/'first-attempt'
        if not attempt.exists(): attempt.touch(); sys.exit(7)
        (root/'base-installed').touch()
elif tool=='b6_python':
    if any('setup_soma_assets.py' in a for a in args) and not (root/'base-installed').exists():
        print('ModuleNotFoundError: interrupted base installation was skipped',file=sys.stderr); sys.exit(12)
elif tool in ('b6_curl','b6_apt','b6_unzip'):
    sys.exit('unexpected network/install boundary: '+tool)
'''


class BootstrapTests(Sandbox):
    def bootstrap(self):
        root = self.root
        bins = self.work / 'bin'; bins.mkdir()
        for tool in ('b6_gpu', 'b6_ldconfig', 'b6_nproc', 'b6_uv', 'b6_python',
                     'b6_git', 'b6_git-lfs', 'b6_unzip', 'b6_curl', 'b6_apt'):
            file = bins / tool; file.write_text(BOOTSTRAP_BOUNDARY); file.chmod(0o755)
        for path in ('external/video_to_data/.git', 'external/kit/v2d_submission_kit',
                     'external/video_to_data/robotic_grounding/flash_chord'):
            (root/path).mkdir(parents=True,exist_ok=True)
        (root/'scripts/apply_retarget_patch.sh').write_text('#!/bin/sh\nexit 0\n')
        script=root/'scripts/gpu_bootstrap.sh'
        source=script.read_text()
        for old,new in [('nvidia-smi','b6_gpu'),('ldconfig','b6_ldconfig'),('nproc','b6_nproc'),
                        ('git-lfs','b6_git-lfs'),('git','b6_git'),('uv','b6_uv'),('unzip','b6_unzip'),
                        ('curl','b6_curl'),('apt-get','b6_apt')]:
            source=re.sub(r'(?<![\w-])'+re.escape(old)+r'(?![\w-])',new,source)
        script.write_text(source)
        self.env.update(PATH=str(bins)+os.pathsep+os.environ['PATH'],
                        BOOTSTRAP_FIXTURE=str(self.work), BOOTSTRAP_TIERS=' ')
        return script

    def test_retry_finishes_interrupted_dependency_installation(self):
        root = self.root
        script=self.bootstrap()
        first=self.execute(['bash',str(script)])
        self.assertEqual(first.returncode,7,first.stdout+first.stderr)
        self.assertTrue((root/'.venv/bin/python').is_file())
        second=self.execute(['bash',str(script)])
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        self.assertEqual((self.work/'events').read_text().splitlines(),['base-install','base-install'])
        self.assertTrue((self.work/'base-installed').exists())

    def retarget_jobs(self, cpus, gpu_free):
        script=self.bootstrap()
        (self.work/'first-attempt').touch()
        (self.root/'scripts/retarget_tier.sh').write_text('#!/bin/sh\necho "retarget $3" >> "$BOOTSTRAP_FIXTURE/events"\n')
        result=self.execute(['bash',str(script)],BOOTSTRAP_TIERS='tier1',B6_NPROC=cpus,B6_GPU_FREE=gpu_free)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return [e for e in (self.work/'events').read_text().splitlines() if e.startswith('retarget ')]

    def test_retarget_parallelism_fits_gpu_memory_on_many_core_hosts(self):
        # Oct 26 canary: 64 visible cores gave 32 parallel retargets on one 48 GB GPU, which ran out of memory.
        self.assertEqual(self.retarget_jobs('64','48402'),['retarget 4'])

    def test_retarget_parallelism_still_follows_cpus_on_small_hosts(self):
        self.assertEqual(self.retarget_jobs('4','48402'),['retarget 2'])


class BoxTests(unittest.TestCase):
    def test_bootstrap_failure_emits_terminal_box_failure(self):
        fixture=TrainingTests('runTest'); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        box=fixture.make_box()
        (fixture.root/'scripts/gpu_bootstrap.sh').write_text('#!/bin/sh\nexit 7\n')
        result=fixture.execute(['bash',str(box)])
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertEqual(sum(line.startswith('box failed ') for line in result.stdout.splitlines()),1,
                         'bootstrap exited nonzero without a terminal marker; watchdog waits an hour as if preempted')
        self.assertNotIn('box done ',result.stdout)
        self.assertFalse((fixture.work/'post-ran').exists())


class IntegrityTests(Sandbox):
    def test_tier3_rejects_normal_retargeted_tier1_work_tree(self):
        clean=self.root/'data/work/tier1/motion/whole_body/soma/sequence_id=fixture/robot_name=g1/data.parquet'
        clean.parent.mkdir(parents=True); clean.write_bytes(b'owned synthetic Tier 1 reference')
        support=self.work/'support.usda'; support.write_bytes(b'#usda 1.0')
        def record(path): return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        arm={'training_reference':record(clean),'clean_scoring_reference':record(clean),
             'support':{**record(support),'enabled':False}}
        manifest=self.work/'manifest.json'
        manifest.write_text(json.dumps({'tier':3,'scoring_mode':'self','sequences':[
            {'sequence':'fixture','arms':{'tier3_raw':arm}}]}))
        result=self.execute([sys.executable,'-B',str(self.root/'scripts/pilot_queue.py'),
            '--arms',str(self.root/'scripts/pilot_arms.txt'),'--arm','T2C-contact',
            '--manifest',str(manifest),'--inputs','tier3_raw','--steps','49999872',
            '--out-root',str(self.work/'out'),'--dry-run'])
        self.assertNotEqual(result.returncode,0,'Tier 3 CLI scheduled training/scoring on data/work/tier1:\n'+result.stdout)
        self.assertNotIn('scripts/score_rollout.py',result.stdout)
        self.assertFalse((self.work/'out').exists())


class KitIntegrityTests(unittest.TestCase):
    def setUp(self):
        import shutil
        import pandas as pd
        self.temp=tempfile.TemporaryDirectory(prefix='kit-integrity-',dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.temp.cleanup)
        self.work=Path(self.temp.name)
        self.kit=self.work/'kit'
        shutil.copytree(ROOT/'external/kit/v2d_submission_kit',self.kit,
                        ignore=shutil.ignore_patterns('__pycache__'))
        self.frame=pd.read_csv(self.kit/'data/track_2_sample_submission.csv',dtype=str)
        self.manifest=json.loads((self.kit/'KIT_MANIFEST.json').read_text())['files']
        self.own=hashlib.sha256((ROOT/'scripts/pack_no_clean.py').read_bytes()).hexdigest()
        self.frame['uploader_script_sha256']=self.manifest['submit/v2d_submit.py']
        self.csv=self.work/'fixture.csv'

    def validate(self,track,packer):
        self.frame['packer_script_sha256']=packer
        self.frame.to_csv(self.csv,index=False)
        return subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validate_offline.py'),
            str(self.kit),str(self.csv),track],capture_output=True,text=True,timeout=60)

    def test_each_tier_must_name_the_packer_that_wrote_it(self):
        official=self.manifest['tools/pack_submission.py']
        for track,packer,ok in [('track_2_tier1',official,True),('track_2_tier1',self.own,True),('track_2_tier2',self.own,True),
                                ('track_2_tier3',self.own,True),('track_2_tier1','0'*64,False),('track_2_tier2',official,False),
                                ('track_2_tier3',official,False)]:
            with self.subTest(track=track,packer=packer):
                result=self.validate(track,packer)
                self.assertEqual(result.returncode==0,ok,result.stderr)
                if not ok: self.assertIn('packer_script_sha256',result.stderr)

    def test_modified_kit_file_is_rejected(self):
        file=self.kit/'tools/pack_submission.py'; file.write_bytes(file.read_bytes()+b'\n# owned mutation\n')
        result=self.validate('track_2_tier1',self.manifest['tools/pack_submission.py'])
        self.assertNotEqual(result.returncode,0,'offline gate accepted a kit differing from KIT_MANIFEST.json')
        self.assertIn('KIT_MANIFEST',result.stderr)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='summary-',dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.temp.cleanup)
        self.work=Path(self.temp.name).resolve()
        fixture=PilotQueueTest()
        manifest=fixture.manifest(self.work)
        self.events,self.env=fixture.fake_python(self.work)
        self.command=[str(ROOT/'scripts/pilot_queue.sh'),'--arms',str(ROOT/'scripts/pilot_arms.txt'),
                      '--manifest',str(manifest),'--arm','E1-control','--sequences',SEQUENCES[1],
                      '--out-root',str(self.work/'runs'),'--steps','49999872']
        first=self.run_command()
        self.assertEqual(first.returncode,0,first.stdout+first.stderr)
        positive=self.run_command('--summarize-only')
        self.assertEqual(positive.returncode,0,positive.stderr)
        self.assertEqual(json.loads((self.work/'runs/summary.json').read_text())['candidates'][0]['MPPE'],0.)
        self.assertEqual(self.events.read_text().splitlines(),['train','evaluate'])

    def run_command(self,*args):
        return subprocess.run([*self.command,*args],env=self.env,capture_output=True,text=True,timeout=60)

    def test_summary_rejects_changed_checkpoint(self):
        checkpoint=next((self.work/'runs').rglob('policy_*.safetensors'))
        checkpoint.write_bytes(checkpoint.read_bytes()+b'\n')
        result=self.run_command('--summarize-only')
        self.assertNotEqual(result.returncode,0,'stale perfect scores certified a different checkpoint')
        self.assertEqual(self.events.read_text().splitlines(),['train','evaluate'])

    def test_summary_rejects_replaced_rollout(self):
        import pyarrow.parquet as pq
        import pyarrow as pa
        path=next((self.work/'runs').rglob('trajectory.parquet'))
        table=pq.ParquetFile(path).read()
        index=table.column_names.index('achieved_position_x')
        table=table.set_column(index,'achieved_position_x',pa.array(table['achieved_position_x'].to_numpy()+1.))
        pq.write_table(table,path)
        result=self.run_command('--summarize-only')
        self.assertNotEqual(result.returncode,0,'old MPPE=0 certified a rollout moved by one metre')
        self.assertEqual(self.events.read_text().splitlines(),['train','evaluate'])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(loader.loadTestsFromTestCase(cls)
        for cls in (MoneyTests,BootstrapTests,BoxTests,IntegrityTests,KitIntegrityTests,SummaryTests))


if __name__=='__main__':
    suite=load_tests(unittest.defaultTestLoader,None,None)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
