"""Public-process regression tests. Only local fake transports run; no provider CLI or SSH."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(os.environ.get('BUG_RUNNER_REPO', Path(__file__).resolve().parents[1]))


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='case-', dir=os.environ.get('BUG_RUNNER_TMP', '/tmp'))
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.root = self.work / 'project'
        shutil.copytree(REPO / 'scripts', self.root / 'scripts')
        self.env = {**os.environ, 'PACK_PYTHON': sys.executable, 'PYTHONDONTWRITEBYTECODE': '1'}

    def execute(self, command, **env):
        return subprocess.run(command, cwd=self.root, env={**self.env, **env},
                              text=True, capture_output=True, timeout=30)


class QueueTests(Sandbox):
    def setUp(self):
        super().setUp()
        self.events = self.work / 'events'
        pilot = self.root / 'scripts/pilot_queue.sh'
        pilot.write_text('''#!/usr/bin/env python3
import os,sys,json,time
from pathlib import Path
args=sys.argv[1:]; out=Path(args[args.index('--out-root')+1]); name=out.name
out.mkdir(parents=True,exist_ok=True)
if not (out/'complete').exists():
    with open(os.environ['EVENTS'],'a') as f: f.write(json.dumps({'job':name,'gpu':args[args.index('--gpu')+1]})+'\\n')
    Path(os.environ['EVENTS']+'.ready').touch()
    time.sleep(float(os.environ.get('JOB_DELAY','0')))
    if name in os.environ.get('FAIL_JOBS','').split(','): sys.exit(7)
    (out/'complete').write_text('done')
(out/'summary.tsv').write_text('job\\n'+name+'\\n')
''')
        pilot.chmod(0o755)
        self.jobs = self.work / 'jobs.txt'
        self.out = self.work / 'runs'
        self.env.update(EVENTS=str(self.events), JOBS_FILE=str(self.jobs))

    def plan(self, names, newline=True):
        self.jobs.write_text('\n'.join(f'{n}|arms.txt|arm|clean|seq|2048' for n in names) + ('\n' if newline else ''))

    def run_queue(self, out=None, **env):
        return self.execute(['bash', 'scripts/test_day.sh', '1', str(out or self.out)], **env)

    def calls(self):
        return [json.loads(line)['job'] for line in self.events.read_text().splitlines()] if self.events.exists() else []

    def test_failed_job_retries_after_restart(self):
        self.plan(['bad', 'good'])
        self.run_queue(FAIL_JOBS='bad')
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), ['bad', 'good', 'bad'])
        self.assertEqual((self.out / 'summary.tsv').read_text(), 'job\nbad\ngood\n')

    def test_failure_reaches_caller_and_later_jobs_still_run(self):
        self.plan(['bad', 'good', 'later'])
        result = self.run_queue(FAIL_JOBS='bad')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.calls(), ['bad', 'good', 'later'])

    def test_abandoned_claim_does_not_skip_job(self):
        self.plan(['unfinished'])
        (self.out / '.claim-0').mkdir(parents=True)
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), ['unfinished'])

    def test_reordered_job_list_does_not_transfer_completion(self):
        self.plan(['a', 'b'])
        self.run_queue(FAIL_JOBS='b')
        self.plan(['b', 'a', 'c'])
        self.run_queue()
        self.assertEqual(self.calls(), ['a', 'b', 'b', 'c'])

    def test_unterminated_last_job_is_not_dropped(self):
        self.plan(['a', 'last'], newline=False)
        self.run_queue()
        self.assertEqual(self.calls(), ['a', 'last'])

    def test_duplicate_names_fail_before_training(self):
        self.plan(['a', 'a'])
        result = self.run_queue()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_two_box_shards_cover_each_job_once(self):
        self.plan(['a', 'b', 'c', 'd'])
        for shard in range(2):
            result = self.run_queue(self.work / f'box{shard}', SHARD_COUNT='2', SHARD_INDEX=str(shard))
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(self.calls()), ['a', 'b', 'c', 'd'])

    def test_same_root_concurrent_runner_is_rejected(self):
        self.plan(['a'])
        first = subprocess.Popen(['bash', 'scripts/test_day.sh', '1', str(self.out)], cwd=self.root,
                                 env={**self.env, 'JOB_DELAY': '1'}, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not Path(str(self.events) + '.ready').exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(Path(str(self.events) + '.ready').exists())
            second = self.run_queue()
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(first.wait(timeout=5), 0)
        finally:
            if first.poll() is None:
                first.kill(); first.wait()
            first.communicate()


TRANSPORT = r'''#!/usr/bin/env python3
import json,os,sys,time,shutil,datetime,subprocess
from pathlib import Path
root=Path(os.environ['FAKE_ROOT']); tool=Path(sys.argv[0]).name; args=sys.argv[1:]
clock=root/'clock'; tick=int(clock.read_text()) if clock.exists() else 0
def event(kind, **kw):
    with (root/'transport.jsonl').open('a') as f: f.write(json.dumps(dict(kind=kind,**kw))+'\n')
if tool=='fake_provider':
    if args[:2]==['show','instance']:
        data={'start_date':time.time()-float(os.environ.get('AGE','0')), 'dph_total':1.2,
              'public_ipaddr':'offline.invalid','ports':{'22/tcp':[{'HostPort':'9'}]}}
        if os.environ.get('BAD_INFO'): data.pop('dph_total')
        print(json.dumps(data))
    elif args[:2]==['show','user']:
        drop=os.environ.get('CREDIT_DROP_TICK')
        print(json.dumps({'credit':.5 if drop and tick>=int(drop) else float(os.environ.get('CREDIT','10'))}))
    elif args[:2]==['destroy','instance']:
        event('destroy')
        attempts=sum(json.loads(l)['kind']=='destroy' for l in (root/'transport.jsonl').read_text().splitlines())
        if attempts<=int(os.environ.get('DESTROY_FAILURES','0')): sys.exit(9)
elif tool=='fake_remote':
    cmd=args[-1]
    (root/'remote-ready').touch()
    time.sleep(float(os.environ.get('TRANSPORT_DELAY','0')))
    parsed=subprocess.run(['bash','-n'],input=cmd,text=True,capture_output=True)
    if parsed.returncode:
        print(parsed.stderr,file=sys.stderr); sys.exit(2)
    if 'read -r pid' in cmd:
        event('quiesce')
        if os.environ.get('STOP_FAIL'): sys.exit(1)
    elif 'deadman.sh' in cmd:
        event('arm', cmd=cmd)
    elif 'setsid nohup' in cmd:
        event('launch'); print('launched')
    elif 'echo run tree absent' in cmd:
        print('run tree absent' if os.environ.get('RUN_TREE_ABSENT') else 'run tree present')
    elif 'then echo done' in cmd:
        print('done' if tick>=4800 else os.environ.get('STATE','done'))
    elif 'PYSTATES' in cmd:
        compile(cmd.split("<<'PYSTATES'\n",1)[1].rsplit('\nPYSTATES',1)[0],'<state selector>','exec')
        names=sorted((p for p in (root/'remote').glob('state_*.safetensors')),
                     key=lambda p:int(p.stem.split('_')[1]))
        print('\n'.join(p.name for p in names[-2:]))
    elif 'PYJOBS' in cmd:
        code=cmd.split("<<'PYJOBS'\n",1)[1].rsplit('\nPYJOBS',1)[0]
        fakeproc=root/'proc'; fakeproc.mkdir(exist_ok=True)
        runs=root/'remote_jobs'; runs.mkdir(exist_ok=True)
        for pid in ['101','102']:
            proc=fakeproc/pid; proc.mkdir(exist_ok=True)
            out=runs/pid; out.mkdir(exist_ok=True)
            moving=os.environ.get('GROWING') and not (os.environ.get('PARTIAL_HANG') and pid=='101')
            (out/'train.log').write_text('training progress '+str(tick if moving else 2048)+'\n')
            (proc/'cmdline').write_bytes(('python\0scripts/train_flash_sac.py\0output_dir='+str(out)+'\0').encode())
        code=code.replace("Path('/proc')",'Path('+repr(str(fakeproc))+')')
        result=subprocess.run([sys.executable,'-',str(runs)],input=code,text=True,capture_output=True)
        print(result.stdout,end=''); print(result.stderr,end='',file=sys.stderr); sys.exit(result.returncode)
    elif '-printf' in cmd:
        print('steps=2048'+(str(tick) if os.environ.get('GROWING') else ''))
    else: print('offline status')
elif tool=='fake_sync':
    event('sync',args=args)
    if any('/runs/' in a for a in args):
        if os.environ.get('PULL_FAIL') or os.environ.get('RUN_PULL_FAIL') or os.environ.get('RUN_TREE_ABSENT'): sys.exit(23)
        excludes=[args[i+1] for i,a in enumerate(args[:-1]) if a=='--exclude']
        source=root/'remote'
        rsources=[a for a in args if a.startswith('root@') and '/runs/' in a]
        dest=Path(args[-1]) if rsources and rsources[0].endswith('/') else Path(args[-1])/'batch'
        if args[-1].startswith('root@'): pass
        else:
            dest.mkdir(parents=True,exist_ok=True)
            import fnmatch
            selection=next((Path(a.split('=',1)[1]).read_text().splitlines() for a in args if a.startswith('--files-from=')),None)
            if selection is not None and os.environ.get('STATE_PULL_FAIL'): sys.exit(23)
            for f in source.iterdir():
                if selection is not None and f.name not in selection: continue
                if not any(fnmatch.fnmatch(f.name, pat) for pat in excludes): shutil.copyfile(f,dest/f.name)
    elif os.environ.get('PULL_FAIL'): sys.exit(23)
    if any('/pulled/batch/' in a and not a.startswith('root@') for a in args):
        event('restore')
        source=next(a for a in args if '/pulled/batch/' in a and not a.startswith('root@'))
        shutil.copytree(source,root/'restored',dirs_exist_ok=True)
elif tool=='fake_pause':
    clock.write_text(str(tick+int(args[0])))
elif tool=='fake_date':
    print(100000+tick if args[-1]=='+%s' else datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
'''


class WatchdogTests(Sandbox):
    def setUp(self):
        super().setUp()
        self.bin = self.work / 'bin'; self.bin.mkdir()
        for name in ['fake_provider', 'fake_remote', 'fake_sync', 'fake_pause', 'fake_date']:
            p = self.bin / name; p.write_text(TRANSPORT); p.chmod(0o755)
        # Replace only executable tokens at the external boundary in BOTH original and patched copies.
        # No executable named ssh or vastai is ever invoked, not even a fake of that name.
        import re
        for name in ['watchdog.sh', 'launch.sh']:
            p = self.root / 'scripts/vast' / name
            text = p.read_text()
            for a, b in [('vastai', 'fake_provider'), ('ssh', 'fake_remote'), ('rsync', 'fake_sync'),
                         ('sleep', 'fake_pause'), ('date', 'fake_date')]:
                text = re.sub(r'(?<!\.)\b' + a + r'\b', b, text)  # not Python attributes such as time.sleep
            p.write_text(text)
        self.run = self.work / 'batch'; self.run.mkdir()
        (self.run / 'host').write_text('123 offline.invalid 9\n')
        (self.run / 'jobs.txt').write_text('unused\n')
        remote = self.work / 'remote'; remote.mkdir()
        for name in ['policy_20.safetensors', 'state_9.safetensors', 'state_10.safetensors', 'state_20.safetensors',
                     'replay_20.safetensors', 'summary.tsv', '.state_30.safetensors.tmp']:
            (remote / name).write_text('owned fixture ' + name)
        self.env.update(PATH=str(self.bin)+os.pathsep+os.environ['PATH'], FAKE_ROOT=str(self.work))

    def watchdog(self, cap='100', floor='1', **env):
        return self.execute(['bash', 'scripts/vast/watchdog.sh', str(self.run), cap, floor], **env)

    def events(self):
        p = self.work / 'transport.jsonl'
        return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

    def test_done_preserves_policy_and_resume_state(self):
        result = self.watchdog()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.run / 'pulled/batch/policy_20.safetensors').is_file())
        self.assertEqual((self.run / 'pulled/batch/policy_20.safetensors').read_text(),
                         'owned fixture policy_20.safetensors')
        self.assertTrue((self.run / 'pulled/batch/state_20.safetensors').is_file())
        self.assertFalse((self.run / 'pulled/batch/.state_30.safetensors.tmp').exists())
        self.assertEqual([e['kind'] for e in self.events()].count('destroy'), 1)

    def arming(self):
        import re
        arms = [e['cmd'] for e in self.events() if e['kind'] == 'arm']
        self.assertTrue(arms)
        return [tuple(map(int, re.search(r'echo (\d+) (\d+) > /workspace/batch\.deadline\.tmp', c).groups())) for c in arms]

    def test_box_deadman_is_armed_30_min_past_the_cap(self):
        before = time.time()
        result = self.watchdog(cap='6', AGE='600')
        self.assertEqual(result.returncode, 0, result.stderr)
        # $6 at the fake's $1.20/h is 5 h after a start 600 s ago; the backstop adds 30 min.
        for deadline, grace in self.arming():
            self.assertAlmostEqual(deadline - before, 5 * 3600 + 1800 - 600, delta=30)
            self.assertEqual(grace, 5400)

    def test_long_pull_timeout_lengthens_the_deadman_idle_grace(self):
        result = self.watchdog(cap='6', AGE='600', WATCHDOG_COMMAND_TIMEOUT_SECONDS='1800')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({grace for _, grace in self.arming()}, {10 * 1800 + 1200})

    def test_keep_disables_only_the_deadman_idle_stop(self):
        (self.run / 'KEEP').touch()
        result = self.watchdog(cap='6', AGE='600')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({grace for _, grace in self.arming()}, {0})

    def test_duplicate_watchdog_is_rejected(self):
        first=subprocess.Popen(['bash','scripts/vast/watchdog.sh',str(self.run),'100','1'],cwd=self.root,
            env={**self.env,'TRANSPORT_DELAY':'.2'},stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            deadline=time.monotonic()+5
            while not (self.work/'remote-ready').exists() and time.monotonic()<deadline: time.sleep(.02)
            self.assertTrue((self.work/'remote-ready').exists())
            second=self.watchdog()
            self.assertNotEqual(second.returncode,0)
            self.assertEqual(first.wait(timeout=10),0)
        finally:
            if first.poll() is None: first.kill(); first.wait()
            first.communicate()

    def test_archive_keeps_two_states_and_all_policies(self):
        local = self.run/'pulled/batch'; local.mkdir(parents=True)
        (local/'state_1.safetensors').write_text('old backed up state')
        result = self.watchdog()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(p.name for p in local.glob('*.safetensors')),
                         ['policy_20.safetensors', 'state_10.safetensors', 'state_20.safetensors'])

    def assert_retained_until_credit_floor(self, **env):
        # Credit drops below the floor at the fourth cycle; until then a finished box whose backup fails is kept.
        result = self.watchdog(CREDIT_DROP_TICK='1800', **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        kinds = [e['kind'] for e in self.events()]
        self.assertEqual(kinds.count('destroy'), 1)
        self.assertEqual(kinds.count('quiesce'), 5)  # four soft attempts at ticks 0-1800, then the hard one
        self.assertIn('credit', (self.run / 'STOPPED').read_text())

    def test_failed_state_transfer_keeps_old_backup_and_box(self):
        local = self.run/'pulled/batch'; local.mkdir(parents=True)
        (local/'state_1.safetensors').write_text('old backed up state')
        self.assert_retained_until_credit_floor(STATE_PULL_FAIL='1')
        self.assertEqual((local/'state_1.safetensors').read_text(), 'old backed up state')
        self.assertIn('credit', (self.run / 'BACKUP_FAILED').read_text())

    def test_failed_backup_blocks_soft_destroy(self):
        self.assert_retained_until_credit_floor(PULL_FAIL='1')
        self.assertIn('credit', (self.run / 'BACKUP_FAILED').read_text())

    def test_box_failed_before_creating_its_run_tree_is_destroyed(self):
        # Oct 26 canary: bootstrap failed before runs/<name> existed, every tree pull failed, and the soft
        # destroy kept the box billing toward its cap.
        result = self.watchdog(STATE='failed', RUN_TREE_ABSENT='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        self.assertEqual([e['kind'] for e in events].count('destroy'), 1)
        self.assertEqual((self.run / 'STOPPED').read_text().strip(), 'box failed')
        self.assertFalse((self.run / 'BACKUP_FAILED').exists())
        syncs = [e['args'] for e in events if e['kind'] == 'sync']
        self.assertFalse([a for a in syncs if any('/runs/' in x for x in a)])
        self.assertTrue([a for a in syncs if any(x.endswith(':/workspace/*.log') for x in a)])

    def test_failed_run_pull_not_hidden_by_successful_log_pull(self):
        self.assert_retained_until_credit_floor(RUN_PULL_FAIL='1')
        self.assertTrue((self.run / 'BACKUP_FAILED').exists())

    def test_destroy_failure_is_retried_not_recorded_as_stopped(self):
        result = self.watchdog(DESTROY_FAILURES='2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([e['kind'] for e in self.events()].count('destroy'), 3)
        log = (self.run / 'watchdog.log').read_text()
        self.assertEqual(log.count('destroy failed (finished)'), 2)
        self.assertLess(log.rindex('destroy failed'), log.index('destroy requested'))
        self.assertEqual((self.run / 'STOPPED').read_text().strip(), 'finished')

    def test_budget_stops_writer_then_pulls_then_destroys(self):
        result = self.watchdog(cap='0', STATE='running')
        self.assertEqual(result.returncode, 0, result.stderr)
        kinds = [e['kind'] for e in self.events()]
        self.assertIn('quiesce', kinds)
        stop = kinds.index('quiesce')
        self.assertIn('sync', kinds[stop+1:kinds.index('destroy')])

    def test_stop_failure_retains_instance_until_hard_limit(self):
        self.assert_retained_until_credit_floor(STOP_FAIL='1')
        self.assertFalse((self.run / 'BACKUP_FAILED').exists())

    def test_stop_failure_at_cap_still_destroys(self):
        result = self.watchdog(cap='0', STATE='running', STOP_FAIL='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([e['kind'] for e in self.events()].count('destroy'), 1)
        self.assertIn('budget cap', (self.run / 'STOPPED').read_text())

    def test_fractional_credit_floor_is_enforced(self):
        result = self.watchdog(STATE='running', CREDIT='.996')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('credit', (self.run / 'STOPPED').read_text())

    def test_invalid_instance_metadata_fails_before_arming(self):
        result = self.watchdog(BAD_INFO='1', STATE='running')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('destroy', [e['kind'] for e in self.events()])

    def test_live_process_with_no_progress_is_stopped(self):
        result = self.watchdog(STATE='running')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('no progress', (self.run / 'STOPPED').read_text())

    def test_one_hung_job_cannot_hide_behind_other_gpu(self):
        result=self.watchdog(STATE='running',GROWING='1',PARTIAL_HANG='1')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('job no progress',(self.run/'STOPPED').read_text())

    def test_real_progress_is_not_a_hang(self):
        result = self.watchdog(STATE='running', GROWING='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.run / 'STOPPED').read_text().strip(), 'finished')

    def test_explicit_batch_failure_is_terminal(self):
        result = self.watchdog(STATE='failed')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.run/'STOPPED').read_text().strip(), 'box failed')

    def test_dead_parent_path_preserves_checkpoints(self):
        result = self.watchdog(STATE='dead')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.run/'STOPPED').read_text().strip(), 'box script dead')
        self.assertTrue((self.run/'pulled/batch/state_20.safetensors').is_file())

    def test_unreachable_at_cap_destroys_and_flags_missing_backup(self):
        result = self.watchdog(cap='0', STATE='', STOP_FAIL='1', PULL_FAIL='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([e['kind'] for e in self.events()].count('destroy'), 1)
        self.assertIn('budget cap', (self.run / 'STOPPED').read_text())
        self.assertIn('budget cap', (self.run / 'BACKUP_FAILED').read_text())

    def test_replacement_launch_restores_before_start(self):
        (self.run / 'pulled/batch').mkdir(parents=True)
        (self.run / 'pulled/batch/state_20.safetensors').write_text('resume')
        result = self.execute(['bash', 'scripts/vast/launch.sh', str(self.run), '123'])
        self.assertEqual(result.returncode, 0, result.stderr)
        kinds = [e['kind'] for e in self.events()]
        self.assertIn('restore', kinds)
        self.assertLess(kinds.index('restore'), kinds.index('launch'))
        self.assertEqual((self.work/'restored/state_20.safetensors').read_text(), 'resume')


class TrainingTests(Sandbox):
    def test_latest_state_is_used_with_same_budget(self):
        reference = self.work / 'sequence_id=episode/robot_name=g1/data.parquet'
        reference.parent.mkdir(parents=True); reference.write_text('fixture')
        out = self.work / 'out/episode'; out.mkdir(parents=True)
        for step in [9, 20]: (out / f'state_{step}.safetensors').write_text('fixture')
        fake = self.work / 'fake_train'
        fake.write_text('''#!/usr/bin/env python3
import os,sys,json
if sys.argv[1]=='-': print('false')
else: open(os.environ['TRAIN_EVENT'],'w').write(json.dumps(sys.argv[2:]))
''')
        fake.chmod(0o755)
        result = self.execute(['bash', 'scripts/train_queue.sh', '1', str(out.parent), str(reference)],
                              PYTHON=str(fake), FLASH_ROOT=str(self.root), TRAIN_EVENT=str(self.work/'train.json'),
                              EXTRA_ARGS='training.total_environment_steps=349999104')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads((self.work / 'train.json').read_text())
        self.assertIn('training.resume=true', args)
        self.assertIn(f"training.checkpoint_path='{out}/state_20.safetensors'", args)
        self.assertIn('training.total_environment_steps=349999104', args)

    def make_box(self):
        workspace = self.work/'workspace'; workspace.mkdir()
        (workspace/'v2d-track-2').symlink_to(self.root, target_is_directory=True)
        (self.root/'scripts/gpu_bootstrap.sh').write_text('#!/bin/sh\nexit 0\n')
        queue = self.root/'scripts/test_day.sh'; queue.write_text('#!/bin/sh\nexit 7\n'); queue.chmod(0o755)
        python = self.root/'.venv/bin/python'; python.parent.mkdir(parents=True)
        python.write_text('#!/bin/sh\nexit 0\n'); python.chmod(0o755)
        probe = self.work/'gpu_probe'; probe.write_text('#!/bin/sh\necho GPU\\ 0:\\ fake\n'); probe.chmod(0o755)
        locker = self.work/'lock_probe'
        locker.write_text('''#!/usr/bin/env python3
import fcntl,sys
try: fcntl.flock(int(sys.argv[-1]),fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: sys.exit(1)
'''); locker.chmod(0o755)
        (workspace/'batch.post.sh').write_text(f'#!/bin/sh\necho ran > "{self.work}/post-ran"\n')
        box = workspace/'batch.sh'
        box.write_text((self.root/'scripts/vast/box.sh').read_text().replace('/workspace',str(workspace))
                       .replace('nvidia-smi',str(probe)).replace('flock',str(locker)).replace('sleep 60','sleep 1'))
        return box

    def test_failed_batch_still_runs_post_and_reports_failure(self):
        box=self.make_box()
        result = self.execute(['bash',str(box)])
        self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
        self.assertEqual((self.work/'post-ran').read_text().strip(), 'ran')
        self.assertIn('box failed ',result.stdout)
        self.assertNotIn('box done ',result.stdout)

    def test_duplicate_box_cannot_replace_active_writer_pid(self):
        # Build the same harmless box as the status test, then hold one queue invocation open.
        box=self.make_box()
        queue=self.root/'scripts/test_day.sh'
        queue.write_text(f'#!/bin/sh\necho ready > "{self.work}/box-ready"\nsleep 1\nexit 0\n')
        first=subprocess.Popen(['bash',str(box)],env=self.env,cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            deadline=time.monotonic()+5
            while not (self.work/'box-ready').exists() and time.monotonic()<deadline: time.sleep(.02)
            self.assertTrue((self.work/'box-ready').exists())
            pidfile=self.work/'workspace/batch.pid'
            pid_before=pidfile.read_text() if pidfile.exists() else None
            second=self.execute(['bash',str(box)])
            self.assertEqual(second.returncode,2,second.stdout+second.stderr)
            self.assertEqual(pidfile.read_text(),pid_before)
            self.assertEqual(first.wait(timeout=5),0)
        finally:
            if first.poll() is None: first.kill(); first.wait()
            first.communicate()



class EvaluationTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / 'tests'))
        from test_score_rollout import PilotQueueTest, SEQUENCES
        self.helper, self.sequences = PilotQueueTest(), SEQUENCES
        self.temp = tempfile.TemporaryDirectory(prefix='eval-', dir=os.environ.get('BUG_RUNNER_TMP', '/tmp'))
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()
        manifest = self.helper.manifest(self.work)
        events, env = self.helper.fake_python(self.work)
        original_fake = Path(env['PYTHON'])
        # The fake evaluator can falsely return success without writing either output.
        original_fake.write_text(original_fake.read_text().replace("args = dict(a.split('=', 1)",
            "if os.environ.get('NO_OUTPUT') and sys.argv[1].endswith('evaluate_policy.py'): sys.exit(0)\nargs = dict(a.split('=', 1)"))
        self.env = env
        self.command = [str(REPO/'scripts/pilot_queue.sh'), '--arms', str(REPO/'scripts/pilot_arms.txt'),
                        '--manifest', str(manifest), '--arm', 'E1-control', '--sequences', self.sequences[1],
                        '--out-root', str(self.work/'runs'), '--steps', '49999872']
        initial = subprocess.run(self.command, env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(initial.returncode, 0, initial.stderr)

    def test_old_outputs_cannot_certify_new_checkpoint(self):
        checkpoint = next((self.work/'runs').rglob('policy_*.safetensors'))
        checkpoint.write_text(checkpoint.read_text()+'\n')
        result = subprocess.run(self.command, env={**self.env, 'NO_OUTPUT':'1'}, capture_output=True,
                                text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(list((self.work/'runs').rglob('evaluation.done.json')), [])

    def test_torn_done_marker_retries_evaluation(self):
        marker = next((self.work/'runs').rglob('evaluation.done.json'))
        marker.write_text('{')
        result = subprocess.run(self.command, env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('checkpoint_sha256', json.loads(marker.read_text()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
