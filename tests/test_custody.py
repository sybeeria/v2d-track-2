"""Offline public-process custody regressions. Point CUSTODY_REPO at a disposable copy."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import psutil

ROOT = Path(os.environ.get('CUSTODY_REPO', Path(__file__).resolve().parents[1])).resolve()
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import tests.test_runner_regressions as runner
import tests.test_tier3_morning as morning
from tests.prod_ready.fixture import Fixture


def until(predicate, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.03)
    return False


def alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def stop_pid(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for proc in [*reversed(children), parent]:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs([parent, *children], timeout=2)


class Startup(unittest.TestCase):
    def setUp(self):
        self.h = runner.WatchdogTests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.addCleanup(self.stop_controllers)
        self.h.env.update(STATE='running', GROWING='1')
        # Hold at the public cycle boundary, after the real watchdog's first checked pull.
        pause = self.h.bin / 'fake_pause'
        pause.write_text(pause.read_text().replace("elif tool=='fake_pause':", "elif tool=='fake_pause':\n    if args[0]=='600':\n        (root/'cycle').touch()\n        while not (root/'continue-cycle').exists(): time.sleep(.05)\n        (root/'continue-cycle').unlink()"))
        (self.h.run / 'entry.sh').write_text('exit 0\n')
        # The wrapper now performs the provider bind itself; replace only that executable boundary.
        start = self.h.root / 'scripts/vast/start_run.sh'
        start.write_text(start.read_text().replace("['vastai',", "['fake_provider',"))

    def stop_controllers(self):
        for record in self.h.work.rglob('controller-watchdog.pid'):
            stop_pid(int(record.read_text()))

    def start(self, run=None, cap='7', **env):
        h = self.h
        return h.execute(['bash', 'scripts/vast/start_run.sh', str(run or h.run), '123', cap, '1'], **env)

    def assert_guard(self, run=None):
        run = run or self.h.run
        record = run / 'controller-watchdog.pid'
        self.assertTrue(record.exists(), 'paid instance has no recorded watchdog after launch failure')
        pid = int(record.read_text())
        self.assertTrue(until(lambda: (self.h.work / 'cycle').exists()))
        self.assertTrue(alive(pid), 'wrapper reported a watchdog that already exited')
        self.assertIn('watchdog start id=123', (run / 'watchdog.log').read_text())

    def test_failed_upload_leaves_a_live_financial_guard(self):
        sync = self.h.bin / 'fake_sync'
        sync.write_text(sync.read_text().replace("event('sync',args=args)", "event('sync',args=args)\n    if args[-1].startswith('root@'): sys.exit(23)"))
        result = self.start()
        print('FAILED_UPLOAD:', result.returncode, 'watchdog_record=', (self.h.run / 'controller-watchdog.pid').exists(), flush=True)
        self.assertNotEqual(result.returncode, 0)
        self.assert_guard()
        self.assertNotIn('launch', [e['kind'] for e in self.h.events()])

    def test_retry_after_failed_upload_reuses_the_live_guard(self):
        sync = self.h.bin / 'fake_sync'
        sync.write_text(sync.read_text().replace("event('sync',args=args)", "event('sync',args=args)\n    if args[-1].startswith('root@') and not (root/'upload-failed').exists():\n        (root/'upload-failed').touch(); sys.exit(23)"))
        first = self.start()
        self.assertNotEqual(first.returncode, 0)
        self.assert_guard()
        pid = (self.h.run / 'controller-watchdog.pid').read_text()
        second = self.start()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assert_guard()
        self.assertEqual((self.h.run / 'controller-watchdog.pid').read_text(), pid)
        self.assertEqual([e['kind'] for e in self.h.events()].count('launch'), 1)

    def test_container_not_ready_is_guarded_and_same_guard_survives_retry(self):
        provider = self.h.bin / 'fake_provider'
        provider.write_text(provider.read_text().replace("if os.environ.get('BAD_INFO'):", "if os.environ.get('NO_ENDPOINT'): data.update(public_ipaddr=None, ports={})\n        if os.environ.get('BAD_INFO'):"))
        first = self.start(NO_ENDPOINT='1')
        self.assertNotEqual(first.returncode, 0)
        self.assert_guard()
        pid = (self.h.run / 'controller-watchdog.pid').read_text()
        self.assertEqual((self.h.run / 'host').read_text().split(), ['123'])
        second = self.start()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assert_guard()
        self.assertEqual((self.h.run / 'controller-watchdog.pid').read_text(), pid)
        self.assertEqual([e['kind'] for e in self.h.events()].count('launch'), 1)
        (self.h.work / 'continue-cycle').touch()
        self.assertTrue(until(lambda: (self.h.run / 'pull.success.json').exists()))
        self.assertEqual((self.h.run / 'pulled/batch/policy_20.safetensors').read_text(), 'owned fixture policy_20.safetensors')

    def test_transient_watchdog_metadata_failure_does_not_silently_disarm(self):
        provider = self.h.bin / 'fake_provider'
        code = provider.read_text().replace("if args[:2]==['show','instance']:", "if args[:2]==['show','instance']:\n        counter=root/'instance-reads'; n=int(counter.read_text())+1 if counter.exists() else 1; counter.write_text(str(n))\n        if n==2: sys.exit(29)")
        provider.write_text(code)
        result = self.start()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # Observe the actual child, not Popen's successful return or the PID receipt.
        self.assert_guard()
        print('TRANSIENT_METADATA: watchdog live, instance_reads=', (self.h.work / 'instance-reads').read_text(), flush=True)

    def test_missing_ssh_endpoint_still_enforces_the_instance_cap(self):
        (self.h.run / 'host').write_text('123  \n')
        result = self.h.watchdog(cap='0')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([e['kind'] for e in self.h.events()].count('destroy'), 1)
        self.assertIn('budget cap', (self.h.run / 'STOPPED').read_text())
        self.assertTrue((self.h.run / 'BACKUP_FAILED').exists())

    def test_invalid_cap_is_rejected_before_any_remote_writer(self):
        result = self.start(cap='not-a-number')
        launched = [e for e in self.h.events() if e['kind'] == 'launch']
        print('INVALID_CAP:', result.returncode, 'remote_launches=', len(launched), flush=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(launched, [])

    def test_same_run_cannot_start_twice(self):
        first = self.start()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assert_guard()
        record = (self.h.run / 'controller-watchdog.pid').read_text()
        second = self.start()
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual((self.h.run / 'controller-watchdog.pid').read_text(), record)
        self.assertEqual([e['kind'] for e in self.h.events()].count('launch'), 1)

    def test_one_instance_cannot_be_owned_by_two_run_directories(self):
        first = self.start()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assert_guard()
        other = self.h.work / 'other'; other.mkdir()
        (other / 'entry.sh').write_text('exit 0\n')
        second = self.start(run=other)
        print('SAME_INSTANCE_TWO_RUNS:', second.returncode, 'launches=', [e['kind'] for e in self.h.events()].count('launch'), flush=True)
        self.assertNotEqual(second.returncode, 0, 'second run started a writer on the already-owned paid box')
        self.assertEqual([e['kind'] for e in self.h.events()].count('launch'), 1)


    def test_raw_launcher_also_refuses_an_owned_instance(self):
        first = self.start(); self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assert_guard()
        other = self.h.work / 'other'; other.mkdir()
        (other / 'entry.sh').write_text('exit 0\n')
        result = self.h.execute(['bash', 'scripts/vast/launch.sh', str(other), '123'])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([e['kind'] for e in self.h.events()].count('launch'), 1)

    def test_raw_watchdog_also_refuses_an_owned_instance(self):
        first = self.start(); self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assert_guard()
        other = self.h.work / 'other'; other.mkdir()
        (other / 'host').write_text('123 offline.invalid 9\n')
        p = subprocess.Popen(['bash', 'scripts/vast/watchdog.sh', str(other), '7', '1'], cwd=self.h.root,
            env=self.h.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.addCleanup(stop_pid, p.pid)
        until(lambda: p.poll() is not None or (other / 'watchdog.log').exists())
        self.assertIsNotNone(p.poll(), 'second raw watchdog started controlling the owned instance')
        self.assertNotEqual(p.returncode, 0)
        p.wait()

    def test_controller_crash_does_not_erase_the_instance_binding(self):
        first = self.start(); self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assert_guard()
        pid = int((self.h.run / 'controller-watchdog.pid').read_text())
        stop_pid(pid)
        other = self.h.work / 'other'; other.mkdir()
        (other / 'entry.sh').write_text('exit 0\n')
        result = self.start(run=other)
        self.assertNotEqual(result.returncode, 0, 'controller death made the unfinished instance appear unowned')
        self.assertEqual([e['kind'] for e in self.h.events()].count('launch'), 1)

    def test_guarded_launch_preserves_a_stop_that_races_the_upload(self):
        (self.h.run / 'STOPPED').write_text('budget cap\n')
        result = self.h.execute(['bash', 'scripts/vast/launch.sh', str(self.h.run), '123'], VAST_GUARDED_LAUNCH='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.h.run / 'STOPPED').read_text(), 'budget cap\n')
        self.assertNotIn('launch', [e['kind'] for e in self.h.events()])


class TransientTransport(unittest.TestCase):
    def test_api_and_ssh_errors_are_unknown_until_a_real_terminal_result(self):
        h = runner.WatchdogTests(); h.setUp(); self.addCleanup(h.doCleanups)
        provider = h.bin / 'fake_provider'
        provider.write_text(provider.read_text().replace("elif args[:2]==['show','user']:", "elif args[:2]==['show','user']:\n        if not (root/'credit-error').exists(): (root/'credit-error').touch(); sys.exit(29)").replace("event('destroy')", "event('destroy',tick=tick)"))
        remote = h.bin / 'fake_remote'
        remote.write_text(remote.read_text().replace("elif 'then echo done' in cmd:", "elif 'then echo done' in cmd:\n        if not (root/'state-error').exists(): (root/'state-error').touch(); sys.exit(255)"))
        result = h.watchdog(STATE='running', GROWING='1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        requests = [e for e in h.events() if e['kind'] == 'destroy']
        self.assertEqual([e['tick'] for e in requests], [4800])
        self.assertEqual((h.run / 'STOPPED').read_text().strip(), 'finished')
        self.assertIn('state=unreachable', (h.run / 'watchdog.log').read_text())
        print('API_SSH_TRANSIENTS: unknown on first cycle; finished destruction only at t=4800', flush=True)


class CanaryReceipt(unittest.TestCase):
    def test_first_pull_without_canary_waits_for_a_complete_canary(self):
        h = morning.MorningTests()
        h.setUp(); self.addCleanup(h.doCleanups)
        watch = h.project / 'scripts/vast/watchdog.sh'
        code = watch.read_text().replace("time.sleep(.15)", "(root/'fake-watchdog.pid').write_text(str(os.getpid()))\n(root/'pull.success.json').write_text(json.dumps({'canary':None}))\ntime.sleep(.5)")
        watch.write_text(code)
        def cleanup():
            record = h.run / 'fake-watchdog.pid'
            if record.exists(): stop_pid(int(record.read_text()))
        self.addCleanup(cleanup)
        result = h.invoke()
        print('NULL_CANARY_RECEIPT: exit=', result.returncode, 'GO=', (h.run / 'GO.json').exists(), flush=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((h.run / 'native_status.json').read_text()), {'0':'SUCCESS'})


class LostLaunchReply(unittest.TestCase):
    def test_retry_keeps_original_writer_and_log_after_lost_launch_reply(self):
        with tempfile.TemporaryDirectory(prefix='custody-ack-', dir=os.environ.get('BUG_RUNNER_TMP', os.environ.get('TMPDIR', '/tmp'))) as tmp:
            h = Fixture(Path(tmp))
            try:
                run = Path(tmp) / 'native'; run.mkdir()
                (run / 'jobs.txt').write_text('unused\n')
                (run / 'entry.sh').write_text('sleep 30\n')
                remote = Path(tmp) / 'bin/fake_remote'
                remote.write_text(remote.read_text().replace("cmd=remap(args[-1]);event('remote',command=cmd)", "cmd=remap(args[-1]);event('remote',command=cmd); print('Welcome fixture banner', flush=True)").replace("print('launched')", "print('launched')\n        if not (home/'lost-reply').exists(): (home/'lost-reply').touch(); sys.exit(55)"))
                args = ['bash', str(h.controller / 'scripts/vast/launch.sh'), str(run), '123']
                first = h.command(args)
                self.assertNotEqual(first.returncode, 0)
                self.assertTrue(until(lambda: (h.workspace / 'native.pid').exists()))
                pid = int((h.workspace / 'native.pid').read_text())
                self.assertTrue(until(lambda: 'box start ' in (h.workspace / 'native.log').read_text()))
                old = (h.workspace / 'native.log').read_bytes()
                second = h.command(args)
                self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
                groups = (h.home / 'box-processes').read_text().split()
                print('LOST_LAUNCH_REPLY: first_exit=', first.returncode, 'second_exit=', second.returncode, 'writers=', len(groups), flush=True)
                self.assertEqual(len(groups), 1, 'retry launched a second box and truncated the first writer log')
                self.assertTrue(alive(pid))
                self.assertEqual((h.workspace / 'native.log').read_bytes(), old)
            finally:
                h.stop_box()



class OfflineBackstop(unittest.TestCase):
    def test_deadman_arm_does_not_start_or_truncate_another_box(self):
        with tempfile.TemporaryDirectory(prefix='custody-transport-', dir=os.environ.get('BUG_RUNNER_TMP', os.environ.get('TMPDIR', '/tmp'))) as tmp:
            h = Fixture(Path(tmp)); self.addCleanup(h.stop_box)
            h.workspace.mkdir(exist_ok=True)
            (h.workspace / 'batch.jobs').write_text('unused\n')
            (h.workspace / 'batch.sh').write_text('sleep 30\n')
            log = h.workspace / 'batch.log'; log.write_text('box start ORIGINAL\n')
            result = h.command([str(Path(tmp) / 'bin/fake_remote'), 'root@offline.invalid',
                'echo 9999999999 5400 > /workspace/batch.deadline.tmp && mv /workspace/batch.deadline.tmp /workspace/batch.deadline && { setsid nohup bash /workspace/v2d-track-2/scripts/vast/deadman.sh batch >> /workspace/batch.deadman.log 2>&1 < /dev/null & }'])
            groups = (h.home / 'box-processes').read_text() if (h.home / 'box-processes').exists() else ''
            for pid in map(int, groups.split()): self.addCleanup(stop_pid, pid)
            print('FAKE_DEADMAN_ARM: new_box_groups=', groups.split(), 'log=', repr(log.read_text()), flush=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(log.read_text(), 'box start ORIGINAL\n')
            self.assertEqual(groups, '')


class OfflineIdentity(unittest.TestCase):
    def test_launch_uses_the_requested_script_instead_of_the_first_job_file(self):
        with tempfile.TemporaryDirectory(prefix='custody-name-', dir=os.environ.get('BUG_RUNNER_TMP', '/tmp')) as tmp:
            h = Fixture(Path(tmp))
            try:
                (h.workspace / 'first.jobs').write_text('unused\n')
                for name in ('first', 'second'):
                    (h.workspace / f'{name}.sh').write_text(f'echo {name} > {h.home}/selected\nsleep 30\n')
                result = h.command([str(Path(tmp) / 'bin/fake_remote'), 'root@offline.invalid', 'setsid nohup bash /workspace/second.sh > /workspace/second.log 2>&1 < /dev/null & echo launched'])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(until(lambda: (h.home / 'selected').exists()))
                self.assertEqual((h.home / 'selected').read_text(), 'second\n')
            finally:
                h.stop_box()

    def test_native_script_needs_no_training_job_file(self):
        with tempfile.TemporaryDirectory(prefix='custody-native-', dir=os.environ.get('BUG_RUNNER_TMP', '/tmp')) as tmp:
            h = Fixture(Path(tmp))
            try:
                (h.workspace / 'native.sh').write_text(f'echo native > {h.home}/selected\nsleep 30\n')
                result = h.command([str(Path(tmp) / 'bin/fake_remote'), 'root@offline.invalid', 'setsid nohup bash /workspace/native.sh > /workspace/native.log 2>&1 < /dev/null & echo launched'])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(until(lambda: (h.home / 'selected').exists()))
                self.assertEqual((h.home / 'selected').read_text(), 'native\n')
            finally:
                h.stop_box()

    def test_quiesce_targets_its_pid_file_instead_of_the_last_launched_box(self):
        with tempfile.TemporaryDirectory(prefix='custody-stop-', dir=os.environ.get('BUG_RUNNER_TMP', '/tmp')) as tmp:
            h = Fixture(Path(tmp)); groups = []
            try:
                for name in ('second', 'first'):
                    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
                    groups.append(p); (h.workspace / f'{name}.pid').write_text(str(p.pid))
                (h.home / 'box-process').write_text(str(groups[-1].pid))
                (h.home / 'box-processes').write_text(''.join(str(p.pid)+'\n' for p in groups))
                result = h.command([str(Path(tmp) / 'bin/fake_remote'), 'root@offline.invalid', 'read -r pid < /workspace/second.pid || exit 1; kill -TERM -- -$pid'])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(until(lambda: groups[0].poll() is not None), 'requested second writer stayed alive')
                self.assertIsNone(groups[1].poll(), 'unrelated last-launched writer was stopped')
            finally:
                # These two direct child processes are ours to reap, unlike orphaned fake box groups.
                for p in groups:
                    if p.poll() is None: p.kill()
                    p.wait(timeout=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
