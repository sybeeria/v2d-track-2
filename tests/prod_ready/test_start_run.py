"""scripts/vast/start_run.sh on the offline transport: one command launches the box and a detached, pulling watchdog."""
import os
import json
import signal
import subprocess
import tempfile
import time
import unittest
import psutil
from pathlib import Path
from .fixture import Fixture, wait


class StartRun(unittest.TestCase):
    def test_partial_lookup_retains_emergency_ssh_when_provider_stop_is_denied(self):
        replies=[{'start_date':None,'dph_total':None},
                 {'start_date':None,'dph_total':None,'public_ipaddr':'new.invalid','ports':{'22/tcp':[{'HostPort':'9'}]}}]
        for reply in replies:
            with self.subTest(reply=reply), tempfile.TemporaryDirectory(prefix='start-endpoint-') as work:
                f=Fixture(Path(work));run=Path(work)/'run';run.mkdir()
                (run/'host').write_text('123 known.good.invalid 8\n')
                try:
                    before=int(time.time())
                    result=f.command(['bash',str(f.controller/'scripts/vast/start_run.sh'),str(run),'123','7','1'],
                        {'TRANSPORT_INSTANCE_JSON':json.dumps(reply),'TRANSPORT_STOP_DENIED':'1'},timeout=5)
                    self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                    end=time.monotonic()+5
                    while time.monotonic()<end:
                        log=(run/'watchdog.log').read_text() if (run/'watchdog.log').exists() else ''
                        if 'stop failed (instance metadata unavailable)' in log:break
                        time.sleep(.03)
                    events=[json.loads(line) for line in (f.home/'transport.jsonl').read_text().splitlines()]
                    remotes=[e for e in events if e['kind']=='remote']
                    arms=[e for e in events if e['kind']=='arm_deadman']
                    print(f'ENDPOINT reply={reply} host={(run/"host").read_text().strip()!r} ssh_commands={len(remotes)} deadman_arms={len(arms)}',flush=True)
                    self.assertEqual((run/'host').read_text(),'123 known.good.invalid 8\n')
                    self.assertEqual(result.stderr,'cannot read instance 123 from Vast; no launch started; watchdog recovery active\n')
                    self.assertTrue(arms)
                    self.assertTrue(all(e['args'][-4:-1]==['-p','8','root@known.good.invalid'] for e in remotes))
                    deadline,grace=(run/'deadman-arm-123').read_text().split()
                    self.assertGreaterEqual(int(deadline),before)
                    self.assertLessEqual(int(deadline),int(time.time()))
                    self.assertEqual(grace,'5400')
                    self.assertIn(f'echo {deadline} 5400 >',remotes[0]['command'])
                    self.assertIn('cat >',remotes[0]['command'])
                    self.assertIn('stop failed (instance metadata unavailable)',log)
                    self.assertNotIn('stop requested',log)
                    self.assertFalse((run/'STOPPED').exists())
                    self.assertFalse((f.home/'box-processes').exists())
                finally:
                    record=run/'controller-watchdog.pid'
                    if record.exists():
                        try:
                            parent=psutil.Process(int(record.read_text()))
                            processes=[*parent.children(recursive=True),parent]
                            for process in reversed(processes):
                                try:process.kill()
                                except psutil.NoSuchProcess:pass
                            psutil.wait_procs(processes,timeout=2)
                        except psutil.NoSuchProcess:pass
                    f.stop_box()

    def test_failed_instance_lookup_is_one_line_and_keeps_a_controller_without_a_writer(self):
        for reply in ('api-error', 'empty', 'nonzero', 'malformed', 'missing', 'null', 'bad-port'):
            with self.subTest(reply=reply), tempfile.TemporaryDirectory(prefix='start-lookup-') as work:
                f = Fixture(Path(work))
                run = Path(work)/'run'; run.mkdir()
                (run/'host').write_text('123 old.invalid 8\n')
                (run/'STOPPED').write_text('previous stop\n')
                try:
                    result = f.command(['bash', str(f.controller/'scripts/vast/start_run.sh'), str(run), '123', '7', '1'],
                                       env={'TRANSPORT_INSTANCE_REPLY':reply}, timeout=5)
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, '')
                    self.assertEqual(result.stderr, 'cannot read instance 123 from Vast; no launch started; watchdog recovery active\n')
                    self.assertTrue((run/'controller-watchdog.pid').exists())
                    self.assertTrue((run/'controller-launch.pending.json').exists())
                    wait(run/'deadman-arm-123', 5)
                    events = lambda: [json.loads(line) for line in (f.home/'transport.jsonl').read_text().splitlines()]
                    if reply != 'bad-port':
                        import time
                        end = time.monotonic() + 5
                        while not any(e['kind']=='stop' for e in events()) and time.monotonic() < end: time.sleep(.03)
                        self.assertTrue(any(e['kind']=='stop' for e in events()))
                        self.assertNotIn('watchdog start ', (run/'watchdog.log').read_text())
                    else:
                        wait(run/'watchdog.log', 5)
                        self.assertIn('watchdog start id=123 rate=1.26 cap=7 floor=1', (run/'watchdog.log').read_text())
                    self.assertEqual((run/'host').read_text(), '123 old.invalid 8\n')
                    self.assertFalse((run/'STOPPED').exists())
                    self.assertFalse((f.home/'box-processes').exists())
                    self.assertEqual([e for e in events() if e['kind']=='destroy'], [])
                finally:
                    record = run/'controller-watchdog.pid'
                    if record.exists():
                        try:
                            parent = psutil.Process(int(record.read_text()))
                            processes = [*parent.children(recursive=True), parent]
                            for process in reversed(processes):
                                try: process.kill()
                                except psutil.NoSuchProcess: pass
                            psutil.wait_procs(processes, timeout=2)
                        except psutil.NoSuchProcess: pass
                    f.stop_box()

    def test_one_command_launches_the_box_and_a_detached_pulling_watchdog(self):
        tmp = tempfile.TemporaryDirectory(prefix='start-run-', dir=os.environ.get('TMPDIR', '/tmp')); self.addCleanup(tmp.cleanup)
        # Cleanups run last-in first-out: the box stops before its transport directory is removed.
        f = Fixture(Path(tmp.name)); self.addCleanup(f.stop_box)
        run, _, _ = f.run_spec(2, 't2s')
        result = f.command(['bash', str(f.controller/'scripts/vast/start_run.sh'), str(run), '123', '7', '1'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((run/'host').read_text(),'123 offline.invalid 9\n')
        metadata=json.loads((run/'instance-metadata.json').read_text())
        self.assertEqual((metadata['instance'],metadata['dph_total']),('123',1.26))
        pid = int((run/'controller-watchdog.pid').read_text())
        self.assertIn(f'watchdog {pid}: cap $7, credit floor $1', result.stdout)
        try:
            # The guard starts before upload; release its first idle cycle only after the box finishes.
            f.finished(run); wait(f.home/'periodic-pull', 30)
            args = subprocess.run(['ps', '-o', 'args=', '-p', str(pid)], capture_output=True, text=True).stdout
            self.assertIn(f'scripts/vast/watchdog.sh {run} 7 1', args)
            self.assertEqual(os.getpgid(pid), pid)
            (f.home/'continue-cycle').touch()
            wait(run/'pull.success.json', 30)
            self.assertTrue((run/'pulled'/run.name).is_dir())
            self.assertEqual(len((run/'launch-code.log').read_text().splitlines()), 1)
        finally:
            try: os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError: pass


if __name__ == '__main__':
    unittest.main()
