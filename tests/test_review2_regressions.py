"""REVIEW2 regressions on the paid-box custody path (watchdog, launch, deadman, restore), using the existing fakes."""
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'tests'))
import test_runner_regressions as runner
import prod_ready.test_ep18_pilot as pilot
import test_tier3_prepare as preparation


class Review2(unittest.TestCase):
    def fixture(self):
        helper = runner.WatchdogTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        return helper

    def test_timeout_kills_all_writers_before_receipt(self):
        h = self.fixture()
        sync = h.bin / 'fake_sync'
        code = sync.read_text()
        injection = '''
    if not (root/'first-transfer').exists():
        (root/'first-transfer').touch()
        child=os.fork()
        if child==0:
            import signal
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            for fd in (0,1,2):
                os.dup2(os.open(os.devnull,os.O_RDWR),fd)
            (root/'orphan.pid').write_text(str(os.getpid()))
            while True:
                if (root/'batch/pull.success.json').exists():
                    (root/'batch/pulled/batch/policy_20.safetensors').write_text('late receiver overwrite')
                    (root/'write-after-receipt').touch()
                time.sleep(.02)
        time.sleep(60)
'''
        sync.write_text(code.replace("elif tool=='fake_sync':", "elif tool=='fake_sync':" + injection))
        def cleanup():
            pidfile = h.work / 'orphan.pid'
            if pidfile.exists():
                try:
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.addCleanup(cleanup)
        result = h.watchdog(WATCHDOG_COMMAND_TIMEOUT_SECONDS='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        end = time.monotonic() + 2
        while not (h.work/'write-after-receipt').exists() and time.monotonic() < end:
            time.sleep(.02)
        receipt = (h.run/'pull.success.json').exists()
        stopped = (h.run/'STOPPED').exists()
        policy = (h.run/'pulled/batch/policy_20.safetensors').read_text()
        print(f'TIMEOUT: receipt={receipt} STOPPED={stopped} backup_bytes={policy!r}', flush=True)
        self.assertEqual(policy, 'owned fixture policy_20.safetensors')

    def test_missing_pilot_binding_refuses_upload(self):
        h = pilot.Ep18Pilot()
        h.setUp()
        self.addCleanup(h.doCleanups)
        run, _ = h.stage('review2-pilot')
        (run/'payload/runs'/run.name/'input-assets/binding.json').unlink()
        relative = Path('data/challenge/track_2/tier_1_multiview_caption/forbidden.parquet')
        forbidden = run/'payload'/relative
        forbidden.parent.mkdir(parents=True)
        forbidden.write_bytes(b'synthetic forbidden clean fixture')
        result = h.f.command(['bash', str(h.f.controller/'scripts/vast/launch.sh'), str(run), '123'])
        uploaded = (h.f.remote/relative).exists()
        print(f'PILOT: missing binding; launch_exit={result.returncode} forbidden_clean_uploaded={uploaded}', flush=True)
        self.assertFalse(uploaded, 'pilot clean/Tier 1 payload was uploaded before missing binding failed in pre.sh')

    def test_replacement_cannot_inherit_quiescence(self):
        h = self.fixture()
        p = preparation.PreparationTests()
        p.setUp()
        self.addCleanup(p.doCleanups)
        initial = p.invoke()
        self.assertEqual(initial.returncode, 0, initial.stderr)
        shutil.copytree(p.run/'payload', h.run/'payload')
        shutil.copyfile(p.run/'upload.list', h.run/'upload.list')
        self.assertEqual(h.watchdog().returncode, 0)
        old = (h.run/'QUIESCED').read_bytes()
        launched = h.execute(['bash', 'scripts/vast/launch.sh', str(h.run), '456'])
        self.assertEqual(launched.returncode, 0, launched.stderr)
        result = h.watchdog(cap='0', STATE='running', STOP_FAIL='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        stale = (h.run/'QUIESCED').exists() and (h.run/'QUIESCED').read_bytes() == old
        restored = p.invoke(p.base/'next/batch', '--restore-from', str(h.run))
        print(f'REPLACEMENT: stop_failed=True old_QUIESCED={stale} prepare_restore_exit={restored.returncode}', flush=True)
        self.assertNotEqual(restored.returncode, 0, 'public prepare CLI accepted failed replacement stop using old quiescence receipt')

    def test_deadman_does_not_stop_live_final_pull(self):
        h = self.fixture()
        # Preserve production T=1800 and grace arithmetic. Advance only external
        # fake clock/transport durations; the real watchdog/deadman make decisions.
        for tool in ('fake_sync','fake_remote'):
            path = h.bin/tool
            code = path.read_text()
            if tool == 'fake_sync':
                code = code.replace("event('sync',args=args)", "event('sync',args=args)\n    clock.write_text(str(tick+1750)); time.sleep(.15)")
            else:
                code = code.replace("event('arm', cmd=cmd)", "event('arm', cmd=cmd)\n        import re\n        grace=re.search(r'echo ([0-9]+) ([0-9]+)',cmd)[2]\n        (root/'heartbeat').write_text(str(tick))\n        (root/'batch.deadline').write_text('9999999999 '+grace+'\\n')")
                code = code.replace("elif 'PYSTATES' in cmd:", "elif 'PYSTATES' in cmd:\n        clock.write_text(str(tick+1750)); time.sleep(.15)")
            path.write_text(code)
        date = h.bin/'fake_date'
        code = date.read_text().replace("elif tool=='fake_date':", "elif tool=='fake_date':\n    if '-r' in args:\n        print(100000+int((root/'heartbeat').read_text())); sys.exit(0)")
        date.write_text(code)
        for name, body in {
            'flock': 'import fcntl,sys\nfcntl.flock(int(sys.argv[-1]),fcntl.LOCK_EX|fcntl.LOCK_NB)\n',
            'fake_pgrep': 'import sys\nsys.exit(1)\n',
            'offline_stop': 'import os\nfrom pathlib import Path\np=Path(os.environ["FAKE_ROOT"]);(p/"stop-tick").write_text((p/"clock").read_text())\n',
        }.items():
            path=h.bin/name; path.write_text('#!'+sys.executable+'\n'+body); path.chmod(0o755)
        deadman = h.work/'deadman.sh'
        source = (ROOT/'scripts/vast/deadman.sh').read_text()
        source = re.sub(r'\bdate\b', 'fake_date', source)
        source = re.sub(r'\bpgrep\b', 'fake_pgrep', source)
        source = source.replace('DEADMAN_KEY=$key python3', 'DEADMAN_KEY=$key offline_stop')
        deadman.write_text(source)
        (h.work/'heartbeat').write_text('0')
        (h.work/'batch.deadline').write_text('9999999999 8400\n')
        with (h.work/'deadman.log').open('w') as log:
            process = subprocess.Popen(['bash', str(deadman), 'batch'], env={**h.env, 'DEADMAN_WORKSPACE':str(h.work), 'DEADMAN_INTERVAL':'.02', 'CONTAINER_ID':'4242', 'CONTAINER_API_KEY':'synthetic'}, stdout=log, stderr=log)
            try:
                result=h.watchdog(WATCHDOG_COMMAND_TIMEOUT_SECONDS='1800')
                self.assertEqual(result.returncode,0,result.stderr)
                time.sleep(.15)
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
        stop_tick=int((h.work/'stop-tick').read_text()) if (h.work/'stop-tick').exists() else None
        grace = (h.work/'batch.deadline').read_text().split()[1]
        print(f'DEADMAN: T=1800 grace={grace} stop_at_elapsed={stop_tick} final_pull_complete_at_elapsed={int((h.work/"clock").read_text())}', flush=True)
        self.assertFalse((h.work/'stop-tick').exists(), 'live watchdog was in final backup; deadman requested stop anyway')


if __name__ == '__main__':
    unittest.main(verbosity=2)
