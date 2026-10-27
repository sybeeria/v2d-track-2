"""Provider refusals and runbook rearming through local subprocess boundaries only."""
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
import tests.test_runner_regressions as runner


class WatchdogProvider(unittest.TestCase):
    def fixture(self):
        f = runner.WatchdogTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        return f

    def test_refused_destroy_stops_once_vast_positively_reports_missing(self):
        f = self.fixture()
        ticks = f.destroy_ticks()
        result = f.watchdog(DESTROY_FAILURES='1', DESTROY_LOOKUP='0=missing')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(ticks(), [0])
        reason = 'instance 123 no longer exists at Vast (finished)\n'
        self.assertEqual((f.run/'STOPPED').read_text(), reason)
        log = (f.run/'watchdog.log').read_text()
        self.assertIn(reason, log)
        self.assertNotIn('destroy failed', log)
        self.assertNotIn('destroy requested', log)
        self.assertEqual([e['kind'] for e in f.events()].count('instance'), 2)

    def test_failed_absence_checks_retry_until_a_positive_missing_result(self):
        for reply in ('api-error', 'empty', 'malformed', 'null', 'error-json', 'missing-nonzero', 'timeout'):
            with self.subTest(reply=reply):
                f = self.fixture()
                ticks = f.destroy_ticks()
                result = f.watchdog(DESTROY_401S='2', DESTROY_LOOKUP=f'0={reply} 600=missing',
                                    WATCHDOG_COMMAND_TIMEOUT_SECONDS='1')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(ticks(), [0, 600])
                self.assertEqual((f.run/'STOPPED').read_text(), 'instance 123 no longer exists at Vast (finished)\n')
                log = (f.run/'watchdog.log').read_text()
                self.assertEqual(log.count('destroy failed (finished); retrying next cycle'), 1)
                self.assertNotIn('destroy requested', log)
                self.assertEqual([e['kind'] for e in f.events()].count('quiesce'), 1)
                self.assertIn((100000, 5400), f.arming())


class MetadataCustody(unittest.TestCase):
    fixture = WatchdogProvider.fixture
    def observe(self, f, *, cap='0', **env):
        p = subprocess.Popen(['bash', 'scripts/vast/watchdog.sh', str(f.run), cap, '1'], cwd=f.root,
                             env={**f.env, **env}, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
        try:
            end = time.monotonic() + 15
            while p.poll() is None and time.monotonic() < end:
                clock = f.work/'clock'
                if clock.exists() and int(clock.read_text()) >= 130: break
                time.sleep(.03)
        finally:
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            out, err = p.communicate()
        return p.returncode, out, err

    def test_unreadable_metadata_stops_within_three_attempts_and_backs_up_before_destroy(self):
        f = self.fixture()
        ticks = f.destroy_ticks()
        code, _, err = self.observe(f, BAD_INFO='1', STATE='running')
        self.assertEqual(code, 0, err)
        self.assertEqual(ticks(), [10])
        self.assertEqual([e['tick'] for e in f.events() if e['kind']=='stop'], [10])
        self.assertEqual((f.run/'STOPPED').read_text(), 'instance metadata unavailable\n')
        self.assertEqual((f.run/'pulled/batch/state_20.safetensors').read_text(), 'owned fixture state_20.safetensors')
        self.assertEqual((f.run/'deadman-arm-123').read_text(), '100010 5400\n')
        self.assertEqual(set(f.arming()), {(100010, 5400)})
        self.assertIn('cat > /workspace/', next(e['cmd'] for e in f.events() if e['kind']=='arm'))
        self.assertEqual(next(e['script'] for e in f.events() if e['kind']=='arm'),
                         (f.root/'scripts/vast/deadman.sh').read_text())
        self.assertNotIn('watchdog start ', (f.run/'watchdog.log').read_text())

    def test_no_ssh_stops_through_provider_and_keeps_unbacked_work(self):
        f = self.fixture()
        code, _, _ = self.observe(f, BAD_INFO='1', UNREACHABLE='1')
        self.assertEqual(code, -signal.SIGKILL, 'controller exited while the disk still needed recovery')
        stops = [e['tick'] for e in f.events() if e['kind']=='stop']
        self.assertTrue(stops)
        self.assertEqual(stops[0], 10)
        self.assertEqual([e for e in f.events() if e['kind']=='destroy'], [])
        self.assertFalse((f.run/'STOPPED').exists())
        self.assertFalse((f.run/'pull.success.json').exists())
        self.assertEqual((f.work/'remote/state_20.safetensors').read_text(), 'owned fixture state_20.safetensors')
        self.assertIn('stop requested for instance 123 (instance metadata unavailable)', (f.run/'watchdog.log').read_text())

    def test_auth_failures_keep_retrying_without_false_stop_or_destroy_receipts(self):
        f = self.fixture()
        code, _, _ = self.observe(f, BAD_INFO='1', UNREACHABLE='1', STOP_401='1', DESTROY_401S='999')
        self.assertEqual(code, -signal.SIGKILL, 'controller exited after a refused stop')
        stops = [e['tick'] for e in f.events() if e['kind']=='stop']
        self.assertGreaterEqual(len(stops), 2)
        self.assertEqual(stops[:2], [10, 70])
        self.assertFalse((f.run/'STOPPED').exists())
        log = (f.run/'watchdog.log').read_text()
        self.assertIn('stop failed (instance metadata unavailable)', log)
        self.assertNotIn('stop requested', log)
        self.assertNotIn('watchdog start ', log)

    def test_restart_reuses_bound_metadata_and_enforces_original_rental_cap(self):
        f = self.fixture()
        (f.run/'instance-metadata.json').write_text(json.dumps({
            'instance':'123', 'start_date':time.time()-7200, 'dph_total':1.2}))
        ticks = f.destroy_ticks()
        code, _, err = self.observe(f, cap='1', BAD_INFO='1', STATE='running')
        self.assertEqual(code, 0, err)
        self.assertEqual(ticks(), [0])
        self.assertEqual([e for e in f.events() if e['kind']=='instance'], [])
        self.assertEqual((f.run/'STOPPED').read_text(), 'budget cap $1\n')
        self.assertIn('watchdog start id=123 rate=1.2 cap=1 floor=1', (f.run/'watchdog.log').read_text())

    def test_cached_metadata_still_enforces_the_credit_floor(self):
        f = self.fixture()
        (f.run/'instance-metadata.json').write_text(json.dumps({
            'instance':'123', 'start_date':time.time(), 'dph_total':1.2}))
        code, _, err = self.observe(f, cap='100', BAD_INFO='1', CREDIT='.996')
        self.assertEqual(code, 0, err)
        self.assertEqual([e for e in f.events() if e['kind']=='instance'], [])
        self.assertEqual((f.run/'STOPPED').read_text(), 'credit $0.996 below $1\n')

    def test_nonzero_provider_exit_cannot_certify_its_metadata(self):
        f = self.fixture()
        ticks = f.destroy_ticks()
        code, _, err = self.observe(f, INFO_NONZERO='1')
        self.assertEqual(code, 0, err)
        self.assertEqual(ticks(), [10])
        self.assertEqual((f.run/'STOPPED').read_text(), 'instance metadata unavailable\n')
        self.assertFalse((f.run/'instance-metadata.json').exists())
        self.assertNotIn('watchdog start ', (f.run/'watchdog.log').read_text())

    def test_replacement_cannot_reuse_another_instances_metadata(self):
        f = self.fixture()
        (f.run/'instance-metadata.json').write_text(json.dumps({
            'instance':'122', 'start_date':time.time()-7200, 'dph_total':1.2}))
        ticks = f.destroy_ticks()
        code, _, err = self.observe(f, BAD_INFO='1')
        self.assertEqual(code, 0, err)
        self.assertEqual(ticks(), [10])
        self.assertEqual((f.run/'STOPPED').read_text(), 'instance metadata unavailable\n')
        self.assertNotIn('watchdog start ', (f.run/'watchdog.log').read_text())


class RunbookLimits(unittest.TestCase):
    def test_morning_initial_launch_arms_before_upload(self):
        text = (runner.REPO/'data/brainstorm6/PROD-READY/MORNING.md').read_text()
        command = next(line for line in text.splitlines() if line.startswith('bash scripts/vast/'))
        with tempfile.TemporaryDirectory(prefix='watchdog-card-') as work:
            result = subprocess.run(['/bin/bash', '-c',
                'P=data/brainstorm6/PROD-READY; ID_CANARY=123\nbash() { printf "%s\\n" "$@"; }\n'+command],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, 'scripts/vast/start_run.sh\ndata/brainstorm6/PROD-READY/runs/canary\n123\n16\n')

    def test_launchers_use_the_latest_logged_limits(self):
        for name, marker in (('CANARY.md', 'PYCTRL'), ('MORNING.md', 'PY')):
            with self.subTest(runbook=name), tempfile.TemporaryDirectory(prefix='watchdog-card-') as work:
                run = Path(work).resolve()
                (run/'watchdog.log').write_text(
                    '2026-10-26T01:00:00Z watchdog start id=123 rate=1.26 cap=12 floor=1\n'
                    '2026-10-27T01:00:00Z watchdog start id=123 rate=1.26 cap=16 floor=4\n'
                    '2026-10-27T01:10:00Z state=running spend=$1 credit=$8\n')
                text = (runner.REPO/'data/brainstorm6/PROD-READY'/name).read_text()
                if name == 'CANARY.md': text = text.split('start_controller() {', 1)[1]
                code = re.search(r"<<'"+marker+r"'\n(.*?)\n"+marker, text, re.S)[1]
                boundary = '''import json,subprocess,types
from pathlib import Path
def record_launch(args, **kwargs):
    Path('launch.json').write_text(json.dumps({'args':args,'detached':kwargs['start_new_session']}))
    return types.SimpleNamespace(pid=2468)
subprocess.Popen=record_launch
'''
                result = subprocess.run([sys.executable, '-', str(run), 'controller-watchdog.pid'],
                                        input=boundary+code, text=True, capture_output=True, cwd=run)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads((run/'launch.json').read_text()), {
                    'args':['caffeinate','-i','bash','scripts/vast/watchdog.sh',str(run),'16','4'], 'detached':True})
                self.assertEqual((run/'controller-watchdog.pid').read_text(), '2468\n')
                self.assertEqual(result.stdout, 'watchdog 2468: cap $16, credit floor $4\n')

    def test_canary_envelope_matches_its_replacement_command(self):
        text = (runner.REPO/'data/brainstorm6/PROD-READY/CANARY.md').read_text()
        self.assertIn('Cap $16 for the canary/replacement combined', text.splitlines()[-1])
        self.assertIn('bash scripts/vast/start_run.sh "$D" "$NEW_ID" 16', text)


if __name__ == '__main__':
    unittest.main()
