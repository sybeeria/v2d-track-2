import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/vast/sentinel.py'
spec = importlib.util.spec_from_file_location('sentinel', SCRIPT)
sentinel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sentinel)

T0 = 1_800_000_000
RUNNING = {'id': 57839714, 'actual_status': 'running', 'dph_total': .66}
# t1h5/watchdog.log, Oct 27: the CLI printed a 401 and exited 0, so the watchdog recorded a successful destroy.
BOX10 = ['Failed with error 401: Authorization Error. Your key lacks proper privileges to perform this action as it '
         'requires you to have logged in using Two Factor Authentication',
         '2026-10-27T02:52:17Z destroy requested for instance 57839714 (finished)']
API_ALERT = ('api', 'Vast API failing for 10 min (2FA login expired?): watchdogs cannot destroy; 1 instance(s) or '
                   'run(s) may be billing. Run: vastai tfa login --method-type totp --code <6 digits>')
ORPHAN = ('orphan-57839714', 'instance 57839714 (running, $0.66/h) has no live watchdog and bills unattended. '
                             'Destroy: vastai destroy instance 57839714 -y')


class Check(unittest.TestCase):
    def test_box10_false_destroy_alerts_then_flags_the_orphan_hourly(self):
        state = {}
        self.assertEqual(sentinel.check(state, T0, [RUNNING], {57839714}, {'t1h5': BOX10}),
                         [('t1h5-vast-error', 't1h5: Vast API error 401 in the watchdog log')])
        self.assertEqual(sentinel.check(state, T0 + 300, [RUNNING], set(), {}), [])
        self.assertEqual(sentinel.check(state, T0 + 1200, [RUNNING], set(), {}), [ORPHAN])
        self.assertEqual(sentinel.check(state, T0 + 1500, [RUNNING], set(), {}), [])
        self.assertEqual(sentinel.check(state, T0 + 1200 + 3600, [RUNNING], set(), {}), [ORPHAN])
        self.assertEqual(sentinel.check(state, T0 + 6000, [], set(), {}), [])
        self.assertEqual(state['orphan_since'], {})

    def test_watched_instance_and_normal_finish_stay_quiet(self):
        state = {}
        for t in range(0, 7200, 300):
            self.assertEqual(sentinel.check(state, T0 + t, [RUNNING], {57839714}, {}), [])
        self.assertEqual(sentinel.check(state, T0 + 7200, [], set(), {'t1h0': [BOX10[1]]}), [])

    def test_api_failure_alerts_only_while_instances_were_running(self):
        state = {}
        sentinel.check(state, T0, [RUNNING], {57839714}, {})
        self.assertEqual(sentinel.check(state, T0 + 300, None, {57839714}, {}), [])
        self.assertEqual(sentinel.check(state, T0 + 900, None, set(), {}), [API_ALERT])
        idle = {}
        sentinel.check(idle, T0, [], set(), {})
        sentinel.check(idle, T0 + 300, None, set(), {})
        self.assertEqual(sentinel.check(idle, T0 + 3600, None, set(), {}), [])

    def test_api_failure_before_any_inventory_alerts_while_a_run_may_bill(self):
        fresh = {}
        self.assertEqual(sentinel.check(fresh, T0, None, set(), {}, exposed=1), [])
        self.assertEqual(sentinel.check(fresh, T0 + 600, None, set(), {}, exposed=1),
                         [API_ALERT])

    def test_watchdog_failures_alert_once_per_hour(self):
        state = {}
        lines = ['2026-10-27T01:00:00Z destroy failed (finished); retrying next cycle',
                 '2026-10-27T01:00:01Z backup failed (box failed)',
                 '2026-10-27T01:00:02Z destroy requested for instance 5 (budget cap $12)']
        self.assertEqual(sentinel.check(state, T0, [], set(), {'t1h2': lines}), [
            ('t1h2-destroy-failed', 't1h2: destroy FAILED (finished); the instance is still billing'),
            ('t1h2-backup-failed', 't1h2: backup failed (box failed); the latest work may not be saved'),
            ('t1h2-early-stop', 't1h2: stopped early: budget cap $12')])
        self.assertEqual(sentinel.check(state, T0 + 600, [], set(), {'t1h2': lines[:1]}), [])

    def test_box_unreachable_for_30_min_by_log_time(self):
        state = {}
        line = '2026-10-26T{}Z state={} spend=$4.20 credit=$40 |'
        lines = [line.format(t, 'unreachable') for t in ('14:40:00', '14:50:00', '15:00:00')]
        self.assertEqual(sentinel.check(state, T0, [], set(), {'t1h1': lines}), [])
        self.assertEqual(sentinel.check(state, T0, [], set(), {'t1h1': [line.format('15:10:00', 'unreachable')]}), [
            ('t1h1-unreachable', 't1h1: box unreachable for 30 min (spend ~$4.20); the watchdog cannot stop it, '
                                 'only the cap or the box deadman can (BOX-9)')])
        later = [line.format('16:00:00', 'running'), line.format('16:10:00', 'unreachable')]
        self.assertEqual(sentinel.check(state, T0 + 7200, [], set(), {'t1h1': later}), [])


class Armed(unittest.TestCase):
    def test_only_a_watchdog_that_logged_its_start_after_taking_the_lock_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, lock = Path(tmp) / 'run', Path(tmp) / '57839714.lock'
            run.mkdir()
            lock.write_text(f'{run}\n')
            taken = 1_793_000_000  # 2026-10-26T07:33:20Z
            os.utime(lock, (taken, taken))
            self.assertFalse(sentinel.armed(lock, run))
            log = run / 'watchdog.log'
            log.write_text('2026-10-26T06:00:00Z watchdog start id=57839714 rate=0.66 cap=12 floor=4\n'
                           '2026-10-26T08:00:00Z watchdog start id=5 rate=0.66 cap=12 floor=4\n')
            self.assertFalse(sentinel.armed(lock, run))
            with log.open('a') as f:
                f.write('2026-10-26T07:33:20Z watchdog start id=57839714 rate=0.66 cap=12 floor=4\n')
            self.assertTrue(sentinel.armed(lock, run))


class Delivery(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.run_dir, self.locks, bin_dir = self.tmp / 'runs/t1h2', self.tmp / 'locks', self.tmp / 'bin'
        for d in (self.run_dir, self.locks, bin_dir):
            d.mkdir(parents=True)
        (self.locks / '57775580.lock').write_text(f'{self.run_dir}\n')
        (self.run_dir / 'STOPPED').write_text('finished\n')
        (self.run_dir / 'watchdog.log').write_text('2026-10-26T06:35:00Z watchdog start id=57775580 rate=0.789 cap=14 floor=4\n')
        (bin_dir / 'vastai').write_text('#!/bin/sh\necho "[]"\n')
        (bin_dir / 'vastai').chmod(0o755)
        self.bin = bin_dir
        self.config = self.tmp / 'telegram.env'
        self.config.write_text('TELEGRAM_BOT_TOKEN=123:SECRET \nTELEGRAM_CHAT_ID=42\n')
        for patcher in (mock.patch.dict(os.environ, {'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}'}),
                        mock.patch.object(sentinel, 'CONFIG', self.config)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def main(self, *args):
        out = io.StringIO()
        with mock.patch.object(sys, 'argv', ['sentinel.py', '--state', str(self.tmp / 'state'), '--locks',
                                             str(self.locks), *args]), contextlib.redirect_stdout(out):
            try:
                sentinel.main()
                code = 0
            except SystemExit as exit:
                code = exit.code
        return code, out.getvalue()

    def test_an_alert_that_fails_to_send_is_kept_until_delivered_once(self):
        self.assertEqual(self.main(), (0, ''))
        with (self.run_dir / 'watchdog.log').open('a') as log:
            log.write('2026-10-27T01:00:00Z backup failed (box failed)\n')
        leak = urllib.error.URLError('https://api.telegram.org/bot123:SECRET/sendMessage')
        with mock.patch.object(sentinel, 'send', side_effect=leak):
            code, out = self.main()
        self.assertEqual(code, 0)
        self.assertRegex(out, r'^\S+ send failed t1h2-backup-failed URLError $')
        delivered = []
        with mock.patch.object(sentinel, 'send', side_effect=delivered.append):
            self.main()
            self.main()
        self.assertEqual(delivered, ['\U0001F6A8 V2D t1h2: backup failed (box failed); the latest work may not be saved'])

    def test_test_message_strips_the_config_and_never_prints_the_token(self):
        urls = []

        class Reply(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def urlopen(url, data, timeout):
            urls.append(url)
            return Reply(b'{"ok": true}')

        with mock.patch.object(sentinel.urllib.request, 'urlopen', urlopen):
            self.assertEqual(self.main('--test')[0], 0)
        self.assertEqual(urls, ['https://api.telegram.org/bot123:SECRET/sendMessage'])
        with mock.patch.object(sentinel.urllib.request, 'urlopen',
                               side_effect=ValueError("URL can't contain control characters. '/bot123:SECRET /sendMessage'")):
            code, out = self.main('--test')
        self.assertEqual(code, 1)
        self.assertNotIn('SECRET', out)
        self.assertIn('send failed test ValueError', out)

    def test_hourly_cooldown_runs_from_delivery_not_queueing(self):
        line = '2026-10-27T01:00:00Z backup failed (box failed)\n'
        delivered = []

        def sweep(at, send):
            with mock.patch.object(sentinel.time, 'time', return_value=T0 + at), mock.patch.object(sentinel, 'send', send):
                self.main()

        sweep(0, delivered.append)
        for at, send, new_line in ((0, mock.Mock(side_effect=OSError), True), (3300, delivered.append, False),
                                   (3600, delivered.append, True), (6900, delivered.append, True)):
            if new_line:
                with (self.run_dir / 'watchdog.log').open('a') as log:
                    log.write(line)
            sweep(at, send)
            self.assertEqual(len(delivered), {0: 0, 3300: 1, 3600: 1, 6900: 2}[at], at)

    def test_failed_inventory_call_is_an_api_failure_even_if_it_prints_a_list(self):
        (self.bin / 'vastai').write_text('#!/bin/sh\necho "[]"\nexit 9\n')
        self.assertIsNone(sentinel.vast_instances())
        (self.bin / 'vastai').write_text('#!/bin/sh\necho "[]"\n')
        self.assertEqual(sentinel.vast_instances(), [])

    def test_test_message_is_never_a_dry_run(self):
        send = mock.Mock()
        with mock.patch.object(sentinel, 'send', send), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.main('--dry-run', '--test')[0], 2)
        send.assert_not_called()

    def test_dry_run_refuses_the_real_state(self):
        with mock.patch.object(sys, 'argv', ['sentinel.py', '--dry-run']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit:
                sentinel.main()
        self.assertEqual(exit.exception.code, 2)


class DryRun(unittest.TestCase):
    def test_new_watchdog_log_lines_are_alerted_once_and_history_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, locks, bin_dir = tmp / 'runs/t1h5', tmp / 'locks', tmp / 'bin'
            for d in (run, locks, bin_dir):
                d.mkdir(parents=True)
            (locks / '57839714.lock').write_text(f'{run}\n')
            (run / 'watchdog.log').write_text('2026-10-26T01:00:00Z destroy failed (finished); retrying next cycle\n')
            vastai = bin_dir / 'vastai'
            vastai.write_text('#!/bin/sh\necho "[]"\n')
            vastai.chmod(0o755)
            env = {**os.environ, 'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}', 'V2D_TELEGRAM_ENV': os.devnull}
            command = [sys.executable, '-I', str(SCRIPT), '--dry-run', '--state', str(tmp / 'state'), '--locks', str(locks)]

            def sweep():
                p = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(p.returncode, 0, p.stderr)
                return p.stdout.splitlines()

            self.assertEqual(sweep(), [])
            with (run / 'watchdog.log').open('a') as log:
                log.write('\n'.join(BOX10) + '\n2026-10-27T03:00:00Z partial line')
            self.assertEqual(sweep(), ['\U0001F6A8 V2D t1h5: Vast API error 401 in the watchdog log'])
            self.assertEqual(sweep(), [])
            self.assertEqual(json.loads((tmp / 'state/state.json').read_text())['instances'], [])


if __name__ == '__main__':
    unittest.main()
