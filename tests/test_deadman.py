"""Run the box-side deadman against a local stand-in for the Vast instance API."""
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

PROJECT = Path(__file__).resolve().parents[1]
# macOS has no flock(1); this locks the inherited descriptor the same way.
LOCKER = f'''#!{sys.executable}
import fcntl, sys
try: fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError: sys.exit(1)
'''


class Api(http.server.BaseHTTPRequestHandler):
    def do_PUT(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append((self.path, self.headers['Authorization'], body))
        self.send_response(self.server.statuses.pop(0) if self.server.statuses else 200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(self.server.bodies.pop(0) if self.server.bodies else b'{"success": true}')

    def log_message(self, *args):
        pass


class DeadmanTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='deadman-')
        self.addCleanup(temp.cleanup)
        self.ws = Path(temp.name)
        tools = self.ws / 'bin'
        tools.mkdir()
        (tools / 'flock').write_text(LOCKER)
        (tools / 'flock').chmod(0o755)
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Api)
        self.server.requests, self.server.statuses, self.server.bodies = [], [], []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.env = {**os.environ, 'PATH': f'{tools}{os.pathsep}{os.environ["PATH"]}', 'DEADMAN_WORKSPACE': str(self.ws),
                    'DEADMAN_INTERVAL': '0.1', 'VAST_API_URL': f'http://127.0.0.1:{self.server.server_port}/api/v0',
                    'CONTAINER_ID': '4242', 'CONTAINER_API_KEY': 'instance-secret'}

    def arm(self, deadline, grace):
        (self.ws / 'batch.deadline').write_text(f'{int(deadline)} {grace}\n')

    def start(self, **env):
        process = subprocess.Popen(['bash', str(PROJECT / 'scripts/vast/deadman.sh'), 'batch'], env={**self.env, **env},
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(lambda: process.poll() is not None or (process.kill(), process.communicate()))
        return process

    def box_script(self):
        (self.ws / 'batch.sh').write_text('sleep 30\n')
        box = subprocess.Popen(['bash', str(self.ws / 'batch.sh')], start_new_session=True)
        stop = lambda: box.poll() is not None or (os.killpg(box.pid, signal.SIGKILL), box.wait())
        self.addCleanup(stop)
        return stop

    def stop_request(self):
        return ('/api/v0/instances/4242/', 'Bearer instance-secret', {'state': 'stopped'})

    def test_past_deadline_stops_instance_once_without_logging_key(self):
        self.arm(time.time() - 1, 5400)
        output, _ = self.start().communicate(timeout=10)
        self.assertEqual(self.server.requests, [self.stop_request()])
        self.assertIn('stop requested (deadline', output)
        self.assertNotIn('instance-secret', output)

    def test_grace_runs_only_after_box_script_exits(self):
        stop_box = self.box_script()
        self.arm(time.time() + 3600, 1)
        deadman = self.start()
        time.sleep(2.5)
        self.assertIsNone(deadman.poll())
        self.assertEqual(self.server.requests, [])
        stop_box()
        output, _ = deadman.communicate(timeout=10)
        self.assertEqual(deadman.returncode, 0)
        self.assertEqual(self.server.requests, [self.stop_request()])
        self.assertIn('stop requested (box script not running and watchdog silent for 1s)', output)

    def test_live_watchdog_keeps_finished_box_until_it_goes_silent(self):
        self.arm(time.time() + 3600, 2)
        deadman = self.start()
        for _ in range(8):
            self.arm(time.time() + 3600, 2)
            time.sleep(0.5)
        self.assertIsNone(deadman.poll())
        self.assertEqual(self.server.requests, [])
        output, _ = deadman.communicate(timeout=10)
        self.assertEqual(self.server.requests, [self.stop_request()])
        self.assertIn('watchdog silent for 2s', output)

    def test_zero_grace_keeps_idle_box_until_deadline(self):
        self.arm(time.time() + 3, 0)
        deadman = self.start()
        time.sleep(1.5)
        self.assertEqual(self.server.requests, [])
        output, _ = deadman.communicate(timeout=10)
        self.assertEqual(self.server.requests, [self.stop_request()])
        self.assertIn('stop requested (deadline', output)

    def test_refused_stop_is_retried(self):
        self.server.statuses.append(500)
        self.arm(time.time() - 1, 5400)
        deadman = self.start()
        output, _ = deadman.communicate(timeout=10)
        self.assertEqual(deadman.returncode, 0)
        self.assertEqual(self.server.requests, [self.stop_request()] * 2)
        self.assertIn('stop failed (deadline', output)

    def test_ok_reply_without_explicit_success_is_retried(self):
        self.server.bodies.extend([b'{}', b'{"success": false, "msg": "busy"}'])
        self.arm(time.time() - 1, 5400)
        deadman = self.start()
        output, _ = deadman.communicate(timeout=10)
        self.assertEqual(deadman.returncode, 0)
        self.assertEqual(self.server.requests, [self.stop_request()] * 3)
        self.assertEqual(output.count('stop failed (deadline'), 2)

    def test_missing_instance_identity_disarms_loudly(self):
        self.arm(time.time() - 1, 5400)
        deadman = self.start(CONTAINER_ID='')
        output, _ = deadman.communicate(timeout=10)
        self.assertEqual(deadman.returncode, 1)
        self.assertIn('no CONTAINER_ID; deadman disarmed', output)
        self.assertEqual(self.server.requests, [])

    def test_rearming_does_not_start_a_second_deadman(self):
        self.arm(time.time() + 3600, 0)
        first = self.start()
        time.sleep(1)
        second = self.start()
        output, _ = second.communicate(timeout=10)
        self.assertEqual((second.returncode, output), (0, ''))
        self.assertIsNone(first.poll())


if __name__ == '__main__':
    unittest.main()
