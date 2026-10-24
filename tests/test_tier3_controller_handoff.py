"""The real morning CLI must bound a stalled first-episode approval transfer."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest

from tests import test_tier3_morning as fixtures


class ControllerHandoff(unittest.TestCase):
    def test_provider_stall_fails_before_watchdog_with_an_explicit_unarmed_error(self):
        fixture = fixtures.MorningTests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.provider.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(20)\n')
        script = fixture.project / 'scripts/tier3_native/morning.py'
        script.write_text(script.read_text().replace('timeout=20)', 'timeout=.2)'))
        command = [sys.executable, '-B', str(script), str(fixture.run), '123', '--cap', '2',
                   '--credit-floor', '1', '--hourly-rate-ceiling', '.8', '--gpu-memory-mib', '48000',
                   '--confirm-paid-reconstruction']
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, start_new_session=True)
        try:
            try:
                stdout, stderr = child.communicate(timeout=4)
            except subprocess.TimeoutExpired:
                self.fail('provider lookup blocks indefinitely before any watchdog is armed')
            self.assertNotEqual(child.returncode, 0, stdout + stderr)
            self.assertIn('before watchdog', stderr)
            self.assertFalse((fixture.run / 'watchdog.log').exists())
            self.assertFalse((fixture.run / 'host').exists())
            self.assertFalse((fixture.run / 'GO.json').exists())
        finally:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            child.wait(timeout=5)
            child.communicate()

    def test_stalled_go_transfer_exits_and_retains_watchdog_evidence(self):
        fixture = fixtures.MorningTests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        sync = fixture.base / 'fake_sync'
        sync.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(20)\n')
        script = fixture.project / 'scripts/tier3_native/morning.py'
        # Tighten only this owned fake-transport deadline; the delivered production timeout remains 120s.
        script.write_text(script.read_text().replace('timeout=120)', 'timeout=.2)'))
        command = [sys.executable, '-B', str(script), str(fixture.run), '123', '--cap', '2',
                   '--credit-floor', '1', '--hourly-rate-ceiling', '.8', '--gpu-memory-mib', '48000',
                   '--confirm-paid-reconstruction']
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, start_new_session=True)
        try:
            try:
                stdout, stderr = child.communicate(timeout=4)
            except subprocess.TimeoutExpired:
                self.fail('morning CLI remains blocked in approval transfer despite a finished watchdog')
            self.assertNotEqual(child.returncode, 0, stdout + stderr)
            self.assertIn('deadline', stderr)
            self.assertTrue((fixture.run / 'GO.json').exists(), 'the verified canary reached the transfer boundary')
            self.assertTrue((fixture.run / 'watchdog.log').exists())
            self.assertFalse((fixture.run / 'native_status.json').exists())
            self.assertTrue((fixture.run / 'go_transfer.metrics.json').exists())
        finally:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            child.wait(timeout=5)
            child.communicate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
