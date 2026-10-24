"""scripts/vast/start_run.sh on the offline transport: one command launches the box and a detached, pulling watchdog."""
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from .fixture import Fixture, wait


class StartRun(unittest.TestCase):
    def test_one_command_launches_the_box_and_a_detached_pulling_watchdog(self):
        with tempfile.TemporaryDirectory(prefix='start-run-', dir=os.environ.get('TMPDIR', '/tmp')) as tmp:
            f = Fixture(Path(tmp)); self.addCleanup(f.stop_box)
            run, _, _ = f.run_spec(2, 't2s')
            result = f.command(['bash', str(f.controller/'scripts/vast/start_run.sh'), str(run), '123', '7', '1'])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
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
