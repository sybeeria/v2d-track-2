"""Public watchdog regression; provider and remote boundaries are local fakes only.

.venv/bin/python -m unittest tests.test_box_backup_race
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

ROOT = Path(os.environ.setdefault('BOX_REPO', str(Path(__file__).resolve().parents[1]))).resolve()
os.environ['BUG_RUNNER_REPO'] = str(ROOT)
spec = importlib.util.spec_from_file_location('runner_fixture', ROOT / 'tests/test_runner_regressions.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class BackupRace(unittest.TestCase):
    def test_first_pull_keeps_actor_and_state_committed_between_the_two_syncs(self):
        fixture = base.WatchdogTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        # Execute the production selector, not the existing fixture's simplified state-only model.
        old = '''        compile(cmd.split("<<'PYSTATES'\\n",1)[1].rsplit('\\nPYSTATES',1)[0],'<state selector>','exec')
        names=sorted((p for p in (root/'remote').glob('state_*.safetensors')),
                     key=lambda p:int(p.stem.split('_')[1]))
        print('\\n'.join(p.name for p in names[-2:]))'''
        new = '''        code=cmd.split("<<'PYSTATES'\\n",1)[1].rsplit('\\nPYSTATES',1)[0]
        result=subprocess.run([sys.executable,'-',str(root/'remote')],input=code,text=True,capture_output=True)
        print(result.stdout,end=''); print(result.stderr,end='',file=sys.stderr); sys.exit(result.returncode)'''
        transport = base.TRANSPORT.replace(old, new)
        self.assertNotEqual(transport, base.TRANSPORT)
        # The upstream writer commits policy then state. Commit both after the first actor sync.
        anchor = "    elif os.environ.get('PULL_FAIL'): sys.exit(23)"
        transport = transport.replace(anchor, '''        if selection is None and not (root/'race-written').exists():
            for name in ['policy_49999872.safetensors','state_49999872.safetensors']:
                (source/name).write_text('committed final '+name)
            (root/'race-written').touch()
''' + anchor)
        transport = transport.replace("    clock.write_text(str(tick+int(args[0])))", """    clock.write_text(str(tick+int(args[0])))
    if args[0]=='600':
        (root/'first-pull-complete').touch()
        time.sleep(30)""")
        for name in ['fake_provider', 'fake_remote', 'fake_sync', 'fake_pause', 'fake_date']:
            (fixture.bin / name).write_text(transport)
        log = fixture.work / 'watchdog-output.log'
        with log.open('w') as output:
            process = subprocess.Popen(['bash', 'scripts/vast/watchdog.sh', str(fixture.run), '100', '1'],
                cwd=fixture.root, env={**fixture.env, 'STATE': 'running'}, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 15
            while not (fixture.work / 'first-pull-complete').exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(log.read_text())
                time.sleep(.02)
            self.assertTrue((fixture.work / 'first-pull-complete').exists(), log.read_text())
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        self.assertGreater(json.loads((fixture.run / 'pull.success.json').read_text())['completed_at_ns'], 0)
        pulled = fixture.run / 'pulled/batch'
        self.assertEqual((pulled / 'state_49999872.safetensors').read_text(),
                         'committed final state_49999872.safetensors')
        actor = pulled / 'policy_49999872.safetensors'
        self.assertTrue(actor.is_file(), 'successful pull retained final state without its final actor')
        self.assertEqual(actor.read_text(), 'committed final policy_49999872.safetensors')
        self.assertEqual([e['kind'] for e in fixture.events()].count('destroy'), 0)


if __name__ == '__main__':
    unittest.main()
