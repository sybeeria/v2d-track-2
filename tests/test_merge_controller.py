"""Public controller CLI regressions; every provider/SSH boundary is a named local fake."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from tests.prod_ready.fixture import Fixture
from tests import test_custody as custody
from tests import test_deadman as deadman_fixtures


class ControllerIntegration(unittest.TestCase):
    def test_new_instance_in_same_run_requires_receipt_then_gets_a_fresh_guard(self):
        h=custody.Startup();h.setUp();self.addCleanup(h.doCleanups)
        first=h.start();self.assertEqual(first.returncode,0,first.stdout+first.stderr);h.assert_guard()
        old=int((h.h.run/'controller-watchdog.pid').read_text());custody.stop_pid(old)
        command=['bash','scripts/vast/start_run.sh',str(h.h.run),'456','7','1']
        refused=h.h.execute(command)
        self.assertNotEqual(refused.returncode,0)
        self.assertIn('previous instance has no completed stop',refused.stderr)
        (h.h.run/'STOPPED').write_text('manual canary preemption: destroyed 123\n')
        (h.h.work/'cycle').unlink(missing_ok=True)
        replacement=h.h.execute(command)
        self.assertEqual(replacement.returncode,0,replacement.stdout+replacement.stderr)
        new=int((h.h.run/'controller-watchdog.pid').read_text())
        self.assertNotEqual(old,new);self.assertTrue(custody.alive(new))
        self.assertEqual((h.h.run/'host').read_text().split()[0],'456')
        self.assertTrue(custody.until(lambda:(h.h.work/'cycle').exists()))
        self.assertIn('watchdog start id=456',(h.h.run/'watchdog.log').read_text())
        self.assertEqual([e['kind'] for e in h.h.events()].count('launch'),2)
        print('NEW_INSTANCE original123 -> refused456 without receipt -> live guard456 with receipt')

    def cold_launch(self, refuse=False):
        temporary=tempfile.TemporaryDirectory(prefix='cold-deadman-',dir=os.environ['TMPDIR'])
        self.addCleanup(temporary.cleanup)
        f=Fixture(Path(temporary.name));self.addCleanup(f.stop_box)
        run,_,_=f.run_spec(2,'cold');self.addCleanup(self.stop_guard,run)
        self.assertFalse((f.remote/'scripts/vast/deadman.sh').exists())
        remote=f.work/'bin/fake_remote'
        old="    if 'scripts/vast/deadman.sh' in cmd:\n        event('arm_deadman')"
        new="""    if 'scripts/vast/deadman.sh' in cmd:
        event('arm_deadman')
        if not (workspace/'v2d-track-2/scripts/vast/deadman.sh').is_file():
            event('deadman_missing');sys.exit(1)
        if os.environ.get('B9_DEADMAN_REFUSE')=='1':
            event('deadman_refused');sys.exit(17)
        event('deadman_armed')"""
        code=remote.read_text();self.assertIn(old,code);remote.write_text(code.replace(old,new))
        # A finite upload stall fixes the possible schedule: first pre-upload cycle, then installation.
        sync=f.work/'bin/fake_sync'
        old="    target=local(dest);event('sync',sources=sources,dest=dest)"
        new="""    target=local(dest);event('sync',sources=sources,dest=dest)
    if dest.startswith('root@'):
        until=time.monotonic()+15
        while not (home/'periodic-pull').exists():
            if time.monotonic()>until:sys.exit('first pre-upload financial cycle did not finish')
            time.sleep(.01)"""
        code=sync.read_text();self.assertIn(old,code);sync.write_text(code.replace(old,new))
        result=f.command(['bash',str(f.controller/'scripts/vast/start_run.sh'),str(run),'123','7','1'],
            env={'B9_DEADMAN_REFUSE':'1' if refuse else '0'})
        events=[json.loads(x) for x in (f.home/'transport.jsonl').read_text().splitlines()]
        return f,run,result,events

    def test_fresh_guarded_launch_arms_installed_deadman_before_starting_writer(self):
        f,run,result,events=self.cold_launch()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        f.finished(run)
        kinds=[x['kind'] for x in events]
        self.assertIn('deadman_missing',kinds,'fixture must actually hit a cold-host arm failure')
        self.assertTrue((f.remote/'scripts/vast/deadman.sh').is_file())
        self.assertIn('deadman_armed',kinds,'writer admitted, but no installed box-side backstop was ever armed')
        writer=next(i for i,x in enumerate(events) if x['kind']=='remote' and 'setsid nohup' in x.get('command','') and 'deadman.sh' not in x.get('command',''))
        self.assertLess(kinds.index('deadman_armed'),writer)
        print('COLD_DEADMAN installed+armed before writer, first attempt absent as expected')

    def test_failed_box_deadman_arm_refuses_writer_and_keeps_financial_guard(self):
        f,run,result,events=self.cold_launch(refuse=True)
        self.assertNotEqual(result.returncode,0,'writer started even though its backstop refused to arm')
        self.assertIn('deadman_refused',[x['kind'] for x in events])
        self.assertFalse(any(x['kind']=='remote' and 'setsid nohup' in x.get('command','') and 'deadman.sh' not in x.get('command','') for x in events))
        self.assertTrue(custody.alive(int((run/'controller-watchdog.pid').read_text())))
        self.assertFalse((run/'STOPPED').exists())
        print('FAILED_DEADMAN no writer; local financial guard remains live')

    def test_deadman_ready_receipt_names_its_live_initialized_process(self):
        h=deadman_fixtures.DeadmanTests();h.setUp();self.addCleanup(h.doCleanups)
        h.arm(time.time()+3600,0)
        process=h.start()
        ready=h.ws/'batch.deadman.ready'
        self.assertTrue(custody.until(ready.exists))
        self.assertEqual(ready.read_text().split(),['4242',str(process.pid)])
        self.assertIsNone(process.poll())
        self.assertEqual(h.server.requests,[])
        process.terminate();process.communicate(timeout=5)
        self.assertFalse(ready.exists())

    @staticmethod
    def stop_guard(run):
        record=run/'controller-watchdog.pid'
        if record.exists():custody.stop_pid(int(record.read_text()))


if __name__=='__main__':unittest.main(verbosity=2)
