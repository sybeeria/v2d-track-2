"""launch.sh on the same container after its writer died, its deadman readiness poll, and CANARY.md's controller rearm."""
import os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from .fixture import Fixture,BASE,put

def until(predicate,message='condition not reached',timeout=15):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate():return True
        time.sleep(.05)
    raise AssertionError(message)
def gone(group):
    try:os.killpg(group,0)
    except ProcessLookupError:return True
    return False

class SameContainer(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='box-control-',dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.f=Fixture(Path(self.temp.name));self.addCleanup(self.f.stop_box)
        self.run=Path(self.temp.name)/'native';put(self.run/'jobs.txt','unused\n');put(self.run/'entry.sh','sleep 30\n')
        self.log=self.f.workspace/'native.log'
    def launch(self,env=None):
        return self.f.command(['bash',str(self.f.controller/'scripts/vast/launch.sh'),str(self.run),'123'],env)
    def box_starts(self):return self.log.read_text().count('box start ') if self.log.exists() else 0

    def test_relaunch_after_the_writer_died_starts_a_writer_and_keeps_the_interrupted_log(self):
        r=self.launch();self.assertEqual(r.returncode,0,r.stdout+r.stderr);until(lambda:self.box_starts()==1)
        first=int((self.f.home/'box-process').read_text())
        # A container restart kills the whole group before any terminal marker; its pid file and log survive.
        os.killpg(first,signal.SIGKILL);until(lambda:gone(first))
        interrupted=self.log.read_text()
        r=self.launch();self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(r.stdout.splitlines()[-1],'launched')
        groups=list(map(int,(self.f.home/'box-processes').read_text().split()))
        self.assertEqual(len(groups),2);self.assertFalse(gone(groups[1]))
        until(lambda:self.box_starts()==2,'the new writer did not append to the interrupted log')
        self.assertTrue(self.log.read_text().startswith(interrupted))

    def test_guarded_launch_polls_for_the_deadman_receipt_without_a_missing_file_error(self):
        put(self.run/'deadman-arm-123','9999999999 5400\n')
        # The box deadman publishes its receipt after the first readiness poll, then exits on its own.
        put(self.f.controller/'scripts/vast/deadman.sh','sleep .5\necho "123 $$" > /workspace/$1.deadman.ready\nsleep 5\n')
        put(self.f.work/'bin/setsid','#!/bin/sh\nexec "$@"\n');(self.f.work/'bin/setsid').chmod(0o755)
        remote=self.f.work/'bin/fake_remote';code=remote.read_text()
        old="    if 'scripts/vast/deadman.sh' in cmd:\n        event('arm_deadman')\n"
        self.assertIn(old,code);remote.write_text(code.replace(old,old+"        sys.exit(subprocess.run(['bash','-c',cmd]).returncode)\n"))
        r=self.launch({'VAST_GUARDED_LAUNCH':'1'})
        ready=self.f.workspace/'native.deadman.ready'
        if ready.exists():self.addCleanup(os.kill,int(ready.read_text().split()[1]),signal.SIGKILL)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(r.stdout.splitlines()[-1],'launched')
        self.assertEqual(r.stderr,'')

class CanaryRunbook(unittest.TestCase):
    def test_rearmed_controller_keeps_the_runs_own_cap_and_floor(self):
        text=(BASE/'data/brainstorm6/PROD-READY/CANARY.md').read_text()
        helper=text[text.index('start_controller() {'):];helper=helper[:helper.index('\n}\n')+3]
        with tempfile.TemporaryDirectory(prefix='canary-rearm-',dir='/tmp') as tmp:
            root=Path(tmp);d=root/'runs/canary';bindir=root/'bin'
            put(d/'watchdog.log','2026-10-26T01:06:14Z watchdog start id=57745227 rate=1.49 cap=12 floor=1\n'
                '2026-10-26T01:24:51Z watchdog start id=57747544 rate=1.49 cap=16 floor=4\n2026-10-26T01:34:52Z state=running\n')
            put(root/'scripts/vast/watchdog.sh','echo "$@" > "$1/args.tmp" && mv "$1/args.tmp" "$1/watchdog-args"\n')
            put(bindir/'caffeinate','#!/bin/sh\nshift\nexec "$@"\n');(bindir/'caffeinate').chmod(0o755)
            (root/'.venv/bin').mkdir(parents=True);(root/'.venv/bin/python').symlink_to(sys.executable)
            r=subprocess.run(['bash','-c',helper+'start_controller failure-watchdog.pid'],cwd=root,capture_output=True,text=True,
                env={**os.environ,'D':str(d),'PATH':str(bindir)+os.pathsep+os.environ['PATH']})
            self.assertEqual(r.returncode,0,r.stderr)
            until((d/'watchdog-args').exists)
            self.assertEqual((d/'watchdog-args').read_text(),f'{d} 16 4\n')

if __name__=='__main__':unittest.main()
