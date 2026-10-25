"""Public start_run regression for a live watchdog still transitioning through its launcher."""
import json,os,shutil,subprocess,sys,tempfile,time,unittest
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tests.prod_ready.fixture import Fixture,put

class StartupTransition(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='startup-transition-',dir=os.environ.get('TMPDIR','/tmp'))
        self.addCleanup(self.temp.cleanup);self.work=Path(self.temp.name).resolve();self.f=Fixture(self.work);self.addCleanup(self.stop)
        self.run_dir=self.work/'canary';self.run_dir.mkdir();put(self.run_dir/'entry.sh','echo fixture-entry\n')
    def stop(self):
        own={}
        for p in psutil.process_iter(['cmdline','name']):
            try:
                if p.pid!=os.getpid() and p.info['name'] not in ('zsh','rg') and str(self.work) in ' '.join(p.info['cmdline'] or []):
                    own[p.pid]=p
                    for q in p.children(recursive=True):own[q.pid]=q
            except psutil.Error:pass
        for p in own.values():
            try:p.suspend()
            except psutil.Error:pass
        for p in reversed(list(own.values())):
            try:p.kill()
            except psutil.Error:pass
        psutil.wait_procs(list(own.values()),timeout=2)
    def start(self):
        return self.f.command(['bash',self.f.controller/'scripts/vast/start_run.sh',self.run_dir,'998001','9','4'],timeout=12)
    def test_start_waits_for_live_launcher_to_become_the_armed_watchdog(self):
        # A real exec transition drops the final watchdog argv temporarily. Keep PID and inherited lock FDs.
        helper=self.work/'launcher_transition.py'
        put(helper,"import json,os,sys,time\nfrom pathlib import Path\np=Path(sys.argv[1]);cmd=json.loads(p.read_text());time.sleep(.8);os.execvp(cmd[0],cmd)\n")
        launcher=self.work/'bin/caffeinate'
        put(launcher,f'#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\np=Path({str(self.work)!r})/f"command-{{os.getpid()}}.json";p.write_text(json.dumps(sys.argv[2:]))\nos.execv(sys.executable,[sys.executable,{str(helper)!r},str(p)])\n');launcher.chmod(0o755)
        result=self.start()
        until=time.monotonic()+4
        while not (self.run_dir/'watchdog.log').exists() and time.monotonic()<until:time.sleep(.05)
        print('TRANSITION',result.returncode,result.stderr.strip(),flush=True)
        self.assertTrue((self.run_dir/'watchdog.log').exists(),'fixture never became a real watchdog')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertFalse((self.run_dir/'controller-launch.pending.json').exists())
        events=[json.loads(line) for line in (self.f.home/'transport.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['kind']=='remote' and 'setsid nohup bash' in e['command'] and 'deadman.sh' not in e['command'] for e in events),1)
        self.assertEqual((self.run_dir/'launch-code.log').read_text().count('\n'),1)
    def test_exited_launcher_refuses_the_writer(self):
        launcher=self.work/'bin/caffeinate';put(launcher,'#!/bin/sh\nexit 42\n');launcher.chmod(0o755)
        result=self.start();self.assertNotEqual(result.returncode,0);self.assertIn('watchdog failed to arm',result.stderr)
        self.assertFalse((self.run_dir/'launch-code.log').exists())
        transport=self.f.home/'transport.jsonl'
        self.assertFalse(transport.exists() and 'setsid nohup bash' in transport.read_text())

if __name__=='__main__':unittest.main()
