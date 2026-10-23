"""Native batch connects to real launch/box/watchdog with existing fake transports."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import test_runner_regressions as runner


class NativeLaunchTests(unittest.TestCase):
    def setUp(self):
        helper=runner.WatchdogTests();helper.setUp();self.addCleanup(helper.doCleanups)
        for name in ('run','work','root','execute','events','watchdog'):setattr(self,name,getattr(helper,name))

    def test_native_entry_upload_bypasses_screen_seed_configs(self):
        (self.run/'entry.sh').write_text('exit 0\n')
        payload=self.run/'payload';payload.mkdir();(payload/'plan.json').write_text('{}')
        result=self.execute(['bash','scripts/vast/launch.sh',str(self.run),'123'])
        self.assertEqual(result.returncode,0,result.stderr)
        sync=[e['args'] for e in self.events() if e['kind']=='sync']
        self.assertTrue(any(str(self.run/'entry.sh') in args for args in sync))
        self.assertFalse(any(any('*config.json' in a for a in args) for args in sync))
        self.assertTrue(any(str(payload)+'/' in args for args in sync))
        self.assertFalse(any(str(self.root.resolve())+'/' in args for args in sync))
        self.assertTrue(any(str((self.root/'scripts/tier3_initializers.py').resolve()) in args for args in sync))
        backstop=self.root/'scripts/vast/deadman.sh'
        if backstop.exists():self.assertTrue(any(str(backstop.resolve()) in args for args in sync))

    def test_direct_launch_and_watchdog_drop_hf_credentials(self):
        for name in ('fake_provider','fake_remote','fake_sync'):
            path=self.work/'bin'/name;lines=path.read_text().splitlines()
            path.write_text(lines[0]+'\nimport os; assert "HF_TOKEN" not in os.environ\n'+'\n'.join(lines[1:])+'\n')
        (self.run/'entry.sh').write_text('exit 0\n')
        result=self.execute(['bash','scripts/vast/launch.sh',str(self.run),'123'],HF_TOKEN='synthetic-not-a-secret')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        result=self.watchdog(HF_TOKEN='synthetic-not-a-secret')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_native_outputs_are_backed_up_and_uncommitted_staging_is_excluded(self):
        for name in ('initializers.pth','human.pkl','.staging'):(self.work/'remote'/name).write_text('native '+name)
        result=self.watchdog()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((self.run/'pulled/batch/initializers.pth').read_text(),'native initializers.pth')
        self.assertEqual((self.run/'pulled/batch/human.pkl').read_text(),'native human.pkl')
        self.assertFalse((self.run/'pulled/batch/.staging').exists())
        excludes=[e['args'] for e in self.events() if e['kind']=='sync' and '--exclude' in e['args']]
        self.assertTrue(any('build' in args and '.staging' in args for args in excludes))


class NativeBoxTests(unittest.TestCase):
    def setUp(self):
        helper=runner.TrainingTests();helper.setUp();self.addCleanup(helper.doCleanups)
        for name in ('run','work','root','execute','make_box'):
            if hasattr(helper,name):setattr(self,name,getattr(helper,name))

    def test_native_entry_owns_setup_and_failure_is_terminal(self):
        box=self.make_box()
        (self.root/'scripts/gpu_bootstrap.sh').write_text('exit 97\n')
        entry=self.work/'workspace/batch.entry.sh';entry.write_text('echo native-entry\nexit 0\n')
        result=self.execute(['bash',str(box)])
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('native-entry',result.stdout);self.assertIn('box done ',result.stdout)
        self.assertFalse((self.work/'post-ran').exists())
        entry.write_text('exit 7\n')
        result=self.execute(['bash',str(box)])
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('box failed ',result.stdout);self.assertNotIn('box done ',result.stdout)


class NativeTransferTests(unittest.TestCase):
    def test_deadline_retains_real_rsync_partial_and_retry_recovers_exact_bytes(self):
        import hashlib
        import subprocess
        helper=runner.WatchdogTests();helper.setUp();self.addCleanup(helper.doCleanups)
        source=helper.work/'remote'
        for p in source.iterdir():p.unlink()
        data=os.urandom(8*1024*1024);(source/'depth.npy').write_bytes(data)
        dest=helper.run/'pulled/batch';dest.mkdir(parents=True)
        transport=helper.work/'local_rsync_transport'
        transport.write_text('#!'+sys.executable+'\nimport os,sys\nassert sys.argv[1]=="offline.invalid"\nos.execv("/usr/bin/rsync",sys.argv[2:])\n');transport.chmod(0o755)
        # A local protocol pipe, never SSH or a network connection. The actual
        # watchdog deadline kills this real slow rsync, not an imitation of it.
        sync=helper.bin/'fake_sync'
        sync.write_text('#!'+sys.executable+'\nimport os\nos.execv("/usr/bin/rsync",["rsync","-a","--bwlimit=100","--partial-dir=.rsync-partial","-e",'+repr(str(transport))+',"offline.invalid:'+str(source)+'/",'+repr(str(dest)+'/')+'])\n')
        result=helper.watchdog(cap='0',STATE='running',WATCHDOG_COMMAND_TIMEOUT_SECONDS='1')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        partial=dest/'.rsync-partial/depth.npy'
        self.assertTrue(partial.exists());self.assertGreater(partial.stat().st_size,0)
        self.assertFalse((dest/'depth.npy').exists())
        self.assertTrue((helper.run/'BACKUP_FAILED').exists())
        self.assertIn('budget cap',(helper.run/'STOPPED').read_text())
        retry=subprocess.run(['/usr/bin/rsync','-a','--partial-dir=.rsync-partial','-e',str(transport),'offline.invalid:'+str(source)+'/',str(dest)+'/'],capture_output=True,text=True,timeout=10)
        self.assertEqual(retry.returncode,0,retry.stderr)
        self.assertEqual(hashlib.sha256((dest/'depth.npy').read_bytes()).digest(),hashlib.sha256(data).digest())
