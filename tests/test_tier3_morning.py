"""Morning command's controller seam; fake provider/transport/box-GPU boundaries only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(os.environ.get('T3_BATCH_REPO',Path(__file__).resolve().parents[1])).resolve()


class MorningTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ.get('T3_BATCH_TMP','/tmp'));self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.project=self.base/'project';shutil.copytree(ROOT/'scripts',self.project/'scripts')
        self.run=self.base/'batch';(self.run/'payload').mkdir(parents=True)
        self.plan=self.run/'payload/plan.json';self.plan.write_text(json.dumps({'episodes':[{'episode':0,'sequence':'ep00','frames':15}]}))
        (self.run/'upload.list').write_text('')
        (self.run/'payload/upload.json').write_text(json.dumps({'local_project':str(self.project),'files':{}}))
        self.provider=self.base/'fake_provider'
        self.provider.write_text('#!'+sys.executable+'\nimport json\nprint(json.dumps({"dph_total":.79,"public_ipaddr":"127.0.0.1","ports":{"22/tcp":[{"HostPort":"9"}]}}))\n');self.provider.chmod(0o755)
        sync=self.base/'fake_sync';sync.write_text('#!'+sys.executable+'\nimport sys\nfrom pathlib import Path\nPath('+repr(str(self.base/'sync.txt'))+').write_text(" ".join(sys.argv[1:]))\n');sync.chmod(0o755)
        morning=self.project/'scripts/tier3_native/morning.py'
        morning.write_text(morning.read_text().replace("['vastai',",'['+repr(str(self.provider))+',').replace("['rsync',",'['+repr(str(sync))+',').replace('time.sleep(10)','time.sleep(.01)').replace('shutil.disk_usage(a.run).free','(200*1024**3)'))
        (self.project/'scripts/vast/launch.sh').write_text('echo "123 offline.invalid 9" > "$1/host"\n')
        (self.project/'scripts/tier3_initializers.py').write_text('import sys\nfrom pathlib import Path\nargs=sys.argv[1:];Path(args[args.index("--out")+1]).write_text("validated fixture")\n')
        # The fake box has one native result, still paused at its first-episode barrier.
        episode=self.run/'pulled/batch/episodes/ep00';episode.mkdir(parents=True)
        (episode/'initializers.pth').write_text('native fixture')
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.expected={'plan_sha256':sha(self.plan),'canary_sha256':sha(episode/'initializers.pth')}
        metrics=[{'stage':s,'exit_code':0} for s in ('video','sam2','pack_masks','01_depth','02_export','03_human','04_align','05_object','validate')]
        result={'episode':0,'status':'SUCCESS','stage_metrics':metrics,'inputs':{'plan_sha256':sha(self.plan)},'initializers_sha256':sha(episode/'initializers.pth'),
                'wall_seconds_this_attempt':1,'peak_gpu_memory_mib_sampled':20000,'gpu_memory_total_mib':48000}
        (episode/'result.json').write_text(json.dumps(result))
        watch=self.project/'scripts/vast/watchdog.sh'
        watch.write_text('''exec "'''+sys.executable+'''" - "$1" <<'PYWATCH'
import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);(root/'watchdog.log').write_text('watchdog start fake')
if os.environ.get('STOP_BEFORE_LAUNCH'):
 (root/'STOPPED').write_text('credit floor');sys.exit(0)
time.sleep(.15)
if (root/'GO.json').exists():(root/'EARLY_GO').touch()
if os.environ.get('NO_BACKUP'):
 (root/'STOPPED').write_text('budget cap');sys.exit(0)
expected={**json.loads((root/'expected.json').read_text()),**json.loads((root/'payload/session.json').read_text())}
result_path=root/'pulled/batch/episodes/ep00/result.json';result=json.loads(result_path.read_text());result['session_id']=expected['session_id'];result_path.write_text(json.dumps(result))
(root/'pull.success.json').write_text(json.dumps({'canary':expected}))
end=time.monotonic()+5
while not (root/'GO.json').exists() and not (root/'CANARY_NO_GO.json').exists() and time.monotonic()<end:time.sleep(.01)
(root/'STOPPED').write_text('finished')
PYWATCH
''')
        (self.run/'expected.json').write_text(json.dumps(self.expected))

    def invoke(self,**env):
        return subprocess.run([sys.executable,'-B',str(self.project/'scripts/tier3_native/morning.py'),str(self.run),'123','--cap','2','--credit-floor','1',
             '--hourly-rate-ceiling','.8','--gpu-memory-mib','48000','--confirm-paid-reconstruction'],env={**os.environ,**env},capture_output=True,text=True,timeout=10)

    def test_one_command_waits_for_backup_then_releases_canary_and_checks_complete_set(self):
        result=self.invoke();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        session=json.loads((self.run/'payload/session.json').read_text())
        self.assertEqual(json.loads((self.run/'GO.json').read_text()),{**self.expected,**session})
        self.assertEqual(json.loads((self.run/'native_status.json').read_text()),{'0':'SUCCESS'})
        self.assertTrue((self.run/'canary_check/initializers.pth').exists())
        self.assertIn('GO.json',(self.base/'sync.txt').read_text())

    def test_hf_credentials_do_not_reach_provider_or_transport(self):
        script=self.provider.read_text().replace('import json', 'import json,os; assert "HF_TOKEN" not in os.environ')
        self.provider.write_text(script)
        result=self.invoke(HF_TOKEN='synthetic-not-a-secret')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_partial_pull_cannot_release_canary(self):
        result=self.invoke(NO_BACKUP='1');self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.run/'GO.json').exists());self.assertFalse((self.base/'sync.txt').exists())

    def test_old_canary_receipt_cannot_release_replacement_before_fresh_backup(self):
        old={**self.expected,'session_id':'0'*32}
        (self.run/'pull.success.json').write_text(json.dumps({'canary':old}))
        path=self.run/'pulled/batch/episodes/ep00/result.json';r=json.loads(path.read_text());r['session_id']=old['session_id'];path.write_text(json.dumps(r))
        result=self.invoke();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertFalse((self.run/'EARLY_GO').exists())
        self.assertNotEqual(json.loads((self.run/'GO.json').read_text())['session_id'],old['session_id'])

    def test_watchdog_stop_before_launch_prevents_writer_start(self):
        launch=self.project/'scripts/vast/launch.sh'
        launch.write_text('touch "$1/WRITER_STARTED"\n')
        result=self.invoke(STOP_BEFORE_LAUNCH='1');self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.run/'WRITER_STARTED').exists());self.assertFalse((self.run/'GO.json').exists())

    def test_reported_device_capacity_overrides_an_overstated_operator_capacity(self):
        import time
        path=self.run/'pulled/batch/episodes/ep00/result.json';r=json.loads(path.read_text());r['gpu_memory_total_mib']=24000;r['peak_gpu_memory_mib_sampled']=23000;path.write_text(json.dumps(r))
        result=self.invoke();self.assertNotEqual(result.returncode,0)
        self.assertTrue((self.run/'CANARY_NO_GO.json').exists());self.assertFalse((self.run/'GO.json').exists())
        deadline=time.monotonic()+1
        while not (self.run/'STOPPED').exists() and time.monotonic()<deadline:time.sleep(.01)
        self.assertTrue((self.run/'STOPPED').exists())

    def test_mac_capacity_reserve_scales_with_the_cohort_frames(self):
        morning=self.project/'scripts/tier3_native/morning.py'
        morning.write_text(morning.read_text().replace('(200*1024**3)',"int(os.environ['FIXTURE_FREE_GIB'])*1024**3"))
        for frames,free,refused in ((22132,152,True),(22132,153,False),(8366,57,True),(8366,58,False)):
            with self.subTest(frames=frames,free=free):
                self.plan.write_text(json.dumps({'episodes':[{'episode':0,'sequence':'ep00','frames':frames}]}))
                result=self.invoke(FIXTURE_FREE_GIB=str(free))
                self.assertNotEqual(result.returncode,0)
                reserve=f'reserve {152.7 if frames==22132 else 57.7:.0f} GiB Mac capacity for {frames} frames'
                if refused:
                    self.assertIn(reserve,result.stderr)
                    self.assertFalse((self.run/'host').exists())
                else:
                    # Past the reserve, the next guard (cap vs projected inference) is what stops this fixture.
                    self.assertNotIn('Mac capacity',result.stderr)
                    self.assertIn('cap below 2x serial inference',result.stderr)

    def test_rate_ceiling_fails_before_launch(self):
        self.provider.write_text('#!'+sys.executable+'\nprint(\'{"dph_total":100}\')\n')
        result=self.invoke();self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.run/'host').exists());self.assertFalse((self.run/'GO.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
