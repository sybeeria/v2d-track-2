"""launch.sh records the uploaded snapshot on the rsync-only export host."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest

from .fixture import Fixture,BASE
sys.path[:0]=[str(BASE)]

SHA=subprocess.check_output(['git','-C',str(BASE),'rev-parse','HEAD'],text=True).strip()


class LaunchCommit(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='launch-commit-',dir='/tmp');self.addCleanup(temporary.cleanup)
        self.f=Fixture(Path(temporary.name));self.addCleanup(self.f.stop_box)
        git=self.f.work/'bin/git'
        git.write_text('#!'+sys.executable+'\nimport os,subprocess,sys\n'
            f'if sys.argv[-2:]==["rev-parse","HEAD"]:print({SHA!r})\n'
            'elif sys.argv[-2:]==["status","--porcelain"]:sys.stdout.write(" M scripts/export.py\\n"*int(os.environ.get("FIXTURE_DIRTY",0)))\n'
            f'else:sys.exit(subprocess.run([{shutil.which("git")!r},"-C",{str(BASE)!r},*sys.argv[3:]]).returncode)\n');git.chmod(0o755)

    def run_dir(self,name):
        run=self.f.work/name;run.mkdir(parents=True)
        (run/'env.sh').write_text('export BUILD_SCREEN=0\n')
        (run/'jobs.txt').write_text('')
        return run

    def assert_record(self,run,changed):
        expected=f'{SHA} changed_files={changed}\n'
        self.assertEqual((self.f.remote/'LAUNCH_COMMIT').read_text(),expected)
        self.assertEqual(' '.join((run/'launch-code.log').read_text().split()[2:])+'\n',expected)

    def test_added_payload_files_are_dirty_unless_they_are_run_data(self):
        run=self.run_dir('added-data');payload=run/'payload'
        seed='data/brainstorm6/PROD-READY/seed-inputs/added-data/'
        allowed=[seed+'manifest.json']+[seed+'files/data/work/tier1/asset'+ext
            for ext in ('.parquet','.npy','.obj','.mtl','.png','.urdf','.usda')]
        unexpected=['scripts/policy.json','scripts/extension.pth','data/unscoped.json','plan.json',
                    seed+'files/data/work/tier1/loader.py',seed+'files/data/work/tier1/loader.so']
        for rel in allowed+unexpected:
            target=payload/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('fixture data\n')
        (payload/seed/'files/data/work/tier1/link.png').symlink_to('asset.png')
        self.f.launch(run)
        self.assert_record(run,len(unexpected)+1)

    def test_added_import_changes_execution_and_refuses_export(self):
        from tests.test_export_custody import ExportCustody
        run=self.run_dir('added-import');marker=self.f.work/'import-executed'
        added=run/'payload/scripts/argparse.py';added.parent.mkdir(parents=True)
        stdlib=Path(sysconfig.get_path('stdlib'))/'argparse.py'
        added.write_text(f'from pathlib import Path\nPath({str(marker)!r}).write_text("payload executed")\n'
            f'exec(compile(Path({str(stdlib)!r}).read_bytes(),__file__,"exec"))\n')
        self.f.launch(run)
        result=self.f.command([sys.executable,str(self.f.remote/'scripts/production_repeats.py'),'--help'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(marker.read_text(),'payload executed')
        export=ExportCustody();export.setUp();self.addCleanup(export.doCleanups)
        case=export.prepare_export()
        shutil.copyfile(self.f.remote/'LAUNCH_COMMIT',export.commit_record)
        result=export.export(case,'https://github.com/fixture/fixture/commit/'+SHA)
        print(f'ADDED_IMPORT record={export.commit_record.read_text().strip()!r} export_exit={result.returncode} EXPORT_COMPLETE={(case[0]/"EXPORT_COMPLETE").exists()}',flush=True)
        self.assert_record(run,1)
        export.refused(case,result,'clean launch commit')

    def test_real_payloads_remain_clean_and_unknown_data_is_dirty(self):
        sync=self.f.work/'bin/fake_sync'
        anchor="elif tool=='fake_sync':"
        self.assertIn(anchor,sync.read_text())
        sync.write_text(sync.read_text().replace(anchor,anchor+
            "\n    if any(a.rstrip('/')==os.environ.get('AUDIT_PAYLOAD') for a in args):sys.exit(0)"))
        remote=self.f.work/'bin/fake_remote';anchor="elif 'setsid nohup' in cmd:"
        self.assertIn(anchor,remote.read_text())
        remote.write_text(remote.read_text().replace(anchor,anchor+
            "\n        if os.environ.get('AUDIT_PAYLOAD'):print('launched');sys.exit(0)"))
        verifier=Path('data/brainstorm6/PROD-READY/verify_payload.py')
        (self.f.controller/verifier).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(BASE/verifier,self.f.controller/verifier)
        payloads=sorted(set(BASE.glob('data/brainstorm6/PROD-READY/runs/*/payload'))|
                        set(BASE.glob('data/brainstorm6/T3-BATCH/**/payload')))
        self.assertTrue(payloads)
        for payload in payloads:
            with self.subTest(payload=payload):
                run=self.run_dir('audit/'+payload.parent.name)
                (run/'payload').symlink_to(payload,target_is_directory=True)
                native=(payload.parent/'entry.sh').exists()
                if native:(run/'entry.sh').write_text('true\n')
                self.f.launch(run,{'AUDIT_PAYLOAD':str(run/'payload')})
                expected=f'{SHA} changed_files=0\n'
                self.assertEqual(' '.join((run/'launch-code.log').read_text().split()[2:])+'\n',expected)
                if native:self.assertFalse((self.f.remote/'LAUNCH_COMMIT').exists())
                else:self.assertEqual((self.f.remote/'LAUNCH_COMMIT').read_text(),expected)
                print(f'PAYLOAD {payload.parent.relative_to(BASE)} changed_files=0 native={native}',flush=True)
        run=self.run_dir('unknown-data');added=run/'payload/data/new.json';added.parent.mkdir(parents=True);added.write_text('{}\n')
        self.f.launch(run)
        self.assert_record(run,1)

    def test_tracked_payload_overlays_are_counted_against_the_commit(self):
        rel='data/brainstorm6/PROD-READY/generate_jobs.py'
        committed=subprocess.run(['git','-C',str(BASE),'show','HEAD:'+rel],capture_output=True,check=True).stdout
        for changed,dirty in ((False,0),(True,0),(True,2)):
            with self.subTest(changed=changed,dirty=dirty):
                run=self.run_dir(f'payload{changed}{dirty}')
                payload=run/'payload'/rel;payload.parent.mkdir(parents=True,exist_ok=True)
                marker=self.f.work/f'executed-{dirty}'
                payload.write_bytes(f'from pathlib import Path\nPath({str(marker)!r}).write_text("payload executed")\n'.encode() if changed else committed)
                self.f.launch(run,{'FIXTURE_DIRTY':str(dirty)})
                if changed:
                    result=self.f.command([sys.executable,str(self.f.remote/rel)])
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual(marker.read_text(),'payload executed')
                expected=f'{SHA} changed_files={dirty+int(changed)}\n'
                self.assertEqual((self.f.remote/'LAUNCH_COMMIT').read_text(),expected)
                self.assertEqual(' '.join((run/'launch-code.log').read_text().split()[2:])+'\n',expected)
                self.assertEqual((self.f.remote/rel).read_bytes(),payload.read_bytes())

    def test_full_launch_records_clean_and_dirty_snapshot_on_box_and_controller(self):
        for dirty in (0,2):
            with self.subTest(dirty=dirty):
                run=self.run_dir(f'launch{dirty}')
                self.f.launch(run,{'FIXTURE_DIRTY':str(dirty)})
                expected=f'{SHA} changed_files={dirty}\n'
                self.assertEqual((self.f.remote/'LAUNCH_COMMIT').read_text(),expected)
                self.assertEqual(' '.join((run/'launch-code.log').read_text().split()[2:])+'\n',expected)
                self.assertFalse((self.f.remote/'.git').exists())

    def test_native_partial_upload_cannot_certify_a_full_project_commit(self):
        record=self.f.remote/'LAUNCH_COMMIT';record.write_text(SHA+' changed_files=0\n')
        run=self.f.work/'native';run.mkdir();(run/'entry.sh').write_text('true\n')
        self.f.launch(run)
        self.assertFalse(record.exists())

    def test_failed_project_upload_invalidates_the_previous_commit(self):
        record=self.f.remote/'LAUNCH_COMMIT';record.write_text(SHA+' changed_files=0\n')
        sync=self.f.work/'bin/fake_sync'
        sync.write_text(sync.read_text().replace("elif tool=='fake_sync':",
            "elif tool=='fake_sync':\n    if os.environ.get('FAIL_PROJECT_UPLOAD') and any(a.endswith('/controller/') for a in args):sys.exit(23)"))
        run=self.run_dir('failed')
        result=self.f.command(['bash',str(self.f.controller/'scripts/vast/launch.sh'),str(run),self.f.instance(run)],
            {'FAIL_PROJECT_UPLOAD':'1'})
        self.assertEqual(result.returncode,23,result.stdout+result.stderr)
        self.assertFalse(record.exists())
        self.assertFalse(self.f.events.exists())


    def test_unusable_endpoint_keeps_the_host_file_and_starts_nothing(self):
        for ip,port in (('',9),('  ',9),('a b',9),('offline.invalid','')):
            with self.subTest(ip=ip,port=port):
                run=self.run_dir(f'endpoint-{len(ip)}-{port}');instance=self.f.instance(run)
                (run/'host').write_text(f'{instance} known.good.invalid 8\n')
                reply=json.dumps({'start_date':1,'dph_total':1,'public_ipaddr':ip,'ports':{'22/tcp':[{'HostPort':port}]}})
                result=self.f.command(['bash',str(self.f.controller/'scripts/vast/launch.sh'),str(run),instance],
                    {'TRANSPORT_INSTANCE_JSON':reply})
                self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                self.assertIn(f'cannot read the SSH endpoint of instance {instance}',result.stderr)
                self.assertEqual((run/'host').read_text(),f'{instance} known.good.invalid 8\n')
                self.assertFalse((run/'launch-code.log').exists())
        transport=(self.f.home/'transport.jsonl').read_text()
        self.assertEqual(transport.count('"kind": "instance"'),4)
        self.assertNotIn('"kind": "remote"',transport)


if __name__=='__main__':unittest.main(verbosity=2)
