"""Public-CLI regressions for merged Tier 3 input custody. Fixtures are synthetic."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import numpy as np

from tests import test_pipeline_b8 as fixtures
SHORT,put,sha=fixtures.SHORT,fixtures.put,fixtures.sha


class MergeInputs(unittest.TestCase):
    setUp=fixtures.Pipeline.setUp
    command=fixtures.Pipeline.command
    roster=fixtures.Pipeline.roster
    manifest=fixtures.Pipeline.manifest
    admitted_manifest=fixtures.Pipeline.admitted_manifest
    cohort=fixtures.Pipeline.cohort
    qualified=fixtures.Pipeline.qualified
    merge=fixtures.Pipeline.merge
    frozen_work=fixtures.Pipeline.frozen_work
    def selected_rescue(self):
        self.roster(1)
        primary_manifest=self.admitted_manifest([0],'primary')
        primary=self.qualified([0],tier=3,manifest=primary_manifest,shifts={SHORT:[.2,.2,.2]})
        rescue_manifest=self.admitted_manifest([0],'rescue')
        rescue=self.qualified([0],tier=3,manifest=rescue_manifest)
        result=self.merge([primary],'release',rescues=[rescue])
        self.assertEqual(result.returncode,0,result.stderr)
        release=self.work/'release'
        self.assertEqual(json.loads((release/'selected.json').read_text()),json.loads((rescue/'selected.json').read_text()))
        return release,rescue_manifest

    def export_boundary(self,release):
        work=self.work/'frozen'
        if not work.exists():self.frozen_work(release)
        host=self.work/'export-host';(host/'scripts').mkdir(parents=True,exist_ok=True)
        for name in ('production_repeats.py','eval_and_pack.sh'):
            shutil.copyfile(fixtures.ROOT/'scripts'/name,host/'scripts'/name)
        if not (host/'external').exists():(host/'external').symlink_to(fixtures.ROOT/'external',target_is_directory=True)
        roster=host/'data/challenge/track_2/tier_2_synthetic_noise/meta';roster.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(self.work/'roster/meta/info.json',roster/'info.json')
        (roster/'episodes_metadata.jsonl').write_text(''.join(json.dumps({**json.loads(line),'frames':10})+'\n'
            for line in (self.work/'roster/meta/episodes_metadata.jsonl').read_text().splitlines()))
        events=self.work/'export-events'
        boundary=self.work/'fake_python'
        boundary.write_text('#!'+sys.executable+'\nimport os,sys\nfrom pathlib import Path\n'
            'with Path(os.environ["EXPORT_EVENTS"]).open("a") as f:f.write(sys.argv[1]+"\\n")\n')
        boundary.chmod(0o755)
        result=subprocess.run([sys.executable,'-B',str(host/'scripts/production_repeats.py'),'export','--out',str(release),
            '--work',str(work),'--commit',fixtures.EXPORT_URL],capture_output=True,text=True,timeout=90,
            env={**os.environ,'PYTHONPATH':str(fixtures.ROOT/'scripts'),'LAUNCH_COMMIT':str(self.commit_record),
                'PYTHON':str(boundary),'PACK_PYTHON':str(boundary),'EXPORT_EVENTS':str(events)})
        return result,events

    def test_unchanged_selected_rescue_reaches_the_export_boundary(self):
        release,_=self.selected_rescue()
        result,events=self.export_boundary(release)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('run_policy_evaluation.py',events.read_text())
        self.assertTrue((release/'EXPORT_STARTED').exists())

    def test_translated_tier3_work_is_refused_before_claim_then_sealed_work_exports(self):
        release,_=self.selected_rescue()
        work=self.frozen_work(release);path=work/'sequences/seq0/poses.npy';original=path.read_bytes()
        shifted=np.load(path);shifted[:,0,3]+=1.;np.save(path,shifted)
        result,events=self.export_boundary(release)
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('sealed converter poses/reference changed: seq0',result.stderr)
        self.assertFalse((release/'EXPORT_STARTED').exists())
        self.assertFalse(events.exists())
        path.write_bytes(original)
        result,events=self.export_boundary(release)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('run_policy_evaluation.py',events.read_text())
        self.assertTrue((release/'EXPORT_COMPLETE').exists())

    def test_selected_rescue_support_is_rehashed_before_export(self):
        release,rescue_manifest=self.selected_rescue()
        entry=json.loads(rescue_manifest.read_text())['sequences'][0]
        support=Path(next(iter(entry['arms'].values()))['support']['path'])
        support.write_text('#usda 1.0\ndef Cube "changed_rescue_obstacle" {}\n')
        result,events=self.export_boundary(release)
        print('RESCUE_SUPPORT',result.returncode,'export_calls',len(events.read_text().splitlines()) if events.exists() else 0)
        self.assertNotEqual(result.returncode,0,'selected rescue scene changed after merge but export was admitted')
        self.assertIn('manifest SHA-256 mismatch',result.stderr)
        self.assertFalse(events.exists())
        self.assertFalse((release/'EXPORT_STARTED').exists())

    def test_rescue_manifest_itself_remains_frozen_after_merge(self):
        release,rescue_manifest=self.selected_rescue()
        raw=json.loads(rescue_manifest.read_text());raw['unexpected_revision']=1;put(rescue_manifest,raw)
        result,events=self.export_boundary(release)
        print('RESCUE_MANIFEST',result.returncode,'export_calls',len(events.read_text().splitlines()) if events.exists() else 0)
        self.assertNotEqual(result.returncode,0,'merged release lost custody of its rescue manifest')
        self.assertIn('frozen parent manifest changed',result.stderr)
        self.assertFalse(events.exists())
        self.assertFalse((release/'EXPORT_STARTED').exists())

    def test_tmp_root_alias_and_private_json_permissions_allow_qualification(self):
        self.roster(1)
        cohort=self.cohort([0])
        # A symlink of the cohort ROOT is legitimate. Aliases of members inside it are not.
        alias=self.work/'root-alias';alias.symlink_to(cohort,target_is_directory=True)
        result=self.command('qualify','--out',alias)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(set(json.loads((cohort/'selected.json').read_text())),{'0'})
        self.assertEqual((cohort/'selected.json').stat().st_mode & 0o777,0o600)
        self.assertEqual((cohort/'export-binding.json').stat().st_mode & 0o777,0o600)


if __name__=='__main__':unittest.main(verbosity=2)
