"""Public-CLI regressions for merged Tier 3 input custody. Fixtures are synthetic."""
import json
import os
from pathlib import Path
import sys
import unittest

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
        events=self.work/'export-events'
        boundary=self.work/'fake_python'
        boundary.write_text('#!'+sys.executable+'\nimport os,sys\nfrom pathlib import Path\n'
            'with Path(os.environ["EXPORT_EVENTS"]).open("a") as f:f.write(sys.argv[1]+"\\n")\n')
        boundary.chmod(0o755)
        result=self.command('export','--out',release,'--work',self.work/'frozen','--commit',
            'https://github.com/fixture/fixture/commit/0123456789abcdef',
            env={'PYTHON':str(boundary),'PACK_PYTHON':str(boundary),'EXPORT_EVENTS':str(events)})
        return result,events

    def test_unchanged_selected_rescue_reaches_the_export_boundary(self):
        release,_=self.selected_rescue()
        result,events=self.export_boundary(release)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('run_policy_evaluation.py',events.read_text())
        self.assertTrue((release/'EXPORT_STARTED').exists())

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
