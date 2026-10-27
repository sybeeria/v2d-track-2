"""Export admission through the public CLI; no GPU or network calls."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tests import test_pipeline_b8 as fixtures


class ExportCustody(unittest.TestCase):
    setUp=fixtures.Pipeline.setUp
    command=fixtures.Pipeline.command
    roster=fixtures.Pipeline.roster
    manifest=fixtures.Pipeline.manifest
    cohort=fixtures.Pipeline.cohort
    qualified=fixtures.Pipeline.qualified
    merge=fixtures.Pipeline.merge
    frozen_work=fixtures.Pipeline.frozen_work

    def prepare_export(self,tier=2):
        self.roster(1)
        cohort=self.qualified([0],tier=tier)
        work=self.frozen_work(cohort)
        events=cohort/'export-events'
        boundary=self.work/'fake-python'
        boundary.write_text('#!'+sys.executable+'\nimport os,sys\nfrom pathlib import Path\n'
            'with Path(os.environ["EXPORT_EVENTS"]).open("a") as f:f.write(Path(sys.argv[1]).name+"\\n")\n')
        boundary.chmod(0o755)
        return cohort,work,events,{'PYTHON':str(boundary),'PACK_PYTHON':str(boundary),'EXPORT_EVENTS':str(events)}

    def export(self,case,commit=fixtures.EXPORT_URL):
        cohort,work,events,env=case
        return self.command('export','--out',cohort,'--work',work,'--commit',commit,env=env)

    def refused(self,case,result,error):
        cohort,_,events,_=case
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn(error,result.stderr)
        self.assertFalse((cohort/'EXPORT_STARTED').exists())
        self.assertFalse((cohort/'export').exists())
        self.assertFalse(events.exists())

    def honest(self,case):
        result=self.export(case)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        cohort,_,events,_=case
        self.assertEqual(events.read_text().splitlines(),['write_checkpoints.py','frame_references.py',
            'run_policy_evaluation.py','pack_no_clean.py','validate_offline.py'])
        self.assertTrue((cohort/'EXPORT_STARTED').exists())
        self.assertTrue((cohort/'EXPORT_COMPLETE').exists())

    def test_translated_work_refused_before_claim_then_restored_work_exports(self):
        for tier in (1,2):
            with self.subTest(tier=tier):
                case=self.prepare_export(tier)
                poses=case[1]/'sequences/seq0/poses.npy';original=poses.read_bytes()
                shifted=np.load(poses);shifted[:,0,3]+=1.;np.save(poses,shifted)
                self.refused(case,self.export(case),'export work differs from frozen converter poses: seq0')
                poses.write_bytes(original)
                self.honest(case)

    def test_stale_abbreviated_dirty_or_absent_commit_refused_before_claim(self):
        for kind in ('stale','abbreviated','abbreviated-record','dirty','missing','malformed'):
            with self.subTest(kind=kind):
                case=self.prepare_export()
                self.commit_record.write_text(fixtures.EXPORT_SHA+' changed_files=0\n')
                commit=fixtures.EXPORT_URL
                if kind=='stale':commit=commit[:-40]+'f'*40
                if kind=='abbreviated':commit=commit[:-33]
                if kind=='abbreviated-record':
                    commit=commit[:-33];self.commit_record.write_text(fixtures.EXPORT_SHA[:7]+' changed_files=0\n')
                if kind=='dirty':self.commit_record.write_text(fixtures.EXPORT_SHA+' changed_files=1\n')
                if kind=='missing':self.commit_record.unlink()
                if kind=='malformed':self.commit_record.write_text(fixtures.EXPORT_SHA+'\n')
                self.refused(case,self.export(case,commit),'clean launch commit')
                self.commit_record.write_text(fixtures.EXPORT_SHA+' changed_files=0\n')
                self.honest(case)

    def test_missing_or_wrong_manifest_archive_refused_before_claim(self):
        for kind in ('missing','manifest'):
            with self.subTest(kind=kind):
                case=self.prepare_export()
                plan=json.loads((case[0]/'plan.json').read_text())
                archive=Path(plan['source_root'])/'frozen-inputs/manifest.json'
                original=archive.read_bytes()
                if kind=='missing':archive.unlink()
                else:
                    raw=json.loads(original);raw['input_manifest_sha256']='f'*64;fixtures.put(archive,raw)
                self.refused(case,self.export(case),'frozen-inputs' if kind=='missing' else 'different input manifest')
                archive.write_bytes(original)
                self.honest(case)


if __name__=='__main__':unittest.main(verbosity=2)
