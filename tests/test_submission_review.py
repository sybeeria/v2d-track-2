"""CLI-level regressions and real-screen export controls, run only in owned /tmp copies."""
import ast
import base64
import importlib.util
import hashlib
import io
import json
import os
import subprocess
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
SANDBOX = tempfile.TemporaryDirectory(prefix='INTEGRATE-kit-', dir='/tmp')
def setUpModule():
    unittest.addModuleCleanup(SANDBOX.cleanup)
ROOT = Path(SANDBOX.name)
for name in ('scripts', 'patches'):
    shutil.copytree(PROJECT/name, ROOT/name, ignore=shutil.ignore_patterns('__pycache__'))
KIT = ROOT/'external/kit/v2d_submission_kit'
shutil.copytree(PROJECT/'external/kit/v2d_submission_kit', KIT,
                ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile(PROJECT/'tests/guarded_builder.py', ROOT/'guarded_builder.py')

sys.path.insert(0,str(KIT))
from v2dlb import object_pose as op, report_schema as rs
from v2dlb.submission_parquet import write_example
from submit.v2d_submit import validate_submission
CONFIG = json.loads((KIT/'config/tracks.json').read_text())
SAMPLE = KIT/'data/track_2_sample_submission.csv'
COMMIT = 'https://github.com/fixture/fixture/commit/0123456789abcdef'


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def sample_frame(ids):
    frame=pd.DataFrame({'row_id':ids})
    for c in op._POSE_COLUMNS: frame[c] = 1. if c=='quat_w' else 0.
    for c in rs.PROVENANCE_COLUMNS: frame[c]='a'*64
    frame['code_commit_url']=COMMIT
    return frame


def solution_for(submission):
    frame=submission[['row_id']].copy()
    keys=frame.row_id.str.extract(r'e(\d+)/w(\d+)/f(\d+)/o(\d+)').astype(int)
    keys.columns=['episode_index','world_index','frame_index','object_slot']
    frame=pd.concat([frame,keys],axis=1)
    frame['object_name']='fixture'
    for c in op._POSE_COLUMNS: frame['ref_'+c]=1. if c=='quat_w' else 0.
    return frame


def write_rollout(path, episode, reference, achieved=None):
    n=len(reference); achieved=reference if achieved is None else achieved
    columns={k:np.zeros(n,dtype=np.int32) for k in ['environment_id','object_body_id','object_id']}
    columns.update(step=np.arange(n,dtype=np.int32),reference_frame=np.arange(n,dtype=np.int32),
                   time_s=np.arange(n)/50,environment_valid=np.ones(n,dtype=bool),
                   object_body_name=np.array(['object']*n))
    suffixes=['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']
    for prefix,poses in [('achieved',achieved),('reference',reference)]:
        for i,suffix in enumerate(suffixes): columns[prefix+'_'+suffix]=poses[:,i]
    write_example(path,columns,episode_index=episode,eval_hash='a'*64,code_hash='b'*64,
                  checkpoint_hash=f'{episode+1:064x}')


def write_npz(root,episode,poses):
    root.mkdir(parents=True,exist_ok=True)
    np.savez(root/f'episode_{episode:06d}.npz',episode_index=episode,
             coordinate_frame='released_track2_world', frame_index=np.arange(len(poses)),
             pose_xyzw=poses[:,None])


class SubmissionReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.work=Path(self.temp.name)
        self.original_sample=SAMPLE.read_bytes()
        self.addCleanup(SAMPLE.write_bytes,self.original_sample)

    def builder(self,results,refs,track='track_2_tier2',ok=True):
        out=self.work/(track+'.csv')
        run=subprocess.run([sys.executable,'-B',str(ROOT/'guarded_builder.py'),str(ROOT/'scripts/pack_no_clean.py'),
            '--track',track,'--results',str(results),'--frame-references',str(refs),
            '--code-commit',COMMIT,'-o',str(out)],capture_output=True,text=True,cwd=self.work)
        if ok:
            self.assertEqual(run.returncode,0,run.stderr)
            return validate_submission(out,track,CONFIG)
        self.assertNotEqual(run.returncode,0,run.stdout)
        return run

    def test_no_clean_builder_preserves_real_screen_achieved_poses(self):
        manifest=json.loads((PROJECT/'data/screen/tier2/manifest.json').read_text())
        ids=[];expected=[];results=self.work/'rollouts';results.mkdir();refs=self.work/'refs'
        for e,s in enumerate(manifest['sequences']):
            seq=s['sequence']
            paths=list((PROJECT/'data/r5/pulled/r5').glob(f'T2C-E4-*/T2C-E4/raw_repaired/evaluation/{seq}/49999872/trajectory.parquet'))
            self.assertEqual(len(paths),1)
            table=pq.ParquetFile(paths[0]).read();md=dict(table.schema.metadata)
            md[b'flash_chord.episode_index']=str(e).encode()
            pq.write_table(table.replace_schema_metadata(md),results/f'episode_{e:06d}.parquet')
            # Policy input only; the clean scoring reference is never opened here.
            poses=np.load(Path(s['arms']['raw_repaired']['work_dir'])/'sequences'/seq/'poses.npy')
            target=np.c_[poses[:,:3,3],Rotation.from_matrix(poses[:,:3,:3]).as_quat()]
            write_npz(refs,e,target)
            raw=np.stack([table['achieved_'+c].to_numpy() for c in
                ['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']],axis=1)
            recorded=np.stack([table['reference_'+c].to_numpy() for c in
                ['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']],axis=1)
            # Independent scipy frame transform from the kit contract.
            rot=Rotation.from_quat(target[0,3:])*Rotation.from_quat(recorded[0,3:]).inv()
            offset=target[0,:3]-rot.apply(recorded[0,:3])
            expected.append(np.c_[rot.apply(raw[:,:3])+offset,(rot*Rotation.from_quat(raw[:,3:])).as_quat()])
            ids.extend(f'track_2/e{e:06d}/w000/f{f:06d}/o0' for f in range(len(target)))
        sample_frame(ids).to_csv(SAMPLE,index=False)
        for track in ['track_2_tier2','track_2_tier3']:
            got=self.builder(results,refs,track)
            expect=np.concatenate(expected); actual=got[op._POSE_COLUMNS].to_numpy()
            np.testing.assert_allclose(actual[:,:3],expect[:,:3],rtol=0,atol=1e-12)
            dots=np.abs((actual[:,3:]*expect[:,3:]).sum(axis=1))
            np.testing.assert_allclose(dots,1,rtol=0,atol=1e-12)
            self.assertEqual(len(got),1907)
            print(f'REAL_SCREEN_FIXTURE {track}: 1907 unchanged achieved samples; episode ids relabelled')
            self.assertGreater(got.checkpoint_sha256.nunique(),1)

    def full_fixture(self):
        sample=pd.read_csv(SAMPLE,dtype=str)
        groups=sample.row_id.str.extract(r'e(\d+)/w(\d+)/f(\d+)/o(\d+)').astype(int)
        results=self.work/'full-results';results.mkdir();refs=self.work/'refs'
        lengths={r['episode_index']:r['length'] for r in map(json.loads,
            (PROJECT/'data/challenge/track_2/tier_2_synthetic_noise/meta/episodes.jsonl').read_text().splitlines())}
        for e,g in groups.groupby(0):
            n=lengths[e]
            ref=np.tile([0.,0.,0.,0.,0.,0.,1.],(n,1));ref[:,0]=np.arange(n)*.001
            achieved=ref.copy();achieved[:,1]=.02345678912345
            write_rollout(results/f'episode_{e:06d}.parquet',e,ref,achieved)
            write_npz(refs,e,ref)
        return results,refs,sample

    def test_builder_matches_all_public_sample_rows_for_both_hard_tiers(self):
        results,refs,sample=self.full_fixture()
        for track in ['track_2_tier2','track_2_tier3']:
            got=self.builder(results,refs,track)
            self.assertEqual(got.row_id.tolist(),sample.row_id.tolist())
            self.assertEqual(list(got.columns),list(sample.columns))
            self.assertEqual(len(got),22989)
            self.assertEqual(got.checkpoint_sha256.nunique(),30)
            for episode,group in got.groupby(got.row_id.str.extract(r'e(\d{6})/',expand=False)):
                self.assertEqual(set(group.checkpoint_sha256),{f'{int(episode)+1:064x}'})
            self.assertEqual(set(got.packer_script_sha256),{hashlib.sha256((ROOT/'scripts/pack_no_clean.py').read_bytes()).hexdigest()})
            self.assertEqual(set(got.uploader_script_sha256),{hashlib.sha256((KIT/'submit/v2d_submit.py').read_bytes()).hexdigest()})
            print(f'FULL_FIXTURE_ONLY {track}: {len(got)} rows, {(self.work/(track+".csv")).stat().st_size} bytes')
            raw=pd.read_csv(self.work/(track+'.csv'),float_precision='round_trip')
            self.assertTrue((raw.pos_y == .02345678912345).all())
        # Deliberately deform the POLICY INPUT to prove the whole-episode residual gate fails.
        p=refs/'episode_000000.npz'
        with np.load(p) as a: data={k:a[k] for k in a.files}
        data['pose_xyzw'][1,0,0]+=.002
        np.savez(p,**data)
        self.builder(results,refs,ok=False)

    def test_offline_validator_rejects_per_frame_checkpoint_mixing(self):
        sample=pd.read_csv(SAMPLE,dtype=str)
        frame=sample_frame(sample.row_id)
        frame.loc[1,'checkpoint_sha256']='b'*64
        out=self.work/'mixed.csv';frame.to_csv(out,index=False)
        metric=load_module('mp',KIT/'metric_code/track_2/MP-SR.py')
        with self.assertRaisesRegex(metric.ParticipantVisibleError,'identical within each episode'):
            metric.score(solution_for(frame),frame.copy(),'row_id')
        run=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validate_offline.py'),str(KIT),str(out),'track_2_tier1'],
                           capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0,run.stdout)
        self.assertIn("'checkpoint_sha256' must be identical within each episode; episode(s) ['000000']",run.stderr)
        frame.loc[1,'checkpoint_sha256']='a'*64; frame.to_csv(out,index=False)
        run=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validate_offline.py'),str(KIT),str(out),'track_2_tier1'],
                           capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn(f'offline uploader validation: {len(frame)} rows',run.stdout)

    def test_public_shell_routes_hard_tiers_without_kit_packer(self):
        stub=self.work/'python-stub'
        stub.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                        'with open(os.environ["CALL_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
        stub.chmod(0o755)
        for tier in ['tier1','tier2','tier3']:
            log=self.work/(tier+'.log')
            env={**os.environ,'PYTHON':str(stub),'PACK_PYTHON':str(stub),'CALL_LOG':str(log)}
            run=subprocess.run(['bash',str(ROOT/'scripts/eval_and_pack.sh'),tier,
                str(self.work/'runs'),str(self.work/tier),COMMIT,str(self.work/'frozen')],
                env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            calls=[json.loads(s) for s in log.read_text().splitlines()]
            packer=[c for c in calls if c and c[0].endswith('pack_submission.py')]
            self.assertEqual(len(packer),1 if tier=='tier1' else 0,calls)
            if tier!='tier1':
                self.assertEqual(sum(c[0]=='scripts/pack_no_clean.py' for c in calls),1,calls)
                stepcall=next(c for c in calls if c[0]=='scripts/write_checkpoints.py')
                self.assertEqual(stepcall[stepcall.index('--steps')+1],'49999872')
            self.assertEqual(calls[-1][0],'scripts/validate_offline.py',calls)
        # Stamped packer/uploader hashes must stay the official ones.
        manifest=json.loads((KIT/'KIT_MANIFEST.json').read_text())['files']
        for name in ['tools/pack_submission.py','submit/v2d_submit.py']:
            self.assertEqual(hashlib.sha256((KIT/name).read_bytes()).hexdigest(),manifest[name])

    def test_builder_rejects_invalid_grid_world_and_provenance(self):
        results,refs,sample=self.full_fixture()
        path=results/'episode_000000.parquet'
        table=pq.ParquetFile(path).read()
        mutations=[table.slice(1), pa.concat_tables([table,table.slice(0,1)]),
            table.set_column(table.schema.get_field_index('environment_valid'),'environment_valid',
                             pa.array([False]*len(table))),
            table.set_column(table.schema.get_field_index('environment_id'),'environment_id',
                             pa.array([1]*len(table),type=pa.int32())),
            table.replace_schema_metadata({k:v for k,v in table.schema.metadata.items()
                                           if k!=b'flash_chord.checkpoint_sha256'})]
        for malformed in mutations:
            pq.write_table(malformed,path)
            run=self.builder(results,refs,ok=False)
            self.assertIn('error:',run.stderr)
        pq.write_table(table,path)
        self.builder(results,refs)

if __name__ == '__main__': unittest.main(verbosity=2)
