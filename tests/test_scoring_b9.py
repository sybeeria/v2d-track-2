"""Known-truth regressions through qualification, sentinel, export and I1 CLIs; scratch only."""
import hashlib,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'scripts'),str(ROOT/'external/kit/v2d_submission_kit')]
from test_pipeline_b8 import Pipeline,EARLY,FINAL
from test_runbook import I1Decisions
from score_rollout import score

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def replace_achieved(path,x):
    t=pq.ParquetFile(path).read();c='achieved_position_x'
    t=t.set_column(t.schema.get_field_index(c),c,pa.array(x,type=t[c].type));pq.write_table(t,path)
    done=path.with_name('evaluation.done.json')
    if done.exists():
        d=json.loads(done.read_text());d['trajectory_sha256']=sha(path);done.write_text(json.dumps(d))

class ScoreGrid(unittest.TestCase):
    def setUp(self):
        self.f=Pipeline('runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.poses=np.tile([0.,0.,0.,0.,0.,0.,1.],(901,1))
    def test_unscored_ep11_tail_cannot_reverse_the_point05_checkpoint_rule(self):
        self.f.roster(12);q=self.f.cohort([11],tier=1)
        for p in q.glob(f'cohort/*/*/*/evaluation/seq11/{FINAL}/trajectory.parquet'):
            x=np.zeros(901);x[:45]=.2;replace_achieved(p,x)
        r=self.f.command('qualify','--out',q);self.assertEqual(r.returncode,0,r.stderr)
        row=json.loads((q/'selection.json').read_text())[0]
        self.assertEqual(row['selected_step'],EARLY,row)
        self.assertAlmostEqual(row['auc_means'][str(FINAL)],.95,places=12)
    def test_unscored_ep11_tail_cannot_hold_a_passing_sentinel_cohort(self):
        self.f.roster(30);q=self.f.cohort([11,19,29],tier=1)
        for p in q.glob('cohort/*/*/*/evaluation/*/*/trajectory.parquet'):
            seq=p.parents[1].name;x=np.zeros(901)
            if seq=='seq11':x[:900]=.09999;x[900]=.2
            if seq=='seq29':x[:]=.11
            replace_achieved(p,x)
        r=self.f.command('qualify','--out',q);self.assertEqual(r.returncode,0,r.stderr)
        r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/sentinel_gate.py'),'--qualification',str(q)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(json.loads(r.stdout),{'status':'PASS','reasons':[]})

class ExportCommit(unittest.TestCase):
    def test_bad_commit_is_rejected_before_attempt_claim_or_evaluation(self):
        f=Pipeline('runTest');f.setUp();self.addCleanup(f.doCleanups)
        f.roster(1);q=f.qualified([0],tier=2)
        r=f.merge([q],'release');self.assertEqual(r.returncode,0,r.stderr)
        release=f.work/'release';boundary=f.work/'boundary.py';events=f.work/'events'
        boundary.write_text('#!'+sys.executable+'\nimport os\nfrom pathlib import Path\nPath(os.environ["B9_EVENTS"]).write_text("evaluation invoked")\n');boundary.chmod(0o755)
        r=f.command('export','--out',release,'--work',f.work/'unused','--commit','https://github.com/fixture/fixture/tree/main',env={'PYTHON':str(boundary),'PACK_PYTHON':str(boundary),'B9_EVENTS':str(events)})
        self.assertNotEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertFalse((release/'EXPORT_STARTED').exists());self.assertFalse(events.exists())
        self.assertIn('commit',r.stderr.lower())

class I1Bindings(unittest.TestCase):
    def setUp(self):
        self.f=I1Decisions('runTest');self.f.setUp();self.addCleanup(self.f.doCleanups);self.f.fixture(44,3)
        self.archive=self.f.pulled/'frozen-screen';self.archive.mkdir(parents=True,exist_ok=True)
        m=json.loads((self.f.root/'data/screen/tier2/manifest.json').read_text())
        for e in m['sequences']:
            for key,record in e['arms']['raw_repaired'].items():
                if key not in ('training_reference','clean_scoring_reference'):continue
                original=Path(record['path']);suffix=Path(e['sequence'])/'robot_name=g1/data.parquet';target=self.archive/suffix
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
                mesh=target.parent.parent/'object/textured_mesh.obj';mesh.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(original.parent.parent/'object/textured_mesh.obj',mesh)
                record['path']=str(Path('/workspace/v2d-track-2/data/screen/tier2')/suffix)
        (self.archive/'manifest.json').write_text(json.dumps(m))
    def test_decision_uses_the_frozen_host_screen_after_mac_screen_changes(self):
        m=json.loads((self.f.root/'data/screen/tier2/manifest.json').read_text())
        for e in m['sequences']:
            ref=Path(e['arms']['raw_repaired']['training_reference']['path']);rows=pq.ParquetFile(ref).read().to_pylist()
            rows[0]['object_body_position']=np.tile([.05,0.,0.],(10,1,1)).tolist();pq.write_table(pa.Table.from_pylist(rows),ref)
            for key in ('training_reference','clean_scoring_reference'):e['arms']['raw_repaired'][key]['sha256']=sha(ref)
        (self.f.root/'data/screen/tier2/manifest.json').write_text(json.dumps(m))
        r=self.f.decide(44);self.assertEqual(r.returncode,0,r.stderr)
        d=json.loads((self.f.root/'decision.json').read_text());self.assertEqual(d['verdict'],'GO',r.stdout)
        self.assertEqual([(r['R']['MP'],r['C']['MP']) for r in d['rows']],[(1,0)]*3)
    def test_mac_screen_cannot_authorize_a_false_go_against_the_frozen_truth(self):
        manifest=json.loads((self.archive/'manifest.json').read_text())
        for entry in manifest['sequences']:
            source=self.archive/entry['sequence']/'robot_name=g1/data.parquet'
            target=self.archive/entry['sequence']/'clean/robot_name=g1/data.parquet';target.parent.mkdir(parents=True)
            rows=pq.ParquetFile(source).read().to_pylist();rows[0]['object_body_position']=np.tile([.05,0.,0.],(10,1,1)).tolist()
            pq.write_table(pa.Table.from_pylist(rows),target)
            mesh=target.parent.parent/'object/textured_mesh.obj';mesh.parent.mkdir(parents=True)
            shutil.copyfile(source.parent.parent/'object/textured_mesh.obj',mesh)
            entry['arms']['raw_repaired']['clean_scoring_reference']={'path':str(Path('/workspace/v2d-track-2/data/screen/tier2')/target.relative_to(self.archive)),'sha256':sha(target)}
            # The host trained these members from its own screen, so their recipes name the host's clean truth.
            for recipe in self.f.pulled.glob(f"repeats/i1/*/*/recipe.json"):
                content=json.loads(recipe.read_text());bound=content['references'].get(f"raw_repaired/{entry['sequence']}")
                if bound:bound['scoring']=entry['arms']['raw_repaired']['clean_scoring_reference'];recipe.write_text(json.dumps(content))
        (self.archive/'manifest.json').write_text(json.dumps(manifest))
        result=self.f.decide(44);self.assertEqual(result.returncode,0,result.stderr)
        decision=json.loads((self.f.root/'decision.json').read_text())
        self.assertEqual(decision['verdict'],'NO-GO',result.stdout)
        self.assertEqual([(r['R']['MP'],r['C']['MP']) for r in decision['rows']],[(0,1)]*3)

    def test_different_repeat_checkpoints_cannot_authorize_i1(self):
        p=next(self.f.pulled.glob('repeats/i1/*-r03/T2C-E4-seed44/raw_repaired/evaluation/snack/*/trajectory.parquet'))
        t=pq.ParquetFile(p).read();pq.write_table(t.replace_schema_metadata({**t.schema.metadata,b'flash_chord.checkpoint_sha256':b'd'*64}),p)
        r=self.f.decide(44);self.assertNotEqual(r.returncode,0,r.stdout+r.stderr);self.assertFalse((self.f.root/'decision.json').exists())
    def test_corrupted_frozen_input_is_rejected(self):
        p=self.archive/'snack/robot_name=g1/data.parquet';p.write_bytes(p.read_bytes()+b'changed')
        r=self.f.decide(44);self.assertNotEqual(r.returncode,0,r.stdout+r.stderr);self.assertFalse((self.f.root/'decision.json').exists())
    def test_missing_frozen_screen_is_rejected(self):
        shutil.rmtree(self.archive);r=self.f.decide(44)
        self.assertNotEqual(r.returncode,0,r.stdout+r.stderr);self.assertFalse((self.f.root/'decision.json').exists())

class KnownTruth(unittest.TestCase):
    def test_units_thresholds_and_no_initial_alignment(self):
        ref=np.tile([0.,0.,0.,0.,0.,0.,1.],(2,1));vertices=np.r_[np.eye(3),-np.eye(3)]*.05
        for offset,expected in [(0,[1.,1.,1.,0.]),(.1,[0.,1.,0.,10.]),(.2,[0.,0.,0.,20.])]:
            pred=ref.copy();pred[:,0]=offset;actual=score(ref,pred,vertices)
            for k,v in zip(['AUC','SP-SR','MP-SR','MPPE'],expected):self.assertAlmostEqual(actual[k],v,places=12)
        pred=ref.copy();pred[:,0]=.03;self.assertEqual(score(ref,pred,vertices)['MP-SR'],0.)
        pred=ref.copy();pred[:,5:]=[1.,0.];self.assertAlmostEqual(score(ref,pred,vertices)['MPPE'],20/3,places=12)
        pred=ref.copy();pred[:,5:]=[np.sin(.51/2),np.cos(.51/2)];a=score(ref,pred,vertices)
        self.assertEqual((a['SP-SR'],a['MP-SR']),(0.,1.))

class Tier1Precision(unittest.TestCase):
    def test_real_export_shell_preserves_strict_mp_success_and_checkpoint_hash(self):
        from v2dlb.submission_parquet import write_example
        import pandas as pd
        with tempfile.TemporaryDirectory(prefix='precision-',dir=os.environ.get('TMPDIR','/tmp')) as temp:
            project=Path(temp)
            shutil.copytree(ROOT/'scripts',project/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
            kit=project/'external/kit/v2d_submission_kit';shutil.copytree(ROOT/'external/kit/v2d_submission_kit',kit)
            runtime=project/'external/video_to_data/robotic_grounding/flash_chord';runtime.mkdir(parents=True)
            tier=project/'data/challenge/track_2/tier_1_multiview_caption'
            (tier/'meta').mkdir(parents=True);(tier/'data').mkdir()
            (tier/'meta/info.json').write_text(json.dumps({'total_episodes':1,'chunks_size':1000,'data_path':'data/episode_{episode_index:06d}.parquet'}))
            (tier/'meta/episodes_metadata.jsonl').write_text('{"episode_index":0,"sequence_id":"seq0"}\n')
            (tier/'meta/episodes.jsonl').write_text('{"episode_index":0,"length":4}\n')
            ref=np.tile([0.,0.,0.,0.,0.,0.,1.],(4,1))
            pq.write_table(pa.table({'frame_index':np.arange(4),'observation.object.pose':pa.array(np.c_[ref[:,:3],ref[:,6],ref[:,3:6]].tolist()),'observation.object.visible':[True]*4}),tier/'data/episode_000000.parquet')
            work=project/'work';(work/'sequences/seq0').mkdir(parents=True);np.save(work/'sequences/seq0/poses.npy',np.tile(np.eye(4),(4,1,1)))
            checkpoint=project/f'runs/seq0/policy_{FINAL}.safetensors';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'fixture-policy')
            sample=pd.read_csv(kit/'data/track_2_sample_submission.csv',dtype=str).iloc[:4]
            sample.to_csv(kit/'data/track_2_sample_submission.csv',index=False)
            manifest=json.loads((kit/'KIT_MANIFEST.json').read_text());manifest['files']['data/track_2_sample_submission.csv']=sha(kit/'data/track_2_sample_submission.csv')
            (kit/'KIT_MANIFEST.json').write_text(json.dumps(manifest))
            fake=project/'fake_evaluator.py'
            fake.write_text('''import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,''' + repr(str(kit)) + ''')
from v2dlb.submission_parquet import write_example
args=dict(v.split('=',1) for v in sys.argv[1:]);n=4
columns={k:np.zeros(n,dtype=np.int32) for k in ('environment_id','object_id','object_body_id')}
columns.update(step=np.arange(n,dtype=np.int32),reference_frame=np.arange(n,dtype=np.int32),time_s=np.arange(n)/50,environment_valid=np.ones(n,dtype=bool),object_body_name=['object']*n)
suffix=('position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w')
for prefix in ('reference','achieved'):
 p=np.tile([.0299996 if prefix=='achieved' else 0.,0.,0.,0.,0.,0.,1.],(n,1))
 for i,s in enumerate(suffix):columns[prefix+'_'+s]=p[:,i]
h=hashlib.sha256(Path(args['evaluation.checkpoint']).read_bytes()).hexdigest()
write_example(Path(args['evaluation.object_trajectories_output']),columns,episode_index=0,eval_hash='a'*64,code_hash='b'*64,checkpoint_hash=h)
Path(args['evaluation.metrics_output']).write_text('{}')
''')
            wrapper=project/'python_boundary'
            wrapper.write_text('#!'+sys.executable+'\nimport os,sys\na=sys.argv[1:]\nif a[0].endswith("run_policy_evaluation.py"):a += ["--eval-script",os.environ["B9_EVALUATOR"]]\nos.execv(sys.executable,[sys.executable,*a])\n');wrapper.chmod(0o755)
            out=project/'export'
            r=subprocess.run(['bash',str(project/'scripts/eval_and_pack.sh'),'tier1',str(project/'runs'),str(out),'https://github.com/fixture/fixture/commit/0123456',str(work)],capture_output=True,text=True,
                env={**os.environ,'PYTHON':str(wrapper),'PACK_PYTHON':sys.executable,'B9_EVALUATOR':str(fake)})
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            packed=pd.read_csv(out/'track2_tier1.csv',float_precision='round_trip')
            self.assertEqual(set(packed.checkpoint_sha256),{sha(checkpoint)})
            got=score(ref,packed[['pos_x','pos_y','pos_z','quat_x','quat_y','quat_z','quat_w']].to_numpy(),np.r_[np.eye(3),-np.eye(3)]*.05)
            self.assertEqual(got['MP-SR'],1.,r.stdout)
            self.assertTrue((packed.pos_x==.0299996).all())


class CanonicalMesh(unittest.TestCase):
    def test_tier1_choice_uses_the_released_mesh_instead_of_the_retargeted_mesh(self):
        import trimesh
        f=Pipeline('runTest');f.setUp();self.addCleanup(f.doCleanups)
        trimesh.creation.box(extents=[.100001]*3).export(f.mesh)
        canonical=f.work/'released.glb';trimesh.creation.box(extents=[.1]*3).export(canonical)
        roster=f.roster(1);metadata=roster/'meta/episodes_metadata.jsonl'
        record=json.loads(metadata.read_text());record['mesh']=str(canonical);metadata.write_text(json.dumps(record)+'\n')
        q=f.cohort([0],tier=1)
        theta=2*np.arcsin(.01/(np.sqrt(2)*(.0500005+.05)))
        for p in q.glob('cohort/*/*/*/evaluation/seq0/*/trajectory.parquet'):
            if p.parent.name==str(FINAL):replace_achieved(p,np.full(10,.015));continue
            t=pq.ParquetFile(p).read()
            for c,value in [('achieved_quaternion_x',np.sin(theta/2)),('achieved_quaternion_w',np.cos(theta/2))]:
                t=t.set_column(t.schema.get_field_index(c),c,pa.array([value]*10,type=t[c].type))
            pq.write_table(t,p);done=p.with_name('evaluation.done.json');d=json.loads(done.read_text());d['trajectory_sha256']=sha(p);done.write_text(json.dumps(d))
        r=f.command('qualify','--out',q);self.assertEqual(r.returncode,0,r.stderr)
        row=json.loads((q/'selection.json').read_text())[0]
        self.assertEqual(row['selected_step'],EARLY,row)
        self.assertEqual(row['auc_means'][str(EARLY)],1.)
        self.assertEqual(row['auc_means'][str(FINAL)],.9375)

if __name__=='__main__':unittest.main(verbosity=2)
