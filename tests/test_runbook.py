"""Public command regressions for the launch card; no live provider or GPU calls.

RUNBOOK_REPO points to an owned /tmp copy. Tiny external-screen trajectories use
the prod_ready fake evaluator's schema, solely to exercise complete/incomplete pulls.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('RUNBOOK_REPO', SOURCE))
CARD = REPO / 'data/brainstorm7/TONIGHT.md'
ENV = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', **{name:'1' for name in (
    'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}}


def put(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)


def block(section):
    text = CARD.read_text().split(section,1)[1].split('\n## ',1)[0]
    return re.search(r'```bash\n(.*?)```',text,re.S)[1]


class LaunchGates(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='card-',dir=os.environ.get('TMPDIR','/tmp'));self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.events=self.root/'events'
        wrapper=self.root/'.venv/bin/python'
        put(wrapper, '#!'+sys.executable+'\nimport os,sys\nfrom pathlib import Path\n'
            'if any(a.endswith("sentinel_gate.py") for a in sys.argv):sys.exit(int(os.environ["SENTINEL_STATUS"]))\n'
            'if "-" in sys.argv and any(a.endswith("decision.json") for a in sys.argv):os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n'
            'with Path("events").open("a") as f:f.write("python:"+" ".join(sys.argv[1:])+"\\n")\n'
            'if "jobs" in sys.argv:\n'
            ' for i in range(30):print("job%d|arms|arm|input|seq%d|49999872"%(i,i))\n')
        wrapper.chmod(0o755)
        (self.root/'data/brainstorm6/PROD-READY/generated/tier3').mkdir(parents=True)

    def run_block(self, section, **env):
        script='P=data/brainstorm6/PROD-READY\nbash() { echo "launch:$*" >>events; }\n'
        script+=re.sub(r'<ID_[^>]+>','123',block(section))
        result=subprocess.run(['bash','-c',script],cwd=self.root,env={**ENV,**env},capture_output=True,text=True)
        events=self.events.read_text().splitlines() if self.events.exists() else []
        return result,events

    def test_sentinel_hold_never_stages_or_launches_bulk(self):
        result,events=self.run_block('## ~T0+6',SENTINEL_STATUS='1')
        self.assertEqual(events,[],result.stdout+result.stderr)

    def test_sentinel_pass_reaches_all_seven_hosts(self):
        result,events=self.run_block('## ~T0+6',SENTINEL_STATUS='0')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len([e for e in events if e.startswith('launch:')]),7)
        self.assertEqual(len([e for e in events if 'make_run.py' in e]),7)

    def test_coverage_without_asset_decision_never_stages_or_launches(self):
        result,events=self.run_block('## ~T0+12-16',SENTINEL_STATUS='0')
        self.assertEqual(events,[],result.stdout+result.stderr)

    def test_coverage_without_pilot_verdict_never_stages_or_launches(self):
        result,events=self.run_block('## ~T0+12-16',SENTINEL_STATUS='0',T3_ASSET_DECISION='use')
        self.assertEqual(events,[],result.stdout+result.stderr)

    def pilot(self, bound):
        decision=self.root/'data/brainstorm6/PROD-READY/runs/ep18-pilot-d1/pulled/ep18-pilot-d1/decision.json'
        qualification=decision.parent/'qualification/qualification.json';put(qualification,'{"candidates":[]}')
        put(decision,json.dumps({'decision':'retain 50M','qualification_sha256':hashlib.sha256(qualification.read_bytes()).hexdigest() if bound else '0'*64}))

    def test_changed_pilot_qualification_never_stages_or_launches(self):
        self.pilot(False)
        result,events=self.run_block('## ~T0+12-16',SENTINEL_STATUS='0',T3_ASSET_DECISION='use')
        self.assertEqual(events,[],result.stdout+result.stderr)

    def test_asset_use_and_bound_pilot_reach_five_hosts(self):
        self.pilot(True)
        result,events=self.run_block('## ~T0+12-16',SENTINEL_STATUS='0',T3_ASSET_DECISION='use')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len([e for e in events if e.startswith('launch:')]),5)


class I1Decisions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='i1-',dir=os.environ.get('TMPDIR','/tmp'));self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ('i1_decide.py',):
            put(self.root/'data/brainstorm7/T2-STRATEGY'/name,(REPO/'data/brainstorm7/T2-STRATEGY'/name).read_text())
        put(self.root/'scripts/score_rollout.py',(REPO/'scripts/score_rollout.py').read_text())
        shutil.copytree(SOURCE/'external/kit/v2d_submission_kit/metric_code',self.root/'external/kit/v2d_submission_kit/metric_code')
        shutil.copytree(SOURCE/'external/kit/v2d_submission_kit/v2dlb',self.root/'external/kit/v2d_submission_kit/v2dlb')
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        sys.path.insert(0,str(SOURCE/'external/kit/v2d_submission_kit'))
        from v2dlb.submission_parquet import write_example
        self.sequences=['snack','trash','corn'];self.templates={}
        manifest={'sequences':[]}
        for seq in self.sequences:
            ref=self.root/f'fixtures/{seq}/robot_name=g1/data.parquet';ref.parent.mkdir(parents=True)
            mesh=ref.parent.parent/'object/textured_mesh.obj'
            put(mesh,'v 0 0 0\nv .1 0 0\nv 0 .1 0\nf 1 2 3\n')
            pq.write_table(pa.Table.from_pylist([dict(schema_version='motion_v1',object_body_names=['object'],object_mesh_paths=[str(mesh)],
                object_body_position=np.zeros((10,1,3)).tolist(),object_body_wxyz=np.tile([1.,0,0,0],(10,1,1)).tolist())]),ref)
            identity=dict(path=str(ref),sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
            manifest['sequences'].append({'sequence':seq,'arms':{'raw_repaired':{'training_reference':identity,'clean_scoring_reference':identity}}})
            self.templates[seq]={}
            for kind,offset in [('T2C-E4',0.),('T2C-contact',.05)]:
                path=self.root/f'templates/{seq}/{kind}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
                columns={k:np.zeros(10,dtype=np.int32) for k in ('environment_id','object_id','object_body_id')}
                columns.update(step=np.arange(10,dtype=np.int32),reference_frame=np.arange(10,dtype=np.int32),time_s=np.arange(10)/50,
                               environment_valid=np.ones(10,dtype=bool),object_body_name=np.array(['object']*10))
                suffix=['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']
                for prefix in ('reference','achieved'):
                    poses=np.tile([offset if prefix=='achieved' else 0.,0.,0.,0.,0.,0.,1.],(10,1))
                    for i,name in enumerate(suffix):columns[prefix+'_'+name]=poses[:,i]
                write_example(path,columns,episode_index=0,eval_hash='a'*64,code_hash='b'*64,checkpoint_hash='c'*64)
                self.templates[seq][kind]=path
        put(self.root/'data/screen/tier2/manifest.json',json.dumps(manifest))
        self.pulled=self.root/'data/brainstorm7/runs/i1screen/pulled/i1screen'

    def fixture(self, seed, repeats, *, gains=None, initial_bad=False):
        for seq in self.sequences:
            sources=dict(self.templates[seq])
            if gains is not None and seq not in gains:sources['T2C-E4']=sources['T2C-contact']
            for kind,source in sources.items():
                arm=f'{kind}-seed{seed}'
                for repeat in range(1,repeats+1):
                    target=self.pulled/f'repeats/i1/I1-{kind}-s{seed}-{seq}-r{repeat:02d}/{arm}/raw_repaired/evaluation/{seq}/49999872/trajectory.parquet'
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                if initial_bad and kind=='T2C-E4':
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    target=self.pulled/f'I1-{kind}-s{seed}-{seq}/{arm}/raw_repaired/evaluation/{seq}/49999872/trajectory.parquet'
                    target.parent.mkdir(parents=True,exist_ok=True)
                    table=pq.ParquetFile(source).read();column='achieved_position_x'
                    table=table.set_column(table.schema.get_field_index(column),column,pa.array([x+2 for x in table[column].to_pylist()],type=table[column].type))
                    pq.write_table(table,target)

    def decide(self, *pairs):
        args=[sys.executable,'-I','-B',str(self.root/'data/brainstorm7/T2-STRATEGY/i1_decide.py'),'--root',str(self.pulled),'--out',str(self.root/'decision.json')]
        for seed in pairs:args+=['--pair',f'T2C-E4-seed{seed}=T2C-contact-seed{seed}']
        return subprocess.run(args,cwd=self.root,env=ENV,capture_output=True,text=True)

    def test_missing_two_fresh_members_is_no_go(self):
        self.fixture(44,1)
        result=self.decide(44)
        self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertFalse((self.root/'decision.json').exists())

    def test_extra_fresh_member_is_no_go(self):
        self.fixture(44,4)
        result=self.decide(44)
        self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertFalse((self.root/'decision.json').exists())

    def test_three_fresh_members_pass_without_initial_evaluation(self):
        self.fixture(44,3)
        result=self.decide(44)
        self.assertEqual(result.returncode,0,result.stderr)
        output=json.loads((self.root/'decision.json').read_text())
        self.assertEqual(output['verdict'],'GO')
        self.assertEqual([(r['R']['k'],r['C']['k']) for r in output['rows']],[(3,3)]*3)

    def test_initial_evaluation_does_not_join_the_three_fresh_members(self):
        self.fixture(44,3,initial_bad=True)
        result=self.decide(44)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads((self.root/'decision.json').read_text())['verdict'],'GO',result.stdout)

    def test_one_clean_catastrophe_cannot_be_hidden_by_the_member_mean(self):
        self.fixture(44,3)
        import pyarrow as pa
        import pyarrow.parquet as pq
        for seq in self.sequences:
            for kind,offset in [('T2C-E4',.2),('T2C-contact',2.)]:
                path=next(self.pulled.glob(f'repeats/i1/*-r03/{kind}-seed44/raw_repaired/evaluation/{seq}/49999872/trajectory.parquet'))
                table=pq.ParquetFile(path).read();column='achieved_position_x'
                table=table.set_column(table.schema.get_field_index(column),column,pa.array([x+offset for x in table[column].to_pylist()],type=table[column].type))
                pq.write_table(table,path)
        result=self.decide(44)
        self.assertEqual(result.returncode,0,result.stderr)
        decision=json.loads((self.root/'decision.json').read_text())
        self.assertEqual(decision['verdict'],'NO-GO',result.stdout)
        self.assertFalse(decision['checks']['no clean catastrophe under R'])

    def test_card_uses_joint_plus_two_of_six_rule(self):
        self.fixture(44,3,gains={s for s in self.sequences if 'corn' not in s})
        self.fixture(45,3,gains=set())
        direct=self.decide(44,45)
        self.assertEqual(json.loads((self.root/'decision.json').read_text())['verdict'],'GO',direct.stdout+direct.stderr)
        command=block('## I1 decision')
        result=subprocess.run(['bash','-c',command.replace('.venv/bin/python',shlex_quote(sys.executable))],cwd=self.root,env=ENV,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        output=self.root/'data/brainstorm7/T2-STRATEGY/i1_decision.json'
        self.assertTrue(output.is_file(),result.stdout)
        decision=json.loads(output.read_text())
        self.assertEqual((decision['verdict'],len(decision['rows'])),('GO',6))
        self.assertEqual(result.stdout.count('-> GO'),1)


def shlex_quote(value):
    import shlex
    return shlex.quote(value)


if __name__=='__main__':unittest.main(verbosity=2)
