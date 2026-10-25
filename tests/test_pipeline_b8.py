"""Public CLI regressions. Synthetic evaluator receipts test admission, not GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq
import trimesh
from tests.test_score_rollout import write_reference, write_rollout

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/production_repeats.py'
EARLY, FINAL, SHORT = 330012672, 349999104, 49999872


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='pipeline-', dir=os.environ.get('TMPDIR', '/tmp'))
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name).resolve()
        self.mesh = self.work/'object.ply'
        trimesh.creation.box(extents=[.1, .1, .1]).export(self.mesh)
        self.poses = np.tile([0., 0., 0., 0., 0., 0., 1.], (10, 1))
        self.counter = 0
        self.supports = []

    def command(self, *args, env=None):
        return subprocess.run([sys.executable, '-B', str(SCRIPT), *map(str, args)], cwd=ROOT,
            env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=90)

    def roster(self, count):
        root = self.work/'roster'
        put(root/'meta/info.json', {'total_episodes':count})
        (root/'meta/episodes_metadata.jsonl').write_text(''.join(
            json.dumps({'episode_index':i, 'sequence_id':f'seq{i}', 'mesh':str(self.mesh)})+'\n' for i in range(count)))
        return root

    def manifest(self, episodes, tier, name):
        path = self.work/name/'manifest.json'
        rows = []
        for episode in episodes:
            base = path.parent/f'ep{episode}/assets'
            reference = base/f'human_motion_data/whole_body/soma/sequence_id=seq{episode}/robot_name=g1/data.parquet'
            reference.parent.mkdir(parents=True, exist_ok=True)
            write_reference(reference, self.poses, self.mesh)
            support = base/f'human_motion_data/whole_body/reconstructed_stage/seq{episode}_support.usda'
            support.parent.mkdir(parents=True, exist_ok=True)
            support.write_text('#usda 1.0\ndef Cube "platform" {}\n')
            self.supports.append(support)
            record = {'path':str(reference), 'sha256':sha(reference)}
            label = 'clean' if tier==1 else 'raw_repaired' if tier==2 else 'tier3_synthetic'
            rows.append({'sequence':f'seq{episode}', 'frames':10, 'arms':{label:{
                'training_reference':record, 'clean_scoring_reference':record,
                'support':{'path':str(support), 'sha256':sha(support), 'enabled':True}}},
                'assets':[{'path':str(self.mesh), 'sha256':sha(self.mesh)}]})
        put(path, {'tier':tier, 'scoring_mode':'self', 'sequences':rows})
        return path

    def admitted_manifest(self, episodes, name, coverage='a'*64):
        """Use the real coverage admit CLI to emit the host-specific subset manifest."""
        manifest = self.manifest(episodes, 3, name)
        raw = json.loads(manifest.read_text())
        binding = manifest.parent/'input-assets/binding.json'
        roster = binding.parent/'roster/meta'
        source_roster=self.work/'roster/meta'
        roster.mkdir(parents=True)
        for filename in ('info.json','episodes_metadata.jsonl'):
            shutil.copyfile(source_roster/filename,roster/filename)
        entries = []
        for row in raw['sequences']:
            label, arm = next(iter(row['arms'].items()))
            reference = Path(arm['training_reference']['path'])
            base = reference.parents[5]
            # The six enclosing directories above data.parquet leave the materialized assets root.
            self.assertEqual(base.name, 'assets')
            entries.append({'episode':int(row['sequence'][3:]), 'sequence':row['sequence'], 'frames':10,
                'label':label, 'assets':str(base), 'sha256':{
                    p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file()}})
        put(binding, {'coverage_sha256':coverage, 'episodes':entries, 'arms':{}, 'ep18_sha256':{}, 'recipes_sha256':{},
            'pilot_decision':{'decision':'retain 50M'},
            'roster':{p.name:sha(p) for p in roster.iterdir()}})
        manifest.unlink()
        result = subprocess.run([sys.executable,'-B',str(ROOT/'scripts/tier3_coverage.py'), 'admit',
            '--binding',str(binding),'--output',str(manifest)], capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        return manifest

    def cohort(self, episodes, tier=2, manifest=None, shifts=None, name=None):
        self.counter += 1
        name = name or f'cohort{self.counter}'
        manifest = manifest or self.manifest(episodes,tier,name)
        raw = json.loads(manifest.read_text())
        by_sequence = {row['sequence']:row for row in raw['sequences']}
        source = self.work/name/'source'
        jobs = self.work/name/'jobs.txt'
        arm = 'F6-rebalance-tail' if tier==1 else 'T2C-contact'
        arms_file = 'pilot_arms_350m.txt' if tier==1 else 'pilot_arms.txt'
        steps = [EARLY,FINAL] if tier==1 else [SHORT]
        lines = []
        for episode in episodes:
            sequence=f'seq{episode}'; label=next(iter(by_sequence[sequence]['arms']))
            job=f't{tier}-ep{episode:02d}'
            base=source/job/arm
            put(base/'recipe.json',{'schema_version':2,'overrides':[],'references':{}})
            for step in steps:
                checkpoint=base/label/'train'/sequence/f'policy_{step}.safetensors'
                checkpoint.parent.mkdir(parents=True,exist_ok=True)
                checkpoint.write_text(f'{name}-{episode}-{step}')
                put(base/label/'evaluation'/sequence/str(step)/'evaluation.done.json', {'checkpoint_sha256':sha(checkpoint)})
            lines.append('|'.join([job,arms_file,arm,label,sequence,' '.join(map(str,steps))]))
        jobs.write_text('\n'.join(lines)+'\n')
        cohort=self.work/name/'qualification'
        roster=self.work/'roster'
        if not roster.exists(): self.roster(max(episodes)+1)
        result=self.command('prepare','--jobs',jobs,'--manifest',manifest,'--source',source,
            '--tier-root',roster,'--tier',tier,'--out',cohort)
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads((cohort/'plan.json').read_text())
        for c in plan['candidates']:
            arm_record=by_sequence[c['sequence']]['arms'][c['input']]
            for repeat in range(1,4):
                for step in steps:
                    base=cohort/'cohort'/f"{c['name']}-r{repeat:02d}"/c['arm']/c['input']
                    checkpoint=base/'train'/c['sequence']/f'policy_{step}.safetensors'
                    result_dir=base/'evaluation'/c['sequence']/str(step)
                    result_dir.mkdir(parents=True)
                    trajectory=result_dir/'trajectory.parquet'
                    achieved=self.poses.copy()
                    if shifts: achieved[:,0] += shifts[step][repeat-1]
                    write_rollout(trajectory,achieved)
                    table=pq.ParquetFile(trajectory).read()
                    pq.write_table(table.replace_schema_metadata({**table.schema.metadata,
                        b'flash_chord.checkpoint_sha256':sha(checkpoint).encode(),
                        b'flash_chord.eval_script_sha256':b'a'*64,
                        b'flash_chord.eval_code_sha256':b'b'*64}),trajectory)
                    metrics=result_dir/'metrics.json'
                    put(metrics,{'checkpoint_sha256':sha(checkpoint),'world_count':1,'reset_mode':'explicit',
                        'environment_steps':step,'non_finite_worlds':[]})
                    put(result_dir/'evaluation.done.json',{'checkpoint_sha256':sha(checkpoint),
                        'evaluator':plan['evaluator'],'input':c['input'],
                        'training_reference':arm_record['training_reference'],
                        'scoring_reference':arm_record['clean_scoring_reference'],'support':arm_record['support'],
                        'trajectory_sha256':sha(trajectory),'metrics_sha256':sha(metrics)})
        (cohort/'RUN_COMPLETE').touch()
        return cohort

    def qualified(self, *args, **kwargs):
        cohort=self.cohort(*args,**kwargs)
        result=self.command('qualify','--out',cohort)
        self.assertEqual(result.returncode,0,result.stderr)
        return cohort

    def merge(self, cohorts, name, rescues=()):
        args=['merge','--tier-root',self.work/'roster','--out',self.work/name]
        for cohort in cohorts: args += ['--cohort',cohort]
        for cohort in rescues: args += ['--rescue',cohort]
        return self.command(*args)

    def test_tier3_five_hosts_from_one_coverage_merge_all_30(self):
        self.roster(30)
        cohorts=[]
        for host in range(5):
            episodes=list(range(host,30,5))
            manifest=self.admitted_manifest(episodes,f't3h{host}')
            cohorts.append(self.qualified(episodes,tier=3,manifest=manifest,name=f't3h{host}'))
        result=self.merge(cohorts,'release')
        self.assertEqual(result.returncode,0,result.stderr)
        selected=json.loads((self.work/'release/selected.json').read_text())
        self.assertEqual(set(selected),set(map(str,range(30))))
        self.assertEqual(len(set(selected.values())),30)
        plan=json.loads((self.work/'release/plan.json').read_text())
        merged=json.loads(Path(plan['manifest']).read_text())  # one frozen manifest naming all 30 inputs for export
        self.assertEqual(len(merged['sequences']),30)
        self.assertEqual(plan['manifest_sha256'],sha(plan['manifest']))

    def test_tier3_different_coverage_or_partial_hosts_still_fail(self):
        self.roster(2)
        first=self.admitted_manifest([0],'host0')
        second=self.admitted_manifest([1],'host1',coverage='b'*64)
        cohorts=[self.qualified([0],tier=3,manifest=first),self.qualified([1],tier=3,manifest=second)]
        for name,items in [('different',cohorts),('missing',cohorts[:1]),('duplicate',[cohorts[0],cohorts[0]])]:
            result=self.merge(items,name)
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertFalse((self.work/name).exists())

    def test_reused_file_or_directory_cannot_turn_k1_into_k3(self):
        self.roster(1)
        for kind in ('file','directory','hardlink'):
            with self.subTest(kind=kind):
                cohort=self.cohort([0])
                evaluations=sorted((cohort/'cohort').glob('*/*/*/evaluation'))
                original=evaluations[0]
                for evaluation in evaluations[1:]:
                    if kind in ('file','hardlink'):
                        target=next(evaluation.rglob('trajectory.parquet'))
                        target.unlink()
                        if kind=='file': target.symlink_to(next(original.rglob('trajectory.parquet')))
                        else: os.link(next(original.rglob('trajectory.parquet')),target)
                    else:
                        shutil.rmtree(evaluation);evaluation.symlink_to(original,target_is_directory=True)
                result=self.command('qualify','--out',cohort)
                self.assertNotEqual(result.returncode,0,'one physical rollout was counted three times')
                self.assertFalse((cohort/'selected.json').exists())

    def test_export_rehashes_frozen_support_before_claiming_an_attempt(self):
        self.roster(1)
        cohort=self.qualified([0])
        support=self.supports[-1]
        support.write_text('#usda 1.0\ndef Cube "unexpected_obstacle" {}\n')
        events=self.work/'export-calls'
        boundary=self.work/'python-boundary'
        boundary.write_text('#!'+sys.executable+'\nimport os,sys\nfrom pathlib import Path\n'
            'with Path(os.environ["EXPORT_EVENTS"]).open("a") as f:f.write(sys.argv[1]+"\\n")\n')
        boundary.chmod(0o755)
        result=self.command('export','--out',cohort,'--work',self.work/'frozen','--commit',
            'https://github.com/fixture/fixture/commit/0123456789abcdef',
            env={'PYTHON':str(boundary),'PACK_PYTHON':str(boundary),'EXPORT_EVENTS':str(events)})
        self.assertNotEqual(result.returncode,0,'export evaluated a different scene from the qualified inputs')
        self.assertFalse(events.exists(),'input custody must fail before GPU evaluation or packing')
        self.assertFalse((cohort/'EXPORT_STARTED').exists())

    def test_qualification_can_resume_after_binding_write_failure_without_reroll(self):
        self.roster(1)
        cohort=self.cohort([0])
        guard=self.work/'fault';guard.mkdir()
        (guard/'sitecustomize.py').write_text('''import os
from pathlib import Path
blocked=Path(os.environ['FAIL_BINDING'])
write_text,replace=Path.write_text,os.replace
def write(self,*a,**k):
 if self==blocked:raise OSError('injected binding write failure')
 return write_text(self,*a,**k)
def rename(src,dst,*a,**k):
 if Path(dst)==blocked:raise OSError('injected binding write failure')
 return replace(src,dst,*a,**k)
Path.write_text,os.replace=write,rename
''')
        before={str(p):sha(p) for p in cohort.rglob('trajectory.parquet')}
        first=self.command('qualify','--out',cohort,env={'PYTHONPATH':str(guard),'FAIL_BINDING':str(cohort/'export-binding.json')})
        self.assertNotEqual(first.returncode,0)
        self.assertIn('injected binding write failure',first.stderr)
        self.assertTrue((cohort/'selected.json').exists())
        self.assertFalse((cohort/'export-binding.json').exists())
        retry=self.command('qualify','--out',cohort)
        self.assertEqual(retry.returncode,0,retry.stderr)
        self.assertEqual(before,{str(p):sha(p) for p in cohort.rglob('trajectory.parquet')})
        selected=json.loads((cohort/'selected.json').read_text())
        binding=json.loads((cohort/'export-binding.json').read_text())
        self.assertEqual(binding['selected_sha256'],sha(cohort/'selected.json'))
        self.assertEqual(binding['checkpoints'],{p:sha(p) for p in selected.values()})

    def test_missing_nonfinite_and_extra_members_fail_without_publishing(self):
        self.roster(1)
        for mutation in ('missing','nonfinite','extra'):
            with self.subTest(mutation=mutation):
                cohort=self.cohort([0])
                trajectory=next(cohort.rglob('trajectory.parquet'))
                if mutation=='missing': trajectory.unlink()
                elif mutation=='nonfinite':
                    table=pq.ParquetFile(trajectory).read()
                    import pyarrow as pa
                    table=table.set_column(table.column_names.index('achieved_position_x'),'achieved_position_x',pa.array([float('nan')]*len(table)))
                    pq.write_table(table,trajectory)
                    stamp=trajectory.with_name('evaluation.done.json')
                    done=json.loads(stamp.read_text());done['trajectory_sha256']=sha(trajectory);put(stamp,done)
                if mutation=='extra':
                    extra=cohort/'cohort/extra';extra.mkdir(exist_ok=True);shutil.copyfile(trajectory,extra/'trajectory.parquet')
                result=self.command('qualify','--out',cohort)
                self.assertNotEqual(result.returncode,0,result.stdout)
                self.assertFalse((cohort/'selected.json').exists())

    def test_selection_uses_all_three_rollouts_and_final_wins_ties(self):
        self.roster(1)
        cohort=self.qualified([0],tier=1,shifts={EARLY:[0.,.08,.08],FINAL:[.04,.04,.04]})
        selected=json.loads((cohort/'selected.json').read_text())
        self.assertTrue(selected['0'].endswith(f'policy_{FINAL}.safetensors'))
        summary=json.loads((cohort/'selection.json').read_text())[0]
        self.assertEqual({step:len(members) for step,members in summary['member_metrics'].items()},
            {str(EARLY):3,str(FINAL):3})
        tied=self.qualified([0],tier=1)
        self.assertTrue(json.loads((tied/'selected.json').read_text())['0'].endswith(f'policy_{FINAL}.safetensors'))


    def test_real_job_files_partition_exactly_across_3_5_7_hosts(self):
        generated=ROOT/'data/brainstorm6/PROD-READY/generated'
        expected={30:{3:[10,10,10],5:[6,6,6,6,6],7:[5,5,4,4,4,4,4]},
                  27:{3:[9,9,9],5:[6,6,5,5,5],7:[4,4,4,4,4,4,3]}}
        for source in (generated/'tier2/jobs.txt',generated/'tier1/remaining.txt'):
            original=source.read_text().splitlines()
            for hosts in (3,5,7):
                out=self.work/source.parent.name/str(hosts)
                result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/split_jobs.py'),str(source),str(hosts),str(out)],
                    capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
                parts=[(out/f'{source.stem}-host{k}-of{hosts}.txt').read_text().splitlines() for k in range(hosts)]
                self.assertEqual([len(part) for part in parts],expected[len(original)][hosts])
                self.assertEqual(sorted(line for part in parts for line in part),sorted(original))
                self.assertEqual(len({line.split('|')[4] for part in parts for line in part}),len(original))

    def test_veto_uses_15cm_tier2_and_10cm_tier3_on_every_member(self):
        self.roster(1)
        for tier,expected in ((2,'orientation'),(3,'catastrophe')):
            cohort=self.cohort([0],tier=tier)
            trajectory=sorted(cohort.rglob('trajectory.parquet'))[1]
            table=pq.ParquetFile(trajectory).read()
            import pyarrow as pa
            changes={'achieved_position_x':.099,'achieved_quaternion_z':1.,'achieved_quaternion_w':0.}
            for column,value in changes.items():
                table=table.set_column(table.column_names.index(column),column,pa.array([value]*len(table)))
            pq.write_table(table,trajectory)
            stamp=trajectory.with_name('evaluation.done.json')
            done=json.loads(stamp.read_text());done['trajectory_sha256']=sha(trajectory);put(stamp,done)
            result=self.command('qualify','--out',cohort)
            self.assertEqual(result.returncode,0,result.stderr)
            veto=json.loads((cohort/'veto.json').read_text())['candidates'][0]
            self.assertEqual(veto['status'],expected)
            self.assertEqual([m['catastrophe'] for m in veto['members']], [False,tier==3,False])
            member=veto['members'][1]
            self.assertAlmostEqual(member['mean_position_m'],.099,places=6)
            self.assertAlmostEqual(member['mean_orientation_rad'],np.pi,places=6)
            self.assertGreater(member['mppe_cm'],10)
            self.assertLess(member['mppe_cm'],15)

    def test_tier3_rescue_accepts_same_inputs_and_rejects_changed_inputs(self):
        self.roster(1)
        original_manifest=self.admitted_manifest([0],'primary')
        primary=self.qualified([0],tier=3,manifest=original_manifest,shifts={SHORT:[.2,.2,.2]})
        same_manifest=self.admitted_manifest([0],'rescue_same')
        rescue=self.qualified([0],tier=3,manifest=same_manifest)
        result=self.merge([primary],'same-input-release',rescues=[rescue])
        self.assertEqual(result.returncode,0,result.stderr)
        chosen=json.loads((self.work/'same-input-release/selected.json').read_text())
        self.assertEqual(chosen,json.loads((rescue/'selected.json').read_text()))
        changed_manifest=self.admitted_manifest([0],'rescue_changed')
        raw=json.loads(changed_manifest.read_text())
        arm=raw['sequences'][0]['arms']['tier3_synthetic']
        support=Path(arm['support']['path']);support.write_text('#usda 1.0\ndef Cube "different" {}\n')
        arm['support']['sha256']=sha(support);put(changed_manifest,raw)
        changed=self.qualified([0],tier=3,manifest=changed_manifest)
        result=self.merge([primary],'changed-input-release',rescues=[changed])
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assertFalse((self.work/'changed-input-release').exists())


if __name__=='__main__':unittest.main(verbosity=2)
