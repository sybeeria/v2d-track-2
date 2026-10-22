#!/usr/bin/env python3
"""Frozen fresh evaluation cohorts, whole-checkpoint selection, and a fixed fresh export.

prepare -> run -> qualify. A failed/missing member blocks qualification; retrying a failed
cohort requires a new declared experiment, never an automatic score-based reroll.
"""
import argparse, hashlib, json, math, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_repeats import prepare
from pilot_queue import read_arms, read_inputs, evaluator_identity
from score_rollout import score_rollout, load_reference, load_rollout
import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def invoke(cmd,env=None):subprocess.run(cmd,cwd=ROOT,env={**os.environ,**(env or {})},check=True)

def stage(args):
    if args.k<1:raise ValueError('k must be positive')
    if args.out.exists():raise ValueError('cohort root must be fresh; retained attempts cannot be overwritten')
    candidates=[]
    for line in args.jobs.read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,arms_file,arm,label,seq,steps=line.split('|')
        parsed=read_arms(ROOT/'scripts'/arms_file,[]);total=parsed[arm][1]
        expected=[330012672,349999104] if args.tier==1 else [total]
        if args.tier==1 and (arm!='F6-rebalance-tail' or label!='clean'):
            raise ValueError('Tier 1 requires frozen F6/clean')
        if args.tier==2 and (arm!='T2C-contact' or label!='raw_repaired'):
            raise ValueError('Tier 2 requires frozen T2C-contact/raw_repaired')
        if args.tier in (2,3) and total!=49999872 and not args.ep18:
            raise ValueError('Tier 2/3 production uses the frozen 50M final')
        if args.tier in (2,3):
            raw=json.loads(args.manifest.read_text())
            for entry in raw['sequences']:
                if entry['sequence']==seq and label in entry['arms']:
                    for key in ('training_reference','clean_scoring_reference'):
                        path=(ROOT/entry['arms'][label][key]['path']).resolve()
                        if any(part=='tier1' or 'tier_1' in part for part in path.parts):
                            raise ValueError('hard-tier input path names Tier 1')
        inputs=read_inputs(args.manifest,[seq],[label])
        if args.tier in (2,3) and any(j['training']!=j['scoring'] for j in inputs.values()):
            raise ValueError('hard tiers require own-input scoring')
        candidates.append(dict(name=name,source=f'{name}/{arm}',group='cohort',arm=arm,
            arms_file=arms_file,input=label,sequence=seq,steps=expected,final_step=total))
    if not candidates:raise ValueError('empty jobs')
    if len({c['sequence'] for c in candidates})!=len(candidates):raise ValueError('one checkpoint choice per episode/root required')
    metadata=list(map(json.loads,(args.tier_root/'meta/episodes_metadata.jsonl').read_text().splitlines()))
    lookup={r['sequence_id']:r['episode_index'] for r in metadata}
    if len(lookup)!=len(metadata):raise ValueError('duplicate episode metadata')
    for c in candidates:c['episode']=lookup[c['sequence']]
    if args.ep18 and (args.tier!=3 or any(c['episode']!=18 for c in candidates)):
        raise ValueError('--ep18 is limited to the registered Tier 3 episode 18 pilot')
    plan={'repeats':args.k,'tier':args.tier,'candidates':candidates,'manifest':str(args.manifest.resolve()),
          'manifest_sha256':digest(args.manifest),'source_root':str(args.source.resolve()),'ep18':args.ep18,
          'selection':'mean AUC; tie final' if args.tier==1 else 'all k pass; final only',
          'canonical_export':'one new evaluate-only output per selected checkpoint; no score-based retries',
          'evaluator':evaluator_identity(Path(os.environ.get('FLASH_ROOT',ROOT/'external/video_to_data/robotic_grounding/flash_chord')))}
    args.out.mkdir(parents=True)
    prepare(plan,args.source.resolve(),args.out.resolve())
    write(args.out/'plan.json',plan)

def checked_plan(root):
    plan=json.loads((root/'plan.json').read_text())
    if digest(plan['manifest'])!=plan['manifest_sha256']:raise ValueError('frozen manifest changed')
    return plan

def execute(args):
    plan=checked_plan(args.out)
    if (args.out/'RUN_STARTED').exists():raise ValueError('cohort already attempted; retain it, diagnose failure')
    with (args.out/'RUN_STARTED').open('x') as claim:claim.write('evaluate-only cohort attempt\n')
    invoke(['bash',str(ROOT/'scripts/test_day.sh'),str(args.gpus),str(args.out/'cohort')],
           {'PILOT_EVALUATE_ONLY':'1','JOBS_FILE':str(args.out/'jobs-cohort.txt'),'MANIFEST':plan['manifest']})
    (args.out/'RUN_COMPLETE').touch()

def qualify(args):
    plan=checked_plan(args.out)
    if not (args.out/'RUN_COMPLETE').exists():raise ValueError('cohort not completely executed')
    checked={r['name']:r for r in json.loads((args.out/'preparation.json').read_text())['checked']}
    actual={p.resolve() for p in (args.out/'cohort').rglob('trajectory.parquet')}
    expected=set();mapping={};summary=[]
    for c in plan['candidates']:
        values={step:[] for step in c['steps']}
        for repeat in range(1,plan['repeats']+1):
            base=args.out/'cohort'/f"{c['name']}-r{repeat:02d}"/c['arm']/c['input']
            for step in c['steps']:
                result=base/'evaluation'/c['sequence']/str(step)
                trajectory=result/'trajectory.parquet';expected.add(trajectory.resolve())
                checkpoint=base/'train'/c['sequence']/f'policy_{step}.safetensors'
                done=json.loads((result/'evaluation.done.json').read_text());metrics=json.loads((result/'metrics.json').read_text())
                job=read_inputs(Path(plan['manifest']),[c['sequence']],[c['input']])[f"{c['input']}/{c['sequence']}"]
                identity={'checkpoint_sha256':digest(checkpoint),'evaluator':plan['evaluator'],'input':c['input'],
                          'training_reference':job['training'],'scoring_reference':job['scoring'],'support':job['support'],
                          'trajectory_sha256':digest(trajectory),'metrics_sha256':digest(result/'metrics.json')}
                if done!=identity or done['checkpoint_sha256']!=checked[c['name']]['checkpoint_sha256'][str(step)]:
                    raise ValueError('checkpoint/runtime/input provenance mismatch')
                if (metrics.get('checkpoint_sha256')!=done['checkpoint_sha256'] or metrics.get('world_count')!=1
                    or metrics.get('reset_mode')!='explicit' or metrics.get('environment_steps')!=step
                    or metrics.get('non_finite_worlds')!=[]):raise ValueError('invalid evaluator protocol')
                # Re-read the retained achieved trajectory; never trust an editable summary alone.
                report=score_rollout(trajectory,Path(job['scoring']['path']))
                if not all(math.isfinite(v) for v in report['metrics'].values()):raise ValueError('non-finite score')
                mean=report['metrics']
                if plan['tier']!=1:
                    limit=10 if plan['ep18'] else 15
                    reference,_,_,body=load_reference(Path(job['training']['path']))
                    achieved=load_rollout(trajectory,len(reference),body)
                    q=reference[:,3:]/np.linalg.norm(reference[:,3:],axis=1,keepdims=True)
                    r=achieved[:,3:]/np.linalg.norm(achieved[:,3:],axis=1,keepdims=True)
                    tracking={'mean_position_m':float(np.linalg.norm(reference[:,:3]-achieved[:,:3],axis=1).mean()),
                              'mean_orientation_rad':float((2*np.arccos(np.clip(np.abs((q*r).sum(axis=1)),0,1))).mean())}
                    if not (tracking['mean_position_m']<=.1 and tracking['mean_orientation_rad']<=.5 and mean['MPPE']<=limit):
                        raise ValueError(f"veto {c['name']} repeat {repeat}: {tracking}, {mean}")
                values[step].append(mean['AUC'])
        means={step:sum(v)/len(v) for step,v in values.items()}
        chosen=max(c['steps'],key=lambda step:(means[step],step)) if plan['tier']==1 else c['final_step']
        source=Path(plan['source_root'])/c['source'];checkpoint=source/c['input']/'train'/c['sequence']/f'policy_{chosen}.safetensors'
        mapping[str(c['episode'])]=str(checkpoint.resolve())
        summary.append({'name':c['name'],'auc_means':means,'selected_step':chosen,'members':plan['repeats']})
    if actual!=expected:raise ValueError('missing or extra cohort rollout; exactly k required')
    if (args.out/'selected.json').exists():raise ValueError('selection already frozen')
    write(args.out/'selected.json',mapping);write(args.out/'selection.json',summary)
    write(args.out/'export-binding.json',{'selected_sha256':digest(args.out/'selected.json'),
          'checkpoints':{path:digest(path) for path in mapping.values()},'predetermined_root':'export'})

def merge(args):
    if args.out.exists():raise ValueError('fresh release root required')
    metadata=list(map(json.loads,(args.tier_root/'meta/episodes_metadata.jsonl').read_text().splitlines()))
    total=json.loads((args.tier_root/'meta/info.json').read_text())['total_episodes']
    if sorted(r['episode_index'] for r in metadata)!=list(range(total)):raise ValueError('invalid release roster')
    mapping={};parents=[];common=None
    for cohort in args.cohort:
        plan=checked_plan(cohort)
        key=(plan['tier'],plan['manifest_sha256'],plan['evaluator'],plan['repeats'])
        if common is not None and common!=key:raise ValueError('release cohort protocol/input/runtime differs')
        common=key
        selected=json.loads((cohort/'selected.json').read_text());binding=json.loads((cohort/'export-binding.json').read_text())
        if digest(cohort/'selected.json')!=binding['selected_sha256']:raise ValueError('selection changed')
        if any(digest(p)!=h for p,h in binding['checkpoints'].items()):raise ValueError('selected policy changed')
        if set(mapping)&set(selected):raise ValueError('duplicate release episodes')
        mapping.update(selected);parents.append({'cohort':str(cohort.resolve()),'selection_sha256':binding['selected_sha256']})
    if set(mapping)!=set(map(str,range(total))):raise ValueError('release needs every episode exactly once')
    args.out.mkdir(parents=True)
    write(args.out/'selected.json',mapping)
    write(args.out/'export-binding.json',{'selected_sha256':digest(args.out/'selected.json'),
          'checkpoints':{path:digest(path) for path in mapping.values()},'predetermined_root':'export'})
    write(args.out/'plan.json',{**plan,'source_root':str(ROOT/'runs'),'parents':parents})


def export(args):
    plan=checked_plan(args.out)
    binding=json.loads((args.out/'export-binding.json').read_text())
    if evaluator_identity(Path(os.environ.get('FLASH_ROOT',ROOT/'external/video_to_data/robotic_grounding/flash_chord')))!=plan['evaluator']:
        raise ValueError('evaluator/runtime changed after cohort')
    if digest(args.out/'selected.json')!=binding['selected_sha256'] or any(digest(p)!=h for p,h in binding['checkpoints'].items()):
        raise ValueError('frozen selection/checkpoint changed')
    if (args.out/'EXPORT_STARTED').exists() or (args.out/'export').exists():raise ValueError('export already attempted; retain predetermined attempt')
    with (args.out/'EXPORT_STARTED').open('x') as claim:claim.write('predetermined export attempt\n')
    invoke(['bash',str(ROOT/'scripts/eval_and_pack.sh'),f"tier{plan['tier']}",plan['source_root'],str(args.out/'export'),args.commit,str(args.work)],
           {'CHECKPOINT_MAP':str(args.out/'selected.json')})
    (args.out/'EXPORT_COMPLETE').touch()

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('prepare');s.add_argument('--jobs',type=Path,required=True);s.add_argument('--manifest',type=Path,required=True)
    s.add_argument('--source',type=Path,required=True);s.add_argument('--tier-root',type=Path,required=True);s.add_argument('--tier',type=int,choices=[1,2,3],required=True)
    s.add_argument('--k',type=int,default=3);s.add_argument('--ep18',action='store_true')
    for action in ['run','qualify']:
        child=sub.add_parser(action);child.add_argument('--out',type=Path,required=True)
        if action=='run':child.add_argument('--gpus',type=int,default=1)
    m=sub.add_parser('merge');m.add_argument('--cohort',type=Path,action='append',required=True);m.add_argument('--tier-root',type=Path,required=True);m.add_argument('--out',type=Path,required=True)
    e=sub.add_parser('export');e.add_argument('--out',type=Path,required=True);e.add_argument('--work',type=Path,required=True);e.add_argument('--commit',required=True)
    s.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    try:{'prepare':stage,'run':execute,'qualify':qualify,'export':export,'merge':merge}[args.action](args)
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as e:p.exit(1,f'error: {e}\n')
if __name__=='__main__':main()
