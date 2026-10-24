#!/usr/bin/env python3
"""Frozen fresh evaluation cohorts, whole-checkpoint selection, and a fixed fresh export.

prepare -> run -> qualify. A failed/missing member blocks qualification; retrying a failed
cohort requires a new declared experiment, never an automatic score-based reroll.
Tier 2/3 own-input vetoes do not block coverage: a vetoed episode keeps its final with failure status
(veto.json); merge --rescue moves it to a newly trained checkpoint only if that one's veto class is strictly
better (pass < orientation-only < catastrophe), earliest --rescue first.
"""
import argparse, hashlib, json, math, os, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_repeats import prepare
from pilot_queue import read_arms, read_inputs, evaluator_identity
from score_rollout import score_rollout, load_reference, load_rollout
import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
TIER1_RULE='final unless an earlier checkpoint has mean AUC >= final +.05 and no lower mean SP-SR/MP-SR and no higher mean MPPE'
def tier1_choice(metrics,final):
    # Evaluation noise moves one rollout's AUC by ~.05 (EVAL-NOISE); a k=3 AUC-only argmax picks noise and can trade a
    # success for a tail. Leave the final unless the earlier checkpoint is clearly better on AUC and no worse elsewhere.
    mean=lambda step,key:sum(m[key] for m in metrics[step])/len(metrics[step])
    clear=[step for step in metrics if step!=final and mean(step,'AUC')>=mean(final,'AUC')+.05
           and mean(step,'SP-SR')>=mean(final,'SP-SR') and mean(step,'MP-SR')>=mean(final,'MP-SR') and mean(step,'MPPE')<=mean(final,'MPPE')]
    return max(clear,key=lambda step:mean(step,'AUC')) if clear else final
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def invoke(cmd,env=None):subprocess.run(cmd,cwd=ROOT,env={**os.environ,**(env or {})},check=True)

def stage(args):
    if args.k<1:raise ValueError('k must be positive')
    if args.ep18:
        from ep18_pilot import validate_jobs
        validate_jobs(args.jobs)
        if args.tier!=3 or args.k!=3:raise ValueError('ep18 requires Tier 3 and exactly k=3')
    if args.out.exists():raise ValueError('cohort root must be fresh; retained attempts cannot be overwritten')
    candidates=[]
    for line in args.jobs.read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,arms_file,arm,label,seq,steps=line.split('|')
        parsed=read_arms(ROOT/'scripts'/arms_file,[]);total=parsed[arm][1]
        expected=[330012672,349999104] if args.tier==1 else [total]
        if args.tier==1 and (arm!='F6-rebalance-tail' or label!='clean'):
            raise ValueError('Tier 1 requires frozen F6/clean')
        if args.tier==2 and (arm not in ('T2C-contact','T2C-contact-seed43') or label!='raw_repaired'):
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
    if not args.ep18 and len({c['sequence'] for c in candidates})!=len(candidates):raise ValueError('one checkpoint choice per episode/root required')
    if args.ep18:
        for c in candidates:c['episode']=18
    else:
        metadata=list(map(json.loads,(args.tier_root/'meta/episodes_metadata.jsonl').read_text().splitlines()))
        lookup={r['sequence_id']:r['episode_index'] for r in metadata}
        if len(lookup)!=len(metadata):raise ValueError('duplicate episode metadata')
        for c in candidates:c['episode']=lookup[c['sequence']]
    plan={'repeats':args.k,'tier':args.tier,'candidates':candidates,'manifest':str(args.manifest.resolve()),
          'manifest_sha256':digest(args.manifest),'source_root':str(args.source.resolve()),'ep18':args.ep18,
          'selection':TIER1_RULE if args.tier==1 else 'all k pass; final only',
          'canonical_export':('none; initial evaluation is fixed member 1' if args.ep18 else
                              'one new evaluate-only output per selected checkpoint; no score-based retries'),
          'evaluator':evaluator_identity(Path(os.environ.get('FLASH_ROOT',ROOT/'external/video_to_data/robotic_grounding/flash_chord')))}
    args.out.mkdir(parents=True)
    prepare(plan,args.source.resolve(),args.out.resolve())
    if args.ep18:
        # The training invocation already evaluated the final once in its own process.
        # Freeze it as member 1, and issue only the remaining two fresh processes.
        for c in candidates:
            source=args.source/c['source']/c['input']/'evaluation'
            target=args.out/'cohort'/f"{c['name']}-r01"/c['arm']/c['input']/'evaluation'
            shutil.copytree(source,target)
        jobs=args.out/'jobs-cohort.txt'
        jobs.write_text('\n'.join(line for line in jobs.read_text().splitlines()
                                 if not line.split('|')[0].endswith('-r01'))+'\n')
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
    if plan['ep18']:
        qualify_ep18(args.out,plan);return
    if not (args.out/'RUN_COMPLETE').exists():raise ValueError('cohort not completely executed')
    checked={r['name']:r for r in json.loads((args.out/'preparation.json').read_text())['checked']}
    actual={p.resolve() for p in (args.out/'cohort').rglob('trajectory.parquet')}
    expected=set();mapping={};summary=[];vetoes=[]
    for c in plan['candidates']:
        values={step:[] for step in c['steps']};members=[]
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
                    limit=10 if plan['tier']==3 else 15  # Tier 3 protocol: self-MPPE <=10 cm (T3-POLICY PROTOCOL-qualified)
                    reference,_,_,body=load_reference(Path(job['training']['path']))
                    achieved=load_rollout(trajectory,len(reference),body)
                    q=reference[:,3:]/np.linalg.norm(reference[:,3:],axis=1,keepdims=True)
                    r=achieved[:,3:]/np.linalg.norm(achieved[:,3:],axis=1,keepdims=True)
                    tracking={'mean_position_m':float(np.linalg.norm(reference[:,:3]-achieved[:,:3],axis=1).mean()),
                              'mean_orientation_rad':float((2*np.arccos(np.clip(np.abs((q*r).sum(axis=1)),0,1))).mean())}
                    # Position/MPPE failures are catastrophes; orientation against a noisy input also vetoes
                    # clean successes, so it ranks last for reseeds (brainstorm 7 T2-STRATEGY F1).
                    members.append({'repeat':repeat,**tracking,'mppe_cm':mean['MPPE'],
                                    'catastrophe':tracking['mean_position_m']>.1 or mean['MPPE']>limit,
                                    'orientation':tracking['mean_orientation_rad']>.5})
                values[step].append({key:mean[key] for key in ('AUC','SP-SR','MP-SR','MPPE')})
        means={step:sum(m['AUC'] for m in v)/len(v) for step,v in values.items()}
        chosen=tier1_choice(values,c['final_step']) if plan['tier']==1 else c['final_step']
        source=Path(plan['source_root'])/c['source'];checkpoint=source/c['input']/'train'/c['sequence']/f'policy_{chosen}.safetensors'
        mapping[str(c['episode'])]=str(checkpoint.resolve())
        summary.append({'name':c['name'],'auc_means':means,'selected_step':chosen,'members':plan['repeats'],'member_metrics':values})
        if plan['tier']!=1:
            status=('catastrophe' if any(m['catastrophe'] for m in members) else
                    'orientation' if any(m['orientation'] for m in members) else 'pass')
            summary[-1]['veto']=status
            vetoes.append({'episode':c['episode'],'sequence':c['sequence'],'name':c['name'],'status':status,
                           'checkpoint_sha256':digest(checkpoint),'members':members})
    if actual!=expected:raise ValueError('missing or extra cohort rollout; exactly k required')
    if (args.out/'selected.json').exists():raise ValueError('selection already frozen')
    write(args.out/'selected.json',mapping);write(args.out/'selection.json',summary)
    if plan['tier']!=1:
        order={'catastrophe':0,'orientation':1}
        write(args.out/'veto.json',{'rule':f"veto iff any member: position >.10 m, orientation >.5 rad or self-MPPE >{10 if plan['tier']==3 else 15} cm; "
                                    'reseed order: catastrophe class, then orientation-only, each by episode index',
                                    'reseed_order':[v['episode'] for v in sorted(vetoes,key=lambda v:(order.get(v['status'],2),v['episode']))
                                                    if v['status']!='pass'],'candidates':vetoes})
    write(args.out/'export-binding.json',{'selected_sha256':digest(args.out/'selected.json'),
          **({'veto_sha256':digest(args.out/'veto.json')} if plan['tier']!=1 else {}),
          'checkpoints':{path:digest(path) for path in mapping.values()},'predetermined_root':'export'})

def bound_vetoes(cohort,selected,binding):
    # Rescues are decided by veto.json, so it is bound and checked like the selection.
    if digest(cohort/'veto.json')!=binding.get('veto_sha256'):raise ValueError(f'{cohort}/veto.json changed or unbound')
    vetoes={str(v['episode']):v for v in json.loads((cohort/'veto.json').read_text())['candidates']}
    if set(vetoes)!=set(selected) or any(v['checkpoint_sha256']!=binding['checkpoints'][selected[e]] for e,v in vetoes.items()):
        raise ValueError(f'{cohort}/veto.json does not match its bound selection')
    return vetoes

def qualify_ep18(root,plan):
    import pyarrow.parquet as pq
    from ep18_pilot import PAIR
    if plan['repeats']!=3 or {c['name'] for c in plan['candidates']}!=set(PAIR):
        raise ValueError('invalid registered ep18 cohort')
    checked={r['name']:r for r in json.loads((root/'preparation.json').read_text())['checked']}
    results=[];expected=set()
    for c in plan['candidates']:
        members=[];identities=[]
        for repeat in range(1,4):
            result=root/'cohort'/f"{c['name']}-r{repeat:02d}"/c['arm']/c['input']/'evaluation'/c['sequence']/str(c['final_step'])
            trajectory=result/'trajectory.parquet';expected.add(trajectory.resolve())
            member={'repeat':repeat,'rollout':str(trajectory),'go':False}
            try:
                if trajectory.is_symlink():raise ValueError('separate evaluation artifacts required')
                job=read_inputs(Path(plan['manifest']),[c['sequence']],[c['input']])[f"{c['input']}/{c['sequence']}"]
                checkpoint=root/'cohort'/f"{c['name']}-r{repeat:02d}"/c['arm']/c['input']/'train'/c['sequence']/f"policy_{c['final_step']}.safetensors"
                done=json.loads((result/'evaluation.done.json').read_text())
                metrics=json.loads((result/'metrics.json').read_text())
                h=digest(checkpoint)
                identity={'checkpoint_sha256':h,'evaluator':plan['evaluator'],'input':c['input'],
                    'training_reference':job['training'],'scoring_reference':job['scoring'],'support':job['support'],
                    'trajectory_sha256':digest(trajectory),'metrics_sha256':digest(result/'metrics.json')}
                if done!=identity or h!=checked[c['name']]['checkpoint_sha256'][str(c['final_step'])]:
                    raise ValueError('checkpoint/runtime/input provenance mismatch')
                if (metrics.get('checkpoint_sha256')!=h or metrics.get('world_count')!=1 or
                    metrics.get('reset_mode')!='explicit' or metrics.get('environment_steps')!=c['final_step'] or
                    metrics.get('non_finite_worlds')!=[]):raise ValueError('invalid evaluator protocol')
                metadata=pq.ParquetFile(trajectory).schema_arrow.metadata or {}
                keys=[b'flash_chord.checkpoint_sha256',b'flash_chord.eval_code_sha256',b'flash_chord.eval_script_sha256']
                if any(not metadata.get(k) for k in keys) or metadata[keys[0]].decode()!=h:
                    raise ValueError('invalid evaluator parquet provenance')
                identities.append(tuple(metadata[k] for k in keys))
                report=score_rollout(trajectory,Path(job['scoring']['path']))
                reference,_,_,body=load_reference(Path(job['training']['path']))
                if len(reference)!=858:raise ValueError('ep18 must retain all 858 source frames')
                achieved=load_rollout(trajectory,858,body)
                # Faithful carry checks from T3-POLICY/analyze_ep18.py inspect()/qualify().
                drop=achieved[:300,2] < reference[0,2]-.2
                early_drop=bool(np.any(np.convolve(drop.astype(int),np.ones(10,int),'valid')==10))
                displacement=float(achieved[450,0]-achieved[300,0])
                error450=float(np.linalg.norm(achieved[450,:3]-reference[450,:3]))
                mppe=report['metrics']['MPPE']
                if not all(math.isfinite(v) for v in report['metrics'].values()):raise ValueError('non-finite score')
                member.update(frames=858,mppe_cm=mppe,early_drop=early_drop,
                    x300_to450_m=displacement,position_error450_m=error450,
                    go=bool(mppe<=10 and not early_drop and abs(displacement)>=.6 and error450<=.2))
            except (OSError,ValueError,KeyError,TypeError) as error:
                member['error']=str(error)
            members.append(member)
        same_evaluator=len(identities)==3 and len(set(identities))==1
        results.append({'name':c['name'],'members':members,'same_checkpoint_and_evaluator':same_evaluator,
                        'go':same_evaluator and all(m['go'] for m in members)})
    actual={p.resolve() for p in (root/'cohort').rglob('trajectory.parquet')}
    if actual-expected:
        for result in results:result.update(go=False,error='extra cohort rollout; exactly k required')
    output=root/'qualification.json'
    if output.exists():raise ValueError('qualification already frozen')
    write(output,{'candidates':results,'rule':'all three: 858 valid frames, MPPE <=10cm, no early drop, carry >=.60m, frame450 error <=.20m'})
    if not all(r['go'] for r in results):raise ValueError('ep18 pair contains failed qualification; see qualification.json')


def merge(args):
    if args.out.exists():raise ValueError('fresh release root required')
    metadata=list(map(json.loads,(args.tier_root/'meta/episodes_metadata.jsonl').read_text().splitlines()))
    total=json.loads((args.tier_root/'meta/info.json').read_text())['total_episodes']
    if sorted(r['episode_index'] for r in metadata)!=list(range(total)):raise ValueError('invalid release roster')
    mapping={};parents=[];common=None;status={}
    for cohort in args.cohort:
        plan=checked_plan(cohort)
        key=(plan['tier'],plan['manifest_sha256'],plan['evaluator'],plan['repeats'])
        if common is not None and common!=key:raise ValueError('release cohort protocol/input/runtime differs')
        common=key
        selected=json.loads((cohort/'selected.json').read_text());binding=json.loads((cohort/'export-binding.json').read_text())
        if digest(cohort/'selected.json')!=binding['selected_sha256']:raise ValueError('selection changed')
        if any(digest(p)!=h for p,h in binding['checkpoints'].items()):raise ValueError('selected policy changed')
        if set(mapping)&set(selected):raise ValueError('duplicate release episodes')
        if plan['tier']!=1:status.update(bound_vetoes(cohort,selected,binding))
        mapping.update(selected);parents.append({'cohort':str(cohort.resolve()),'selection_sha256':binding['selected_sha256']})
    if set(mapping)!=set(map(str,range(total))):raise ValueError('release needs every episode exactly once')
    rank={'pass':0,'orientation':1,'catastrophe':2}
    outcome={episode:{'original':v['status'],'final':v['status']} for episode,v in status.items()}
    evaluated={episode:{v['checkpoint_sha256']} for episode,v in status.items()}
    for cohort in args.rescue or []:  # earliest seed first; a later one replaces only a strictly worse class
        rescue=checked_plan(cohort)
        if (rescue['tier'],rescue['manifest_sha256'],rescue['evaluator'],rescue['repeats'])!=common or common[0]==1:
            raise ValueError('rescue cohort protocol/input/runtime differs, or Tier 1')
        selected=json.loads((cohort/'selected.json').read_text());binding=json.loads((cohort/'export-binding.json').read_text())
        if digest(cohort/'selected.json')!=binding['selected_sha256']:raise ValueError('selection changed')
        if any(digest(p)!=h for p,h in binding['checkpoints'].items()):raise ValueError('selected policy changed')
        for episode,v in bound_vetoes(cohort,selected,binding).items():
            # Each checkpoint gets one cohort per episode, whether it repeats the original or an earlier rescue.
            if v['checkpoint_sha256'] in evaluated[episode]:
                raise ValueError('rescue needs a newly trained checkpoint, never a re-evaluation')
            evaluated[episode].add(v['checkpoint_sha256'])
            if rank[v['status']]<rank[outcome[episode]['final']]:
                mapping[episode]=selected[episode];outcome[episode].update(final=v['status'],cohort=str(cohort.resolve()))
        parents.append({'cohort':str(cohort.resolve()),'selection_sha256':binding['selected_sha256'],'rescue':True})
    args.out.mkdir(parents=True)
    write(args.out/'selected.json',mapping)
    write(args.out/'export-binding.json',{'selected_sha256':digest(args.out/'selected.json'),
          'checkpoints':{path:digest(path) for path in mapping.values()},'predetermined_root':'export'})
    if outcome:write(args.out/'veto.json',{'episodes':outcome})
    write(args.out/'plan.json',{**plan,'source_root':str(ROOT/'runs'),'parents':parents})


def export(args):
    plan=checked_plan(args.out)
    if plan['ep18']:raise ValueError('ep18 pilot has no submission export')
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
    m=sub.add_parser('merge');m.add_argument('--cohort',type=Path,action='append',required=True);m.add_argument('--rescue',type=Path,action='append')
    m.add_argument('--tier-root',type=Path,required=True);m.add_argument('--out',type=Path,required=True)
    e=sub.add_parser('export');e.add_argument('--out',type=Path,required=True);e.add_argument('--work',type=Path,required=True);e.add_argument('--commit',required=True)
    s.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    try:{'prepare':stage,'run':execute,'qualify':qualify,'export':export,'merge':merge}[args.action](args)
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as e:p.exit(1,f'error: {e}\n')
if __name__=='__main__':main()
