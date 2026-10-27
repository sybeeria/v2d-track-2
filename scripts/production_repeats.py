#!/usr/bin/env python3
"""Frozen fresh evaluation cohorts, whole-checkpoint selection, and a fixed fresh export.

prepare -> run -> qualify, always k=3. A Tier 2/3 member whose evaluation ran and failed (its runner finished
without sealing it) is a catastrophe only with matching attempt custody; missing attempts, or any failed Tier 1 member, block
qualification. Retrying requires a new declared experiment, never an automatic score-based reroll.
Tier 2/3 own-input vetoes do not block coverage: a vetoed episode keeps its final with failure status
(veto.json); merge --rescue moves it to a newly trained checkpoint only if that one's veto class is strictly
better (pass < orientation-only < catastrophe), earliest --rescue first.
"""
import argparse, hashlib, json, math, os, re, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_repeats import prepare
from pilot_queue import read_arms, read_inputs, evaluator_identity, checked_record
from score_rollout import score_rollout, load_reference, load_rollout, write_json, metric_modules
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
def write(p,value):write_json(p,value)
def invoke(cmd,env=None):subprocess.run(cmd,cwd=ROOT,env={**os.environ,**(env or {})},check=True)

def stage(args):
    if args.k!=3:raise ValueError('production requires the frozen k=3')
    if args.ep18:
        from ep18_pilot import validate_jobs
        validate_jobs(args.jobs)
        if args.tier!=3:raise ValueError('ep18 requires Tier 3')
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
        registered_rescue=False
        if args.tier in (2,3):
            raw=json.loads(args.manifest.read_text())
            if args.tier==3 and not args.ep18:
                from ep18_pilot import SEQUENCE,LABEL,PAIR,ARM_SHA256
                rescue_arm,rescue_step=PAIR['ep18-qualified350']
                registered_rescue=(seq==SEQUENCE and label==LABEL and arm==rescue_arm and total==rescue_step
                    and raw.get('tier3_coverage',{}).get('pilot_decision',{}).get('decision')=='selective rescue'
                    and digest(ROOT/'scripts'/arms_file)==ARM_SHA256[rescue_arm])
            for entry in raw['sequences']:
                if entry['sequence']==seq and label in entry['arms']:
                    for key in ('training_reference','clean_scoring_reference'):
                        path=(ROOT/entry['arms'][label][key]['path']).resolve()
                        if any(part=='tier1' or 'tier_1' in part for part in path.parts):
                            raise ValueError('hard-tier input path names Tier 1')
        if args.tier in (2,3) and total!=49999872 and not args.ep18 and not registered_rescue:
            raise ValueError('Tier 2/3 production uses the frozen 50M final, except the registered ep18 selective rescue')
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
        for c in candidates:
            c['episode']=lookup[c['sequence']]
            if args.tier==1:
                row=next(r for r in metadata if r['episode_index']==c['episode'])
                mesh=(args.tier_root/row['mesh']).resolve()
                c['scoring_mesh']={'path':str(mesh),'sha256':digest(mesh)}
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
    if plan['repeats']!=3:raise ValueError('production requires the frozen k=3')
    if digest(plan['manifest'])!=plan['manifest_sha256']:raise ValueError('frozen manifest changed')
    if any(digest(path)!=h for path,h in plan.get('input_manifests',{}).items()):
        raise ValueError('frozen parent manifest changed')
    return plan

def sequence_inputs(entry):
    # Coverage shards name host-specific paths; the bytes they bind must still match across primary and rescue.
    return (entry['frames'],{label:(arm['training_reference']['sha256'],arm['clean_scoring_reference']['sha256'],
            arm['support']['sha256'],arm['support']['enabled']) for label,arm in entry['arms'].items()},
            sorted(asset['sha256'] for asset in entry.get('assets',[])))

def regular_member(root,path):
    if not path.is_file() or path.resolve()!=root.resolve()/path.relative_to(root) or path.stat().st_nlink!=1:
        raise ValueError('separate regular cohort artifacts required')

def checked_attempt(root,result,identity):
    attempt=result/'evaluation.attempt.json'
    if not attempt.exists() and not attempt.is_symlink():return False
    regular_member(root,attempt)
    if json.loads(attempt.read_text())!=identity:raise ValueError('checkpoint/runtime/input attempt provenance mismatch')
    return True

def failed_member(member,result,c,step):
    # pilot_queue creates result before exec; only the evaluator's Hydra startup writes this config.
    startup=result/'hydra/.hydra/config.yaml'
    if ((result/'evaluation.done.json').exists() or not startup.is_file() or not startup.stat().st_size
        or not (member/'summary.json').is_file()):return False
    regular_member(member,startup)
    rows=json.loads((member/'summary.json').read_text())['candidates']
    return not any((r['arm'],r['input'],r['sequence'],r['step'])==(c['arm'],c['input'],c['sequence'],step) for r in rows)

def execute(args):
    plan=checked_plan(args.out)
    if (args.out/'RUN_STARTED').exists():raise ValueError('cohort already attempted; retain it, diagnose failure')
    with (args.out/'RUN_STARTED').open('x') as claim:claim.write('evaluate-only cohort attempt\n')
    # test_day attempts every member even after one fails; qualify classifies each member from its own receipts.
    try:invoke(['bash',str(ROOT/'scripts/test_day.sh'),str(args.gpus),str(args.out/'cohort')],
               {'PILOT_EVALUATE_ONLY':'1','JOBS_FILE':str(args.out/'jobs-cohort.txt'),'MANIFEST':plan['manifest']});status=0
    except subprocess.CalledProcessError as error:status=error.returncode
    (args.out/'RUN_COMPLETE').write_text(f'test_day exit {status}\n')

def qualify(args):
    plan=checked_plan(args.out)
    if plan['ep18']:
        qualify_ep18(args.out,plan);return
    if not (args.out/'RUN_COMPLETE').exists():raise ValueError('cohort not completely executed')
    preparation=json.loads((args.out/'preparation.json').read_text())
    checked={r['name']:r for r in preparation['checked']}
    if plan['evaluator']!=preparation['plan']['evaluator']:raise ValueError('prepared evaluator changed')
    output_errors=(ValueError,KeyError,TypeError,*(m.ParticipantVisibleError for m in metric_modules().values()))
    actual={p.resolve() for p in (args.out/'cohort').rglob('trajectory.parquet')}
    expected=set();mapping={};summary=[];vetoes=[]
    for c in plan['candidates']:
        values={step:[] for step in c['steps']};members=[];unrecorded=[]
        for repeat in range(1,plan['repeats']+1):
            member=args.out/'cohort'/f"{c['name']}-r{repeat:02d}";base=member/c['arm']/c['input']
            for step in c['steps']:
                result=base/'evaluation'/c['sequence']/str(step)
                trajectory=result/'trajectory.parquet'
                checkpoint=base/'train'/c['sequence']/f'policy_{step}.safetensors'
                checkpoint_hash=digest(checkpoint)
                if checkpoint_hash!=checked[c['name']]['checkpoint_sha256'][str(step)]:raise ValueError('prepared checkpoint changed')
                job=read_inputs(Path(plan['manifest']),[c['sequence']],[c['input']])[f"{c['input']}/{c['sequence']}"]
                identity={'checkpoint_sha256':checkpoint_hash,'evaluator':plan['evaluator'],'input':c['input'],
                          'training_reference':job['training'],'scoring_reference':job['scoring'],'support':job['support']}
                if trajectory.exists():regular_member(args.out,trajectory);expected.add(trajectory.resolve())
                if failed_member(member,result,c,step):
                    if plan['tier']==1:raise ValueError(f'{result}: evaluation failed; Tier 1 has no failure class')
                    if not checked_attempt(args.out,result,identity):unrecorded.append((len(members),result))
                    members.append({'repeat':repeat,'failed':True,'catastrophe':True,'orientation':False})
                    values[step].append(None)
                    continue
                if not (result/'evaluation.done.json').exists():raise ValueError(f'{result}: evaluation never completed; cohort incomplete')
                regular_member(args.out,trajectory);expected.add(trajectory.resolve())
                done=json.loads((result/'evaluation.done.json').read_text())
                if done!={**identity,'trajectory_sha256':digest(trajectory),'metrics_sha256':digest(result/'metrics.json')}:
                    raise ValueError('checkpoint/runtime/input provenance mismatch')
                try:metrics=json.loads((result/'metrics.json').read_text())
                except ValueError:metrics=None
                if isinstance(metrics,dict) and (metrics.get('checkpoint_sha256')!=done['checkpoint_sha256'] or metrics.get('world_count')!=1
                    or metrics.get('reset_mode')!='explicit' or metrics.get('environment_steps')!=step):raise ValueError('invalid evaluator protocol')
                mesh=Path(checked_record(c['scoring_mesh'])['path']) if plan['tier']==1 else None
                if plan['tier']!=1:reference,_,_,body=load_reference(Path(job['training']['path']))
                # Custody and reference validation stay outside the output-failure boundary.
                try:
                    if not isinstance(metrics,dict) or metrics.get('non_finite_worlds')!=[]:raise ValueError('invalid evaluator output')
                    json.dumps(metrics,allow_nan=False)
                    report=score_rollout(trajectory,Path(job['scoring']['path']),mesh,
                                         episode_index=c['episode'] if plan['tier']==1 else None)
                    if not all(math.isfinite(v) for v in report['metrics'].values()):raise ValueError('non-finite score')
                    mean=report['metrics']
                    if plan['tier']!=1:achieved=load_rollout(trajectory,len(reference),body)
                except output_errors as error:
                    if plan['tier']==1:raise ValueError(f'{result}: evaluation failed; Tier 1 has no failure class') from error
                    if not checked_attempt(args.out,result,identity):unrecorded.append((len(members),result))
                    members.append({'repeat':repeat,'failed':True,'catastrophe':True,'orientation':False})
                    values[step].append(None)
                    continue
                if plan['tier']!=1:
                    limit=10 if plan['tier']==3 else 15  # Tier 3 protocol: self-MPPE <=10 cm (T3-POLICY PROTOCOL-qualified)
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
        # Cohorts from before attempt records cannot show which runtime failed; such a failure may stand only when a
        # sealed member already makes the candidate a catastrophe, so it changes neither its class nor the reseed order.
        if unrecorded and not any(m['catastrophe'] for i,m in enumerate(members) if i not in dict(unrecorded)):
            raise ValueError(f'{unrecorded[0][1]}: evaluation attempt missing; cohort incomplete')
        means={step:None if None in v else sum(m['AUC'] for m in v)/len(v) for step,v in values.items()}
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
    if (args.out/'export-binding.json').exists():raise ValueError('selection already frozen')
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
    if {c['name'] for c in plan['candidates']}!=set(PAIR):
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
                regular_member(root,trajectory)
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
    mapping={};parents=[];common=None;status={};combined=None;combined_sequences={};candidates=[];manifests={}
    def input_identity(plan):
        manifest=json.loads(Path(plan['manifest']).read_text())
        coverage=manifest.get('tier3_coverage') if plan['tier']==3 else None
        return coverage if coverage is not None else plan['manifest_sha256']
    for cohort in args.cohort:
        plan=checked_plan(cohort)
        manifests.update(plan.get('input_manifests',{}));manifests[plan['manifest']]=plan['manifest_sha256']
        key=(plan['tier'],input_identity(plan),plan['evaluator'],plan['repeats'])
        if common is not None and common!=key:raise ValueError('release cohort protocol/input/runtime differs')
        common=key
        selected=json.loads((cohort/'selected.json').read_text());binding=json.loads((cohort/'export-binding.json').read_text())
        if digest(cohort/'selected.json')!=binding['selected_sha256']:raise ValueError('selection changed')
        if any(digest(p)!=h for p,h in binding['checkpoints'].items()):raise ValueError('selected policy changed')
        if set(mapping)&set(selected):raise ValueError('duplicate release episodes')
        if plan['tier']!=1:status.update(bound_vetoes(cohort,selected,binding))
        if plan['tier']==3 and isinstance(key[1],dict):
            content=json.loads(Path(plan['manifest']).read_text())
            combined=content
            for entry in content['sequences']:
                if entry['sequence'] in combined_sequences:raise ValueError('duplicate coverage input sequence')
                combined_sequences[entry['sequence']]=entry
            candidates.extend(plan['candidates'])
        mapping.update(selected);parents.append({'cohort':str(cohort.resolve()),'selection_sha256':binding['selected_sha256']})
    if set(mapping)!=set(map(str,range(total))):raise ValueError('release needs every episode exactly once')
    rank={'pass':0,'orientation':1,'catastrophe':2}
    outcome={episode:{'original':v['status'],'final':v['status']} for episode,v in status.items()}
    evaluated={episode:{v['checkpoint_sha256']} for episode,v in status.items()}
    for cohort in args.rescue or []:  # earliest seed first; a later one replaces only a strictly worse class
        rescue=checked_plan(cohort)
        manifests.update(rescue.get('input_manifests',{}));manifests[rescue['manifest']]=rescue['manifest_sha256']
        if (rescue['tier'],input_identity(rescue),rescue['evaluator'],rescue['repeats'])!=common or common[0]==1:
            raise ValueError('rescue cohort protocol/input/runtime differs, or Tier 1')
        if combined is not None and any(entry['sequence'] not in combined_sequences or
                sequence_inputs(entry)!=sequence_inputs(combined_sequences[entry['sequence']])
                for entry in json.loads(Path(rescue['manifest']).read_text())['sequences']):
            raise ValueError('rescue cohort episode inputs differ')
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
    if combined is not None:
        target=args.out/'manifest.json'
        write(target,{**combined,'sequences':[combined_sequences[s] for s in sorted(combined_sequences)]})
        plan={**plan,'manifest':str(target.resolve()),'manifest_sha256':digest(target),'candidates':candidates}
    write(args.out/'plan.json',{**plan,'source_root':str(ROOT/'runs'),'parents':parents,'input_manifests':manifests})


def checked_work(root,plan,work):
    if plan['tier']==3:
        from assemble_tier3_work import verify
        verify(Path(plan['manifest']),ROOT/'data/challenge/track_2/tier_2_synthetic_noise',work)
        return
    expected={}
    for parent in plan.get('parents',[{'cohort':str(root)}]):
        member=checked_plan(Path(parent['cohort']))
        frozen=json.loads((Path(member['source_root'])/'frozen-inputs/manifest.json').read_text())
        if frozen['input_manifest_sha256']!=member['manifest_sha256']:
            raise ValueError('frozen-inputs archive belongs to a different input manifest')
        for entry in json.loads(Path(member['manifest']).read_text())['sequences']:
            h=frozen['files'][f"data/work/tier{plan['tier']}/sequences/{entry['sequence']}/poses.npy"]
            if expected.setdefault(entry['sequence'],h)!=h:raise ValueError('cohorts disagree on converter poses')
    if not expected:raise ValueError('missing frozen converter poses')
    for sequence,h in expected.items():
        if digest(work/'sequences'/sequence/'poses.npy')!=h:
            raise ValueError('export work differs from frozen converter poses: '+sequence)


def export(args):
    from pack_no_clean import rs
    if not re.fullmatch(rs.CODE_COMMIT_PATTERN,args.commit):
        raise ValueError('a pinned GitHub commit URL is required before export')
    plan=checked_plan(args.out)
    if plan['ep18']:raise ValueError('ep18 pilot has no submission export')
    for path in {plan['manifest'],*plan.get('input_manifests',{})}:
        for entry in json.loads(Path(path).read_text())['sequences']:
            read_inputs(Path(path),[entry['sequence']],list(entry['arms']))
            for asset in entry.get('assets',[]):checked_record(asset)
    binding=json.loads((args.out/'export-binding.json').read_text())
    if evaluator_identity(Path(os.environ.get('FLASH_ROOT',ROOT/'external/video_to_data/robotic_grounding/flash_chord')))!=plan['evaluator']:
        raise ValueError('evaluator/runtime changed after cohort')
    if digest(args.out/'selected.json')!=binding['selected_sha256'] or any(digest(p)!=h for p,h in binding['checkpoints'].items()):
        raise ValueError('frozen selection/checkpoint changed')
    checked_work(args.out,plan,args.work)
    # The export host has no .git; launch.sh records the clean commit it uploaded.
    commit=args.commit.rstrip('/').rsplit('/',1)[-1].lower()
    try:launched=Path(os.environ.get('LAUNCH_COMMIT',ROOT/'LAUNCH_COMMIT')).read_text().split()
    except OSError as e:raise ValueError('missing clean launch commit record') from e
    if not re.fullmatch('[0-9a-f]{40}',commit) or launched!=[commit,'changed_files=0']:
        raise ValueError('code_commit_url must name the full clean launch commit uploaded to this host')
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
