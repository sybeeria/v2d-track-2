#!/usr/bin/env python3
"""Reconcile B only, rehash actual inputs, and exercise every job through the real queue."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'external/video_to_data/robotic_grounding/flash_chord/src')]
from pilot_queue import read_arms, read_inputs

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value if isinstance(value,str) else json.dumps(value,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--production-only',action='store_true',help='Tier 1/2 host staging; no dependency on unavailable Tier 3 research files');args=p.parse_args()
    spec=importlib.util.spec_from_file_location('old_generator',ROOT/'data/brainstorm5/PROD-PLAN/generate_jobs.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    episodes=sorted(map(json.loads,(ROOT/'data/challenge/track_2/tier_2_synthetic_noise/meta/episodes_metadata.jsonl').read_text().splitlines()),key=lambda e:e['episode_index'])
    assert [e['episode_index'] for e in episodes]==list(range(30))
    frozen=old.load_arms();gen=HERE/'generated';counts={}
    for tier,label,arm,steps in [(1,'clean','F6-rebalance-tail',[330012672,349999104]),(2,'raw_repaired','T2C-contact',[49999872])]:
        armfile=gen/'arms'/f'{arm}.txt';write(armfile,' '.join([arm,*frozen[arm]])+'\n');read_arms(armfile,steps)
        manifest={'schema_version':1,'tier':tier,'scoring_mode':'self','sequences':[]}
        for e in episodes:
            seq=e['sequence_id'];ref=ROOT/f'data/work/tier{tier}/motion/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet'
            record,missing,assets=old.inspect_reference(ref,e['frames'])
            if missing:raise ValueError(missing)
            manifest['sequences'].append({'sequence':seq,'frames':e['frames'],'arms':{label:record},'assets':assets})
        out=gen/f'tier{tier}';write(out/'manifest.json',manifest)
        order=[19,11,29]+[i for i in range(30) if i not in (19,11,29)] if tier==1 else list(range(30))
        lines=[f"t{tier}-ep{i:02d}|{os.path.relpath(armfile,ROOT/'scripts')}|{arm}|{label}|{episodes[i]['sequence_id']}|{' '.join(map(str,steps))}" for i in order]
        write(out/'jobs.txt','\n'.join(lines)+'\n')
        if tier==1:
            write(out/'sentinels.txt','\n'.join(lines[:3])+'\n');write(out/'remaining.txt','\n'.join(lines[3:])+'\n')
            for host in (0,1):write(out/f'remaining-host{host}.txt','\n'.join(lines[3:][host::2])+'\n')
        for host in (0,1):write(out/f'host{host}.txt','\n'.join(lines[host::2])+'\n')
        for line in lines:
            name,af,a,l,s,st=line.split('|');read_arms(ROOT/'scripts'/af,list(map(int,st.split())));read_inputs(out/'manifest.json',[s],[l])
        if args.verify:
            r=subprocess.run(['bash',str(ROOT/'scripts/test_day.sh'),'2',str(out/'DRY-ONLY'),'--dry-run'],env={**os.environ,'JOBS_FILE':str(out/'jobs.txt'),'MANIFEST':str(out/'manifest.json')},capture_output=True,text=True)
            write(out/'dry-run.log',r.stdout+r.stderr)
            assert r.returncode==0,r.stderr
            assert [r.stdout.count(x) for x in ('/scripts/train_queue.sh ','/evaluate_policy.py ','/scripts/score_rollout.py ')]==[30,30*len(steps),30*len(steps)]
            assert not (out/'DRY-ONLY').exists()
        counts[f'tier{tier}']={'jobs':30,'input':label,'steps':steps,'order':order,'hashed_reference_and_support':30}
    if args.production_only:
        write(gen/'verification.json',counts);print(json.dumps(counts,indent=2));return
    t3=gen/'tier3';lines=[]
    for line in (ROOT/'data/brainstorm5/T3-POLICY/jobs-qualified.txt').read_text().splitlines():
        name,af,arm,label,seq,steps=line.split('|')
        selected=next(row for row in (ROOT/'scripts'/af).read_text().splitlines() if row.split() and row.split()[0]==arm)
        out=gen/'arms'/f'{arm}.txt';write(out,selected+'\n');read_arms(out,list(map(int,steps.split())))
        lines.append('|'.join([name,os.path.relpath(out,ROOT/'scripts'),arm,label,seq,steps]))
    write(t3/'jobs.GATED.txt','\n'.join(lines)+'\n')
    # Exercise both placeholder lines with the historical HOLD input, solely as a parser contract.
    supplied=json.loads((ROOT/'data/tier3_bridge/manifest.json').read_text())
    sequence=lines[0].split('|')[4]
    entry=next(e for e in supplied['sequences'] if e['sequence']==sequence)
    placeholder={'tier':3,'scoring_mode':'self','sequences':[{'sequence':sequence,'arms':{'tier3_support_static':entry['arms']['tier3_repaired']}}]}
    write(t3/'CONTRACT_ONLY_HOLD.json',placeholder)
    for index,line in enumerate(lines):
        _,af,_,label,seq,steps=line.split('|')
        read_arms(ROOT/'scripts'/af,list(map(int,steps.split())))
        read_inputs(t3/'CONTRACT_ONLY_HOLD.json',[seq],[label])
        if args.verify:
            one=t3/f'line{index}.txt';write(one,line+'\n')
            r=subprocess.run(['bash',str(ROOT/'scripts/test_day.sh'),'1',str(t3/'DRY-ONLY'),'--dry-run'],env={**os.environ,'JOBS_FILE':str(one),'MANIFEST':str(t3/'CONTRACT_ONLY_HOLD.json')},capture_output=True,text=True)
            write(t3/f'contract{index}.log',r.stdout+r.stderr)
            assert r.returncode==0,r.stderr
            assert not (t3/'DRY-ONLY').exists()

    write(t3/'GATE.json',{'state':'HOLD','owner':'T3-EP18','required':'independently qualified candidate_manifest plus self-scored retargeted manifest; use T3-POLICY/run_qualified.sh; no existing repaired-reference substitution','jobs':2})
    counts['tier3']={'jobs':2,'state':'HOLD','budgets':[350,50],'contract':'arm arithmetic verified; qualified manifest unavailable, execution refused'}
    write(gen/'verification.json',counts);print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
