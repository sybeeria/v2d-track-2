#!/usr/bin/env python3
"""Stage explicit launch payloads and existing hooks, without calling a provider."""
import argparse,hashlib,json,os,shutil,sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--tier',type=int,choices=[1,2,3],required=True);p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--frozen-from',type=Path);p.add_argument('--coverage',type=Path,help='Tier 3 bridge coverage_manifest.json (decision D2); omit for the ep18 pilot');p.add_argument('--freeze-other-tier',action='store_true');p.add_argument('--k',type=int,default=3);p.add_argument('--name',required=True);args=p.parse_args()
    if args.k<1:raise ValueError('k must be positive')
    if not args.name.replace('-','').replace('_','').isalnum():raise ValueError('unsafe run name')
    if args.tier==3:
        (stage_tier3_coverage if args.coverage else stage_ep18)(args);return
    if args.frozen_from:
        # capture writes manifest.json last; a pull taken mid-capture must fail here, not after a paid bootstrap.
        try:files=json.loads((args.frozen_from/'manifest.json').read_text())['files']
        except (OSError,ValueError,KeyError):raise ValueError(f'frozen input archive {args.frozen_from} has no complete manifest')
        if any(not (args.frozen_from/'files'/rel).is_file() or hashlib.sha256((args.frozen_from/'files'/rel).read_bytes()).hexdigest()!=h for rel,h in files.items()):
            raise ValueError(f'frozen input archive {args.frozen_from} is incomplete or corrupt')
    out=HERE/'runs'/args.name
    if out.exists():raise ValueError('fresh controller directory required')
    out.mkdir(parents=True)
    shutil.copyfile(args.jobs,out/'jobs.txt')
    payload=out/'payload';relative=HERE.relative_to(ROOT);dest=payload/relative;dest.mkdir(parents=True)
    shutil.copyfile(HERE/'generate_jobs.py',dest/'generate_jobs.py')
    shutil.copytree(HERE/'generated/arms',dest/'generated/arms')
    old=Path('data/brainstorm5/PROD-PLAN/generate_jobs.py');(payload/old).parent.mkdir(parents=True)
    shutil.copyfile(ROOT/old,payload/old)
    if args.frozen_from:shutil.copytree(args.frozen_from,dest/'seed-inputs'/args.name)
    # box pre hooks run after host-local bootstrap. Its own references and hashes are regenerated.
    (out/'env.sh').write_text('export BOOTSTRAP_TIERS="tier1 tier2"\nexport BUILD_SCREEN=0\n')
    other=f'runs/{args.name}/frozen-tier{3-args.tier}'
    (out/'pre.sh').write_text('set -euo pipefail\n'+
        f'archive="runs/{args.name}/frozen-inputs"\n'+
        'if [ -f "$archive/manifest.json" ]; then .venv/bin/python -B scripts/freeze_inputs.py restore --archive "$archive"; '+
        (f'else .venv/bin/python -B scripts/freeze_inputs.py restore --archive {relative}/seed-inputs/{args.name}; ' if args.frozen_from else '')+'fi\n'+
        # A replacement host regenerates references during bootstrap; reuse the captured other-tier closure too.
        (f'if [ -f {other}/manifest.json ]; then .venv/bin/python -B scripts/freeze_inputs.py restore --archive {other}; fi\n' if args.freeze_other_tier else '')+
        f'.venv/bin/python -B {relative}/generate_jobs.py --production-only --verify\n'+
        f'.venv/bin/python -B scripts/freeze_inputs.py capture --manifest {relative}/generated/tier{args.tier}/manifest.json --work data/work/tier{args.tier} --archive "$archive"\n'+
        f'export MANIFEST="$PWD/{relative}/generated/tier{args.tier}/manifest.json"\n'+
        '.venv/bin/python - <<\'DISK\'\nimport shutil\nfree=shutil.disk_usage(".").free\nassert free >= 115_000_000_000, f"production disk admission failed: {free} bytes free"\nDISK\n')
    if args.freeze_other_tier:
        with (out/'pre.sh').open('a') as f:
            f.write(f'.venv/bin/python -B scripts/freeze_inputs.py capture --manifest {relative}/generated/tier{3-args.tier}/manifest.json --work data/work/tier{3-args.tier} --archive {other}\n')
    released='tier_1_multiview_caption' if args.tier==1 else 'tier_2_synthetic_noise'
    (out/'post.sh').write_text('set -euo pipefail\n'+
        f'.venv/bin/python -B scripts/production_repeats.py prepare --jobs /workspace/{args.name}.jobs --manifest "$MANIFEST" --source runs/{args.name} --tier-root data/challenge/track_2/{released} --tier {args.tier} --k {args.k} --out runs/{args.name}/qualification\n'+
        f'.venv/bin/python -B scripts/production_repeats.py run --gpus "$(nvidia-smi -L | wc -l)" --out runs/{args.name}/qualification\n'+
        f'.venv/bin/python -B scripts/production_repeats.py qualify --out runs/{args.name}/qualification\n')
    print(out)

def stage_ep18(args):
    sys.path.insert(0,str(ROOT/'scripts'))
    from ep18_pilot import ASSETS, READY, PAIR, digest, validate_jobs
    validate_jobs(args.jobs)
    if args.k!=3 or args.frozen_from or args.freeze_other_tier:
        raise ValueError('ep18 requires k=3 and fresh candidate_v1 inputs')
    out=HERE/'runs'/args.name
    if out.exists():raise ValueError('fresh controller directory required')
    payload=out/'payload';assets=Path('runs')/args.name/'input-assets/assets'
    # Copy only the six candidate assets, retaining their /assets/ path boundary.
    files=[p for p in (ROOT/ASSETS).rglob('*') if p.is_file()]
    if len(files)!=6 or any(p.is_symlink() for p in (ROOT/ASSETS).rglob('*')):
        raise ValueError('candidate_v1 must contain exactly six regular assets')
    hashes={p.relative_to(ROOT/ASSETS).as_posix():digest(p) for p in files}
    shutil.copytree(ROOT/ASSETS,payload/assets)
    for rel,h in hashes.items():
        if digest(payload/assets/rel)!=h:raise ValueError('candidate changed during staging')
    for arm,_ in PAIR.values():
        relative=READY/'generated/arms'/f'{arm}.txt'
        target=payload/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    binding=assets.parent/'binding.json'
    (payload/binding).write_text(json.dumps({'assets':str(assets),'sha256':hashes},indent=2)+'\n')
    shutil.copyfile(args.jobs,out/'jobs.txt')
    (out/'env.sh').write_text('export BOOTSTRAP_TIERS=" "\nexport BOOTSTRAP_CHALLENGE=0\nexport BUILD_SCREEN=0\n')
    manifest=f'runs/{args.name}/manifest.json'
    (out/'pre.sh').write_text('set -euo pipefail\n'+
        f'if [ -e runs/{args.name}/decision.json ]; then echo "pilot already decided; retain attempts" >&2; exit 1; fi\n'+
        f'.venv/bin/python -B scripts/ep18_pilot.py admit --binding {binding} --output {manifest}\n'+
        f'export MANIFEST="$PWD/{manifest}"\n')
    qualification=f'runs/{args.name}/qualification'
    (out/'post.sh').write_text('set -euo pipefail\n'+
        f'.venv/bin/python -B scripts/production_repeats.py prepare --ep18 --jobs /workspace/{args.name}.jobs --manifest "$MANIFEST" --source runs/{args.name} --tier-root runs/{args.name}/input-assets --tier 3 --k 3 --out {qualification} || true\n'+
        f'.venv/bin/python -B scripts/production_repeats.py run --gpus "$(nvidia-smi -L | wc -l)" --out {qualification} || true\n'+
        f'.venv/bin/python -B scripts/production_repeats.py qualify --out {qualification} || true\n'+
        f'.venv/bin/python -B scripts/ep18_pilot.py decide --qualification {qualification}/qualification.json --output runs/{args.name}/decision.json\n')
    # Do not silently remove the old HOLD: bind its replacement to this concrete run.
    (HERE/'generated/tier3/GATE.json').write_text(json.dumps({'state':'PILOT_STAGED','decision':'D1',
        'run':str(out.relative_to(ROOT)),'jobs':2,'k':3,'candidate_sha256':hashes,
        'scope':'ep18 candidate_v1 only; no broad Tier 3 policy spend'},indent=2)+'\n')
    print(out)

def stage_tier3_coverage(args):
    sys.path.insert(0,str(ROOT/'scripts'))
    import tier3_coverage
    if args.frozen_from or args.freeze_other_tier:raise ValueError('Tier 3 coverage inputs come only from the bridge coverage manifest')
    out=HERE/'runs'/args.name
    if out.exists():raise ValueError('fresh controller directory required')
    payload=out/'payload'
    binding=tier3_coverage.stage(args.coverage,args.jobs,payload,Path('runs')/args.name)
    arms=HERE.relative_to(ROOT)/'generated/arms/T2C-contact.txt'
    (payload/arms).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/arms,payload/arms)
    shutil.copyfile(args.jobs,out/'jobs.txt')
    (out/'env.sh').write_text('export BOOTSTRAP_TIERS=" "\nexport BOOTSTRAP_CHALLENGE=0\nexport BUILD_SCREEN=0\n')
    manifest=f'runs/{args.name}/manifest.json'
    (out/'pre.sh').write_text('set -euo pipefail\n'+
        f'.venv/bin/python -B scripts/tier3_coverage.py admit --binding {binding} --output {manifest}\n'+
        f'export MANIFEST="$PWD/{manifest}"\n')
    qualification=f'runs/{args.name}/qualification'
    (out/'post.sh').write_text('set -euo pipefail\n'+
        f'.venv/bin/python -B scripts/production_repeats.py prepare --jobs /workspace/{args.name}.jobs --manifest "$MANIFEST" --source runs/{args.name} --tier-root {binding.parent}/roster --tier 3 --k {args.k} --out {qualification}\n'+
        f'.venv/bin/python -B scripts/production_repeats.py run --gpus "$(nvidia-smi -L | wc -l)" --out {qualification}\n'+
        f'.venv/bin/python -B scripts/production_repeats.py qualify --out {qualification}\n')
    print(out)

if __name__=='__main__':main()
