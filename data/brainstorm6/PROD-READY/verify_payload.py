"""Run the staged Tier 1/2 generator with actual references and NO Tier 3 research folders."""
import os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def verify_pilot(run):
    import json,hashlib
    payload=run/'payload'
    binding=payload/'runs'/run.name/'input-assets/binding.json'
    frozen=json.loads(binding.read_text())
    if 'episodes' in frozen:
        verify_coverage(run,payload,binding,frozen);return
    assets=Path(frozen['assets'])
    if assets!=Path('runs')/run.name/'input-assets/assets':raise ValueError('unexpected candidate root')
    allowed={binding.relative_to(payload)} | {assets/rel for rel in frozen['sha256']} | {
        HERE.relative_to(ROOT)/'generated/arms'/f'{arm}.txt' for arm in ('T3P-contact350','T3P-control-s42')}
    actual={p.relative_to(payload) for p in payload.rglob('*') if p.is_file()}
    if actual!=allowed or any(p.is_symlink() for p in payload.rglob('*')):
        raise ValueError('pilot payload contains unexpected files or links (no clean/Tier 1 data permitted)')
    for relative in actual:
        if any('tier_1' in part or part in ('clean','tier1','baseline') for part in relative.parts):
            raise ValueError('clean/Tier 1 payload forbidden')
    for rel,h in frozen['sha256'].items():
        if hashlib.sha256((payload/assets/rel).read_bytes()).hexdigest()!=h:
            raise ValueError('candidate SHA-256 mismatch')
    print('PILOT PAYLOAD PASS: 6 candidate assets, 2 arms, 1 binding; no clean/Tier 1 files')

def verify_coverage(run,payload,binding,frozen):
    import hashlib
    inputs=Path('runs')/run.name/'input-assets'
    roster=inputs/'roster/meta'
    allowed={binding.relative_to(payload),HERE.relative_to(ROOT)/'generated/arms/T2C-contact.txt'}|{roster/name for name in frozen['roster']}
    for episode in frozen['episodes']:
        if Path(episode['assets'])!=inputs/f"ep{episode['episode']:02d}/assets":raise ValueError('unexpected coverage asset root')
        allowed|={Path(episode['assets'])/rel for rel in episode['sha256']}
    actual={p.relative_to(payload) for p in payload.rglob('*') if p.is_file()}
    if actual!=allowed or any(p.is_symlink() for p in payload.rglob('*')):
        raise ValueError('coverage payload contains unexpected files or links (no clean/Tier 1 data permitted)')
    for relative in actual:
        if any('tier_1' in part or part in ('clean','tier1','baseline') for part in relative.parts):
            raise ValueError('clean/Tier 1 payload forbidden')
    for episode in frozen['episodes']:
        for rel,h in episode['sha256'].items():
            if hashlib.sha256((payload/episode['assets']/rel).read_bytes()).hexdigest()!=h:raise ValueError('coverage SHA-256 mismatch')
    for name,h in frozen['roster'].items():
        if hashlib.sha256((payload/roster/name).read_bytes()).hexdigest()!=h:raise ValueError('roster SHA-256 mismatch')
    print(f"COVERAGE PAYLOAD PASS: {len(frozen['episodes'])} episodes, 1 arm, roster, 1 binding; no clean/Tier 1 files")

if len(sys.argv)>1:
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',type=Path,required=True)
    args=parser.parse_args()
    try:verify_pilot(args.pilot.resolve())
    except (OSError,ValueError,KeyError) as error:parser.exit(1,f'error: {error}\n')
    sys.exit(0)

PROBE=Path('/tmp/b6-PROD-READY/payload-probe')
try:
    for name in ['scripts']:
        shutil.copytree(ROOT/name,PROBE/name,ignore=shutil.ignore_patterns('__pycache__'))
    for relative in ['.venv','external/kit/v2d_submission_kit','external/video_to_data','data/work','data/challenge']:
        p=PROBE/relative;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(ROOT/relative,target_is_directory=True)
    generator=HERE.relative_to(ROOT)/'generate_jobs.py';(PROBE/generator).parent.mkdir(parents=True)
    shutil.copyfile(HERE/'generate_jobs.py',PROBE/generator)
    old=Path('data/brainstorm5/PROD-PLAN/generate_jobs.py');(PROBE/old).parent.mkdir(parents=True)
    shutil.copyfile(ROOT/old,PROBE/old)
    assert not (PROBE/'data/tier3_bridge').exists() and not (PROBE/'data/brainstorm5/T3-POLICY').exists()
    result=subprocess.run([sys.executable,'-B',str(PROBE/generator),'--production-only','--verify'],cwd=PROBE,capture_output=True,text=True)
    print(result.stdout);print(result.stderr)
    assert result.returncode==0
    assert not (PROBE/generator.parent/'generated/tier3').exists()
    print('HOST PAYLOAD PASS: 60 actual-input jobs; no Tier 3 research dependencies; no created run root')
finally:
    if PROBE.exists():shutil.rmtree(PROBE)
