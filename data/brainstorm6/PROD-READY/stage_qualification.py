#!/usr/bin/env python3
"""Stage the first ep01/08/16 Tier 1 cohort from t1h1's pulled artifacts, offline.

Run data uses launch.sh's existing pulled/<name> restore seam. The only payload
overlay is the unchanged, committed arm file. No training jobs are submitted.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARM = 'F6-rebalance-tail'
ARM_PATH = Path('data/brainstorm6/PROD-READY/generated/arms/F6-rebalance-tail.txt')
MANIFEST = Path('data/brainstorm6/PROD-READY/generated/tier1/manifest.json')
EPISODES = (1, 8, 16)
STEPS = (330012672, 349999104)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content, indent=2)+'\n')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == 'darwin':
        subprocess.run(['cp', '-c', str(source), str(target)], check=True)
    else:
        shutil.copyfile(source, target)


def rebuild(template, template_root, box_root, frozen):
    manifest = json.loads(template.read_text())
    def rebase(node):
        if isinstance(node, dict):
            if 'path' in node and 'sha256' in node:
                rel = Path(node['path']).relative_to(template_root).as_posix()
                node.update(path=str(box_root/rel), sha256=frozen['files'][rel])
            for value in node.values(): rebase(value)
        elif isinstance(node, list):
            for value in node: rebase(value)
    rebase(manifest)
    data = json.dumps(manifest, indent=2)+'\n'
    if hashlib.sha256(data.encode()).hexdigest() != frozen['input_manifest_sha256']:
        raise ValueError('rebuilt manifest differs from t1h1 frozen training manifest')
    return manifest, data


def stage(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.name): raise ValueError('unsafe name')
    source = args.source.resolve()
    if (source/'qualification').exists(): raise ValueError('source already has a qualification attempt')
    frozen = json.loads((source/'frozen-inputs/manifest.json').read_text())
    manifest, data = rebuild(args.template, args.template_root.resolve(), args.box_root, frozen)
    if manifest['tier'] != 1 or manifest['scoring_mode'] != 'self': raise ValueError('Tier 1 self-scored inputs required')
    records = {entry['sequence']: entry for entry in manifest['sequences']}
    roster = [json.loads(line) for line in (args.tier_root/'meta/episodes_metadata.jsonl').read_text().splitlines()]
    episodes = {row['sequence_id']: row for row in roster}
    arm = (ROOT/ARM_PATH).read_text()
    overrides = arm.split()[1:]
    pending = {'frozen-inputs/manifest.json': source/'frozen-inputs/manifest.json'}
    for rel, digest in frozen['files'].items():
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'data': raise ValueError('unsafe frozen input path')
        pending['frozen-inputs/files/'+rel] = source/'frozen-inputs/files'/rel
        if sha(pending['frozen-inputs/files/'+rel]) != digest: raise ValueError('corrupt frozen input: '+rel)
    lines = []
    policies = []
    seen = []
    for line in args.jobs.read_text().splitlines():
        if not line or line.startswith('#'): continue
        name, arms_file, selected_arm, label, sequence, steps = line.split('|')
        episode = episodes[sequence]['episode_index']
        if (name != f't1-ep{episode:02d}' or episode not in EPISODES or selected_arm != ARM or label != 'clean'
                or tuple(map(int,steps.split())) != STEPS): raise ValueError('only frozen ep01/08/16 F6/clean candidates permitted')
        if (ROOT/'scripts'/arms_file).read_text().split() != arm.split(): raise ValueError('source arm changed')
        seen.append(episode)
        recipe_rel = f'{name}/{ARM}/recipe.json'
        recipe = json.loads((source/recipe_rel).read_text())
        expected = records[sequence]['arms']['clean']
        reference = recipe['references']['clean/'+sequence]
        if (recipe['schema_version'] != 2 or recipe['overrides'] != overrides or
                reference != {'input':'clean','sequence':sequence,'training':expected['training_reference'],
                              'scoring':expected['clean_scoring_reference'],'support':expected['support']}):
            raise ValueError('source recipe differs from frozen inputs/arm')
        pending[recipe_rel] = source/recipe_rel
        for step in STEPS:
            base = f'{name}/{ARM}/clean'
            policy = f'{base}/train/{sequence}/policy_{step}.safetensors'
            result = f'{base}/evaluation/{sequence}/{step}'
            receipt = json.loads((source/result/'evaluation.done.json').read_text())
            digest = sha(source/policy)
            if not (source/policy).stat().st_size or receipt['checkpoint_sha256'] != digest:
                raise ValueError('policy differs from training evaluation receipt')
            for key, expected_key in [('training_reference','training_reference'),('scoring_reference','clean_scoring_reference'),('support','support')]:
                if receipt[key] != expected[expected_key]: raise ValueError('receipt input differs from frozen manifest')
            pending[policy] = source/policy
            policies.append({'episode':episode,'step':step,'sha256':digest,'path':policy})
            for filename, key in [('trajectory.parquet','trajectory_sha256'),('metrics.json','metrics_sha256')]:
                rel = result+'/'+filename
                if sha(source/rel) != receipt[key]: raise ValueError('unsealed source evaluation output')
                pending[rel] = source/rel
            pending[result+'/evaluation.done.json'] = source/result/'evaluation.done.json'
        lines.append('|'.join((name, '../'+ARM_PATH.as_posix(), ARM, 'clean', sequence, ' '.join(map(str,STEPS)))))
        mesh = Path(episodes[sequence]['mesh'])
        if mesh.is_absolute() or '..' in mesh.parts: raise ValueError('unsafe scoring mesh')
        pending['released/'+mesh.as_posix()] = args.tier_root/mesh
    if sorted(seen) != list(EPISODES): raise ValueError('exactly ep01, ep08, ep16 required')
    pending['released/meta/episodes_metadata.jsonl'] = args.tier_root/'meta/episodes_metadata.jsonl'
    out = HERE/'runs'/args.name
    free_before = shutil.disk_usage(HERE).free
    out.mkdir(parents=True, exist_ok=False)
    try:
        seed = out/'pulled'/args.name
        for rel, path in pending.items():
            if path.is_symlink(): raise ValueError('regular source artifacts required')
            copy(path, seed/rel)
        write(seed/'input-manifest.json', data)
        write(seed/'qualification.jobs', '\n'.join(lines)+'\n')
        copy(ROOT/ARM_PATH, out/'payload'/ARM_PATH)
        write(out/'jobs.txt', '# Empty outer queue: post.sh owns the first qualification attempt.\n')
        write(out/'env.sh', 'export BOOTSTRAP_TIERS=" "\nexport BOOTSTRAP_CHALLENGE=0\nexport BUILD_SCREEN=0\nexport PILOT_EVALUATE_ONLY=1\n')
        run = f'runs/{args.name}'
        write(out/'pre.sh', 'set -euo pipefail\n'+
            f'.venv/bin/python -B - <<\'VERIFY\'\nimport hashlib,json\nfrom pathlib import Path\nr=Path({run!r})\n'+
            'if (r/"qualification").exists(): raise SystemExit("qualification already attempted; retain it, no rerolls")\n'+
            'index=json.loads((r/"custody.json").read_text())\n'+
            'for rel,h in index["files"].items():\n p=r/rel\n'+
            ' if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=h: raise SystemExit("staged custody mismatch: "+rel)\nVERIFY\n'+
            f'.venv/bin/python -B scripts/freeze_inputs.py restore --archive {run}/frozen-inputs\n'+
            f'mkdir -p {MANIFEST.parent}\ncp {run}/input-manifest.json {MANIFEST}\n'+
            f'export MANIFEST="$PWD/{MANIFEST}"\n')
        tool = '.venv/bin/python -B scripts/production_repeats.py'
        write(out/'post.sh', 'set -euo pipefail\n'+
            f'{tool} prepare --jobs {run}/qualification.jobs --manifest "$MANIFEST" --source {run} --tier-root {run}/released --tier 1 --k 3 --out {run}/qualification\n'+
            f'{tool} run --gpus "$(nvidia-smi -L | wc -l)" --out {run}/qualification\n'+
            f'{tool} qualify --out {run}/qualification\n')
        index = {'source':str(source),'episodes':list(EPISODES),'k':3,'manifest_sha256':frozen['input_manifest_sha256'],
                 'policies':policies,'files':{p.relative_to(seed).as_posix():sha(p) for p in sorted(seed.rglob('*')) if p.is_file()},
                 'originals':{str(path):sha(path) for path in set(pending.values())},
                 'controls':{p.relative_to(out).as_posix():sha(p) for p in [out/'jobs.txt',out/'env.sh',out/'pre.sh',out/'post.sh',out/'payload'/ARM_PATH]}}
        write(seed/'custody.json', index)
        verify(out)
        write(out/'staging-disk.json', {'copy_method':'APFS clone' if sys.platform=='darwin' else 'copy',
              'logical_file_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
              'free_bytes_before':free_before,'free_bytes_after':shutil.disk_usage(HERE).free,
              'limit':'free-space delta includes concurrent filesystem activity; clones share source extents'})
    except BaseException:
        shutil.rmtree(out)
        raise
    print(out)


def verify(out):
    seed = out/'pulled'/out.name
    index = json.loads((seed/'custody.json').read_text())
    if (seed/'qualification').exists(): raise ValueError('qualification already attempted; do not restage')
    for rel, digest in index['files'].items():
        if (seed/rel).is_symlink() or sha(seed/rel) != digest: raise ValueError('staged custody mismatch: '+rel)
    for rel, digest in index['controls'].items():
        if sha(out/rel) != digest: raise ValueError('controller changed: '+rel)
    frozen = json.loads((seed/'frozen-inputs/manifest.json').read_text())
    if sha(seed/'input-manifest.json') != frozen['input_manifest_sha256']: raise ValueError('manifest binding mismatch')
    for original, digest in index['originals'].items():
        if sha(original) != digest: raise ValueError('original source changed: '+original)
    print(f"CUSTODY PASS: {len(index['policies'])} policies, {len(frozen['files'])} frozen inputs, {len(index['files'])} staged files; manifest {index['manifest_sha256']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    stage_parser = sub.add_parser('stage')
    source = HERE/'runs/t1h1'
    stage_parser.add_argument('--source', type=Path, default=source/'pulled/t1h1')
    stage_parser.add_argument('--jobs', type=Path, required=True)
    stage_parser.add_argument('--template', type=Path, default=ROOT/MANIFEST)
    stage_parser.add_argument('--template-root', type=Path, default=ROOT)
    stage_parser.add_argument('--box-root', type=Path, default=Path('/workspace/v2d-track-2'))
    stage_parser.add_argument('--tier-root', type=Path, default=ROOT/'data/challenge/track_2/tier_1_multiview_caption')
    stage_parser.add_argument('--name', default='t1q1')
    sub.add_parser('verify').add_argument('run', type=Path)
    args = parser.parse_args()
    try:
        stage(args) if args.action == 'stage' else verify(args.run.resolve())
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__': main()
