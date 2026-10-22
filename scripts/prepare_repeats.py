"""Stage immutable checkpoint-only evaluation jobs for the existing test_day runner.

No training or evaluator changes. Each repeat has a fresh cache root and retains its rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def jobs(plan, group):
    for candidate in plan['candidates']:
        if candidate['group'] != group:
            continue
        for repeat in range(1, plan['repeats'] + 1):
            name = f"{candidate['name']}-r{repeat:02d}"
            line = '|'.join((name, candidate['arms_file'], candidate['arm'], candidate['input'],
                             candidate['sequence'], ' '.join(map(str, candidate['steps']))))
            yield candidate, name, line


def prepare(plan, source_root, out_root):
    checked = []
    for candidate in plan['candidates']:
        source = source_root / candidate['source']
        recipe = source / 'recipe.json'
        if not recipe.is_file():
            raise ValueError(f'{recipe}: missing source recipe; no training is permitted')
        json.loads(recipe.read_text())
        train = source / candidate['input'] / 'train' / candidate['sequence']
        hashes = {}
        for step in sorted(set([candidate['final_step'], *candidate['steps']])):
            checkpoint = train / f'policy_{step}.safetensors'
            if not checkpoint.is_file() or not checkpoint.stat().st_size:
                raise ValueError(f'{checkpoint}: missing checkpoint; no training is permitted')
            hashes[str(step)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for step in candidate['steps']:
            stamp = source / candidate['input'] / 'evaluation' / candidate['sequence'] / str(step) / 'evaluation.done.json'
            if not stamp.is_file() or json.loads(stamp.read_text())['checkpoint_sha256'] != hashes[str(step)]:
                raise ValueError(f'{stamp}: missing or mismatched checkpoint provenance')
        checked.append(dict(name=candidate['name'], recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
                            checkpoint_sha256=hashes))
    # Check the whole cohort before creating any job that could accidentally retrain.
    out_root.mkdir(parents=True, exist_ok=True)
    for group in sorted({candidate['group'] for candidate in plan['candidates']}):
        lines = []
        for candidate, name, line in jobs(plan, group):
            source = source_root / candidate['source']
            arm = out_root / group / name / candidate['arm']
            recipe = arm / 'recipe.json'
            target = arm / candidate['input'] / 'train' / candidate['sequence']
            original = (source / candidate['input'] / 'train' / candidate['sequence']).resolve()
            if recipe.exists() and recipe.read_bytes() != (source / 'recipe.json').read_bytes():
                raise ValueError(f'{recipe}: existing recipe differs')
            if target.is_symlink():
                if target.resolve() != original:
                    raise ValueError(f'{target}: existing checkpoint link differs')
            elif target.exists():
                raise ValueError(f'{target}: existing non-link; choose a fresh out-root')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_symlink():
                target.symlink_to(original, target_is_directory=True)
            if not recipe.exists():
                shutil.copyfile(source / 'recipe.json', recipe)
            lines.append(line)
        (out_root / f'jobs-{group}.txt').write_text('\n'.join(lines) + '\n')
    (out_root / 'preparation.json').write_text(json.dumps(dict(plan=plan, checked=checked), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--out-root', type=Path)
    parser.add_argument('--emit-jobs-dir', type=Path, help='write exact job lists only; do not stage or evaluate')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if type(plan['repeats']) is not int or plan['repeats'] < 1:
        parser.error('repeats must be a positive integer')
    if args.emit_jobs_dir:
        args.emit_jobs_dir.mkdir(parents=True, exist_ok=True)
        for group in sorted({candidate['group'] for candidate in plan['candidates']}):
            (args.emit_jobs_dir / f'jobs-{group}.txt').write_text('\n'.join(line for _, _, line in jobs(plan, group)) + '\n')
        return
    if args.source_root is None or args.out_root is None:
        parser.error('--source-root and --out-root are required for staging')
    try:
        prepare(plan, args.source_root.resolve(), args.out_root.resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
