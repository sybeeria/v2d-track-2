"""Sequential recipe arm x manifest sequence x input x checkpoint pilot. See pilot_queue.sh --help."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path

from score_rollout import METRICS, ROOT, write_json

FLASH = ROOT / 'external/video_to_data/robotic_grounding/flash_chord'
EVALUATOR = ROOT / 'external/kit/v2d_submission_kit/flash_chord/scripts/evaluate_policy.py'


def evaluator_identity(flash):
    # The kit's eval_code_sha256 covers only the evaluation modules; applied env/objective patches matter too.
    def git(*args):
        return subprocess.run(['git', '-C', str(flash), *args], capture_output=True, check=True).stdout
    digest = hashlib.sha256(EVALUATOR.read_bytes())
    digest.update(git('rev-parse', 'HEAD'))
    digest.update(git('diff', 'HEAD', '--binary', '--', '.'))
    for name in sorted(filter(None, git('ls-files', '--others', '--exclude-standard', '-z', '--', '.').split(b'\0'))):
        digest.update(name + b'\0' + (flash / name.decode()).read_bytes())
    return digest.hexdigest()


def checked_record(record):
    path = Path(record['path'])
    path = (ROOT / path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record['sha256']:
        raise ValueError(f'{path}: manifest SHA-256 mismatch; regenerate the screen')
    return {'path': str(path), 'sha256': digest}


def read_inputs(path, sequences, inputs):
    manifest = json.loads(path.read_text())
    records = {s['sequence']: s for s in manifest['sequences']}
    if len(records) != len(manifest['sequences']):
        raise ValueError('duplicate sequences in manifest')
    # 'clean' (default): every input is scored against the manifest's clean input (the screen).
    # 'self': label-free mode for Tier 3, where each input is scored against its own training reference.
    mode = manifest.get('scoring_mode', 'clean')
    if mode not in ('clean', 'self'):
        raise ValueError(f'unknown scoring_mode {mode!r}')
    tier3 = manifest.get('tier') == 3 or any(
        label.startswith('tier3_') for entry in records.values() for label in entry['arms'])
    if tier3 and mode != 'self':
        raise ValueError('Tier 3 inputs require scoring_mode self')
    if tier3 and mode == 'self' and any('clean' in entry['arms'] for entry in records.values()):
        raise ValueError('Tier 3 self-scoring manifests cannot contain a clean input')
    selected = list(dict.fromkeys(sequences or records))
    if set(selected) - records.keys():
        raise ValueError(f'unknown manifest sequences: {sorted(set(selected) - records.keys())}')
    jobs = {}
    for sequence in selected:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sequence):
            raise ValueError(f'invalid sequence name: {sequence!r}')
        entry = records[sequence]
        if mode == 'clean':
            clean = checked_record(entry['clean_scoring_reference'])
            if clean != checked_record(entry['arms']['clean']['training_reference']):
                raise ValueError(f'{sequence}: scoring reference must be the manifest clean input')
            for label, arm in entry['arms'].items():
                if label != 'clean' and Path(clean['path']).samefile(ROOT / arm['training_reference']['path']):
                    raise ValueError(f'{sequence}: clean scoring reference equals noisy training input {label}')
        for label in dict.fromkeys(inputs):
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', label) or label not in entry['arms']:
                raise ValueError(f'{sequence}: unknown input {label!r}')
            arm = entry['arms'][label]
            source = (ROOT / arm['training_reference']['path']).resolve()
            if tier3 and mode == 'self' and (
                    'tier_1_multiview_caption' in source.parts
                    or source.is_relative_to((ROOT / 'data/work/tier1').resolve())):
                raise ValueError('self-scoring cannot use a Tier 1 training reference')
            training = checked_record(arm['training_reference'])
            scoring = checked_record(arm['clean_scoring_reference'])
            if mode == 'self' and scoring != training:
                raise ValueError(f'{sequence}/{label}: a self-scored input must name its own training reference as its scoring reference')
            if mode == 'clean' and scoring != clean:
                raise ValueError(f'{sequence}/{label}: input scoring reference differs from clean scoring reference')
            support = {**checked_record(arm['support']), 'enabled': arm['support']['enabled']}
            jobs[f'{label}/{sequence}'] = dict(input=label, sequence=sequence, training=training,
                                              scoring=scoring, support=support)
    return jobs


def read_arms(path, steps):
    arms = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, *overrides = line.split()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in arms:
            raise ValueError(f'{path}:{number}: invalid or duplicate arm name {name!r}')
        values = {}
        for override in overrides:
            key, separator, value = override.partition('=')
            key = key.lstrip('+')
            if not separator or not value or key in values:
                raise ValueError(f'{path}:{number}: invalid or duplicate override {override!r}')
            if key.startswith(('task.', 'output_dir', 'hydra.', 'training.resume', 'training.checkpoint_path', 'scene.support')):
                raise ValueError(f'{path}:{number}: {key} would bypass pilot isolation or the full-frame manifest input')
            values[key] = value
        total = int(values['training.total_environment_steps'])
        worlds = int(values.get('scene.world_count', '2048'))
        interval = int(values.get('training.checkpoint.save_interval_environment_steps', '10000384'))
        if min(total, worlds, interval) <= 0 or total % worlds or interval % worlds:
            raise ValueError(f'{name}: total steps and checkpoint interval must be positive multiples of world_count')
        if 'curriculum.thresholds' in values:
            thresholds = json.loads(values['curriculum.thresholds'])
            if (not thresholds or any(type(t) is not int or t <= 0 or t % worlds for t in thresholds)
                    or thresholds != sorted(set(thresholds)) or thresholds[-1] != total):
                raise ValueError(f'{name}: thresholds must increase, divide by world_count, and end at total steps')
        if any(s <= 0 or s > total or (s != total and s % interval) for s in steps):
            raise ValueError(f'{name}: requested steps must be saved interval multiples or the final {total}')
        arms[name] = (overrides, total)
    if not arms:
        raise ValueError(f'{path}: no arms')
    return arms


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def run(command, *, cwd, dry_run, env=None):
    prefix = ['env', *[f'{k}={v}' for k, v in (env or {}).items()]] if env else []
    print(f'(cd {shlex.quote(str(cwd))} && {shlex.join([*prefix, *map(str, command)])})', flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env={**os.environ, **(env or {})}, check=True)


def summary(candidates, out):
    rows = []
    for arm, input_name, sequence, step, scores, reference, trajectory in candidates:
        report = json.loads(scores.read_text())
        episodes = report.get('episodes', [])
        if (len(episodes) != 1 or Path(episodes[0]['reference']).resolve() != Path(reference).resolve()
                or Path(episodes[0]['rollout']).resolve() != trajectory.resolve()
                or episodes[0].get('reference_sha256') != hashlib.sha256(Path(reference).read_bytes()).hexdigest()
                or episodes[0].get('rollout_sha256') != hashlib.sha256(trajectory.read_bytes()).hexdigest()):
            raise ValueError(f'{scores}: scores do not match the frozen reference and rollout')
        metrics = report['mean']
        if set(metrics) != set(METRICS) or not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError(f'{scores}: invalid scores')
        rows.append(dict(arm=arm, input=input_name, sequence=sequence, step=step, **metrics))
    write_json(out / 'summary.json', {'candidates': rows, 'mppe_unit': 'cm'})
    table = out / 'summary.tsv'
    with table.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['arm', 'input', 'sequence', 'step', *METRICS], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(table.read_text(), end='')

def run_sequence(job, arm_dir, overrides, total, args, python, pack_python, flash, evaluator):
    """Train, evaluate and score one manifest sequence; return its summary candidates."""
    candidates = []
    sequence, input_name = job['sequence'], job['input']
    reference = Path(job['training']['path'])
    train_root = arm_dir / input_name / 'train'
    run_dir = train_root / sequence
    if not args.summarize_only and not args.evaluate_only:
        if nonempty(run_dir / f'policy_{total}.safetensors'):
            print(f'# skip training: {run_dir}')
        else:
            run([ROOT / 'scripts/train_queue.sh', args.gpu, train_root, reference], cwd=ROOT,
                dry_run=args.dry_run, env={'PYTHON': python, 'FLASH_ROOT': str(flash),
                                          'EXTRA_ARGS': ' '.join(overrides)})
            if not args.dry_run and not nonempty(run_dir / f'policy_{total}.safetensors'):
                raise ValueError(f'{run_dir}: training returned without final policy_{total}.safetensors')
    for step in args.steps:
        checkpoint = run_dir / f'policy_{step}.safetensors'
        result = arm_dir / input_name / 'evaluation' / sequence / str(step)
        trajectory, metrics = result / 'trajectory.parquet', result / 'metrics.json'
        done, scores = result / 'evaluation.done.json', result / 'scores.json'
        candidates.append((arm_dir.name, input_name, sequence, step, scores, job['scoring']['path'], trajectory))
        if not args.dry_run and not nonempty(checkpoint):
            raise FileNotFoundError(f'missing requested checkpoint: {checkpoint}')
        identity = None if args.dry_run else {
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'evaluator': evaluator,
            'input': input_name, 'training_reference': job['training'],
            'scoring_reference': job['scoring'], 'support': job['support']}
        sealed_identity = None if args.dry_run or not (nonempty(trajectory) and nonempty(metrics)) else {
            **identity, 'trajectory_sha256': hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            'metrics_sha256': hashlib.sha256(metrics.read_bytes()).hexdigest()}
        try:
            finished = (nonempty(done) and nonempty(trajectory) and nonempty(metrics)
                        and (args.dry_run or json.loads(done.read_text()) == sealed_identity))
        except ValueError:
            finished = False
        if args.summarize_only:
            if not finished:
                raise ValueError(f'{result}: missing or stale sealed evaluation; rerun evaluation/scoring explicitly')
            continue
        if not finished:
            if not args.dry_run:
                result.mkdir(parents=True, exist_ok=True)
                done.unlink(missing_ok=True)
                # Outputs from an older attempt cannot prove that this evaluator invocation produced them.
                trajectory.unlink(missing_ok=True)
                metrics.unlink(missing_ok=True)
                write_json(result / 'evaluation.attempt.json', identity)
            run([python, EVALUATOR,
                 f'evaluation.checkpoint={checkpoint}', 'evaluation.world_count=1',
                 'evaluation.reset_mode=explicit', f'evaluation.metrics_output={metrics}',
                 f'evaluation.object_trajectories_output={trajectory}',
                 f'hydra.run.dir={result / "hydra"}', 'logging.mode=disabled'],
                cwd=flash, dry_run=args.dry_run, env={'CUDA_VISIBLE_DEVICES': args.gpu})
            if not args.dry_run:
                if not (nonempty(trajectory) and nonempty(metrics)):
                    raise ValueError(f'{result}: evaluator returned without both outputs')
                write_json(done, {**identity,
                    'trajectory_sha256': hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                    'metrics_sha256': hashlib.sha256(metrics.read_bytes()).hexdigest()})
        else:
            print(f'# skip evaluation: {result}')
        # Re-score CPU outputs on every pass: this also validates cached trajectories.
        run([pack_python, ROOT / 'scripts/score_rollout.py', '--rollout', trajectory,
             '--reference', job['scoring']['path'], '--output', scores], cwd=ROOT, dry_run=args.dry_run)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', type=Path, required=True, help='name + whitespace-free Hydra overrides per line')
    parser.add_argument('--arm', action='append', help='select an arm (repeatable; default all)')
    parser.add_argument('--manifest', type=Path, required=True, help='screen manifest from make_tier2_screen.py')
    parser.add_argument('--inputs', nargs='+', default=['clean'], help='manifest inputs (default: clean)')
    parser.add_argument('--sequences', nargs='+', help='select manifest sequences (default: all)')
    parser.add_argument('--steps', nargs='+', type=int, required=True,
                        help='exact policy_<steps>.safetensors numbers (e.g. 10000384 49999872)')
    parser.add_argument('--out-root', type=Path, required=True, help='fresh experiment directory; never shared across recipes')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dry-run', action='store_true', help='print subprocess commands; create no outputs and run no GPU code')
    parser.add_argument('--summarize-only', action='store_true', help='rebuild summary from existing scores.json files')
    parser.add_argument('--evaluate-only', action='store_true', default=os.environ.get('PILOT_EVALUATE_ONLY') == '1',
                        help='never train; fail on missing requested checkpoints (also PILOT_EVALUATE_ONLY=1)')
    args = parser.parse_args()
    try:
        arms = read_arms(args.arms, args.steps)
        if args.arm:
            unknown = set(args.arm) - arms.keys()
            if unknown:
                raise ValueError(f'unknown arms: {sorted(unknown)}')
            arms = {n: arms[n] for n in dict.fromkeys(args.arm)}
        references = read_inputs(args.manifest, args.sequences, args.inputs)
        args.steps = sorted(set(args.steps))
        out = args.out_root.resolve()
        # Resolving a venv's interpreter symlink would silently select system Python.
        python = str(Path(os.environ.get('PYTHON', ROOT / '.venv-flash/bin/python')).absolute())
        pack_python = str(Path(os.environ.get('PACK_PYTHON', ROOT / '.venv/bin/python')).absolute())
        flash = Path(os.environ.get('FLASH_ROOT', FLASH)).resolve()
        evaluator = None if args.dry_run else evaluator_identity(flash)
        candidates = []
        failures = []
        jobs = []
        for name, (overrides, total) in arms.items():
            arm_dir = out / name
            # Freeze the recipe before any checkpoint can be resumed in this directory.
            recipe = {'schema_version': 2, 'overrides': overrides, 'references': references}
            manifest = arm_dir / 'recipe.json'
            if manifest.exists():
                saved = json.loads(manifest.read_text())
                if (saved.get('schema_version') != 2 or saved['overrides'] != recipe['overrides'] or any(
                        s in saved['references'] and saved['references'][s] != h
                        for s, h in recipe['references'].items())):
                    raise ValueError(f'{arm_dir}: recipe, input, scoring reference or support changed; choose a fresh --out-root')
                recipe['references'] = {**saved['references'], **recipe['references']}
            elif arm_dir.exists() and any(arm_dir.iterdir()):
                raise ValueError(f'{arm_dir}: nonempty directory without recipe.json; choose a fresh --out-root')
            jobs.append((arm_dir, manifest, recipe, references, overrides, total))
        for arm_dir, manifest, recipe, references, overrides, total in jobs:
            if not args.dry_run and not args.summarize_only:
                write_json(manifest, recipe)
            for job in references.values():
                # One crashed training, evaluation or scoring must not cancel the arm's other sequences.
                try:
                    candidates += run_sequence(job, arm_dir, overrides, total, args, python, pack_python,
                                               flash, evaluator)
                except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
                    if args.dry_run or args.summarize_only:
                        raise
                    failures.append(f"{arm_dir.name}/{job['input']}/{job['sequence']}: {error}")
                    print(f'# FAILED {failures[-1]}', flush=True)
        if args.dry_run:
            print(f'# write {out / "summary.json"} and {out / "summary.tsv"} from the scores above')
        else:
            summary(candidates, out)
            if failures:
                raise ValueError(f'{len(failures)} sequence(s) failed and are missing from the summary:\n'
                                 + '\n'.join(failures))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
