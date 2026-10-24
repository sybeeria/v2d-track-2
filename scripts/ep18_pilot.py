#!/usr/bin/env python3
"""Host input admission and mechanical decision for the registered candidate_v1 pair."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = '2026-05-06_10-00-19_blue_trash_can_drag_sideways_08'
LABEL = 'tier3_support_static'
READY = Path('data/brainstorm6/PROD-READY')
ASSETS = Path('data/brainstorm5/T3-REPAIR/candidate_v1/assets')
REFERENCE = Path(f'human_motion_data/whole_body/soma/sequence_id={SEQUENCE}/robot_name=g1/data.parquet')
SUPPORT = Path(f'human_motion_data/whole_body/reconstructed_stage/{SEQUENCE}_support.usda')
PAIR = {'ep18-qualified350': ('T3P-contact350', 349999104),
        'ep18-qualified50': ('T3P-control-s42', 49999872)}

ARM_SHA256 = {'T3P-contact350': 'd09624faab31dd189b1af4ca7260d6c49fd0e25d0e79b813bc161c097449120e', 'T3P-control-s42': '61677a740cc689fd9a006dae9e7cbf59d901a01a64587edc7731b90d2c7fb180'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_jobs(path):
    rows = [line.split('|') for line in path.read_text().splitlines() if line and not line.startswith('#')]
    expected = [[name, f'../{READY}/generated/arms/{arm}.txt', arm, LABEL, SEQUENCE, str(step)]
                for name, (arm, step) in PAIR.items()]
    for arm,h in ARM_SHA256.items():
        if digest(ROOT/READY/'generated/arms'/f'{arm}.txt')!=h:
            raise ValueError('registered ep18 recipe changed')
    if sorted(rows) != sorted(expected):
        raise ValueError('Tier 3 accepts only the registered ep18 50M/350M pilot pair')


def admit(binding, output):
    for arm,h in ARM_SHA256.items():
        if digest(ROOT/READY/'generated/arms'/f'{arm}.txt')!=h:
            raise ValueError('registered ep18 recipe changed')
    frozen = json.loads(binding.read_text())
    base = ROOT / frozen['assets']
    actual = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
    if actual != set(frozen['sha256']):
        raise ValueError('candidate file inventory changed')
    # Verify the entire closure before writing a manifest or permitting any trainer.
    for relative, expected in frozen['sha256'].items():
        path = base / relative
        if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()) or digest(path) != expected:
            raise ValueError(f'candidate SHA-256 mismatch: {relative}')
    def record(relative):
        return {'path': str((base / relative).resolve()), 'sha256': frozen['sha256'][str(relative)]}
    reference = record(REFERENCE)
    manifest = {'schema_version': 1, 'tier': 3, 'scoring_mode': 'self', 'sequences': [
        {'sequence': SEQUENCE, 'frames': 858, 'arms': {LABEL: {
            'training_reference': reference, 'clean_scoring_reference': reference,
            'support': {**record(SUPPORT), 'enabled': False}}},
         'assets': [record(p) for p in frozen['sha256'] if p not in (str(REFERENCE), str(SUPPORT))]}]}
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, indent=2) + '\n'
    if output.exists() and output.read_text() != content:
        raise ValueError('frozen pilot manifest changed')
    output.write_text(content)
    print('EP18 INPUT PASS: all candidate files match staged SHA-256; self scoring; support disabled')


def decision_record(qualification):
    try:
        results = json.loads(qualification.read_text())['candidates']
        by_name = {r['name']: r for r in results}
        valid = len(results) == 2 and set(by_name) == set(PAIR)
        passed = {name: valid and by_name[name]['go'] is True and len(by_name[name]['members']) == 3
                  and all(m['go'] is True for m in by_name[name]['members']) for name in PAIR}
    except (OSError, ValueError, KeyError, TypeError):
        passed = dict.fromkeys(PAIR, False)
    decision = ('retain 50M' if passed['ep18-qualified50'] else
                'selective rescue' if passed['ep18-qualified350'] else 'diagnose')
    return {'decision': decision, 'passes': passed, 'broad_tier3_policy_spend': False,
            'qualification_sha256': digest(qualification) if qualification.is_file() else None}


def decide(qualification, output):
    result = decision_record(qualification)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    pre = sub.add_parser('admit'); pre.add_argument('--binding', type=Path, required=True)
    pre.add_argument('--output', type=Path, required=True)
    post = sub.add_parser('decide'); post.add_argument('--qualification', type=Path, required=True)
    post.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'admit': admit(args.binding, args.output)
        else: decide(args.qualification, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
