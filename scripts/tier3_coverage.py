#!/usr/bin/env python3
"""Stage and admit all Tier 3 coverage inputs under decision D2 and the ep18 pilot verdict.

jobs:  print the job lines for every PASS/HOLD entry of a bridge coverage_manifest.json.
stage: copy the reference closures named by a jobs file into a launch payload, with SHA-256 binding.
admit: on the host, verify every staged file and write the self-scoring manifest the trainers read.
Episode 18 always uses the registered pilot candidate (ep18_pilot.ASSETS), so the pilot tests what ships.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ep18_pilot import (ASSETS as EP18_ASSETS, LABEL as EP18_LABEL, SEQUENCE as EP18_SEQUENCE,
                       READY, PAIR, decision_record, ARM_SHA256)

ROOT = Path(__file__).resolve().parents[1]
ARM = 'T2C-contact'
FINAL_STEP = 49999872
ROSTER = Path('data/challenge/track_2/tier_2_synthetic_noise/meta')  # Tier 3 shares Tier 2's episodes and frame grid
ROSTER_FILES = ('episodes_metadata.jsonl', 'info.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_rel(sequence):
    return f'human_motion_data/whole_body/soma/sequence_id={sequence}/robot_name=g1/data.parquet'


def support_rel(sequence):
    return f'human_motion_data/whole_body/reconstructed_stage/{sequence}_support.usda'


def asset_root(reference, sequence):
    root = Path(str(reference)[:-len(reference_rel(sequence))])
    if root / reference_rel(sequence) != Path(reference):
        raise ValueError(f'reference is not inside a materialized asset root: {reference}')
    return root


def closure(root, sequence):
    """Reference, support stage and object assets of one sequence; nothing else under the root."""
    files = sorted(p for p in (root / f'human_motion_data/whole_body/soma/sequence_id={sequence}').rglob('*') if p.is_file())
    support = root / support_rel(sequence)
    if root / reference_rel(sequence) not in files or not support.is_file():
        raise ValueError(f'incomplete asset closure for {sequence}')
    if any(p.is_symlink() for p in [*files, support]):
        raise ValueError(f'symlinked asset in closure for {sequence}')
    staged = {p.relative_to(root).as_posix(): p for p in [support, *files]}
    import pyarrow.parquet as pq
    table = pq.read_table(root / reference_rel(sequence), columns=['object_mesh_paths', 'object_urdf_paths'])
    # The runtime resolves these below the nearest /assets/ directory; each must be a staged file.
    for column in table.column_names:
        for path in table[column][0].as_py():
            if not path.startswith('/assets/') or path[len('/assets/'):] not in staged:
                raise ValueError(f'{sequence}: {column} entry outside the staged closure: {path}')
    return staged


def checked_decision(path):
    # Always explicit: a default would let tests and tools read or clobber tonight's real pulled pilot verdict.
    path = Path(path)
    record = json.loads(path.read_text())
    expected = decision_record(path.parent / 'qualification/qualification.json')
    if expected['qualification_sha256'] is None or record != expected:
        raise ValueError('pilot decision missing its matching qualification')
    return record


def entries(coverage, decision):
    """One entry per covered episode; episode 18 is replaced by the pilot candidate."""
    manifest = json.loads(Path(coverage).read_text())
    if (manifest.get('tier'), manifest.get('scoring_mode')) != (3, 'self'):
        raise ValueError('not a Tier 3 self-scoring coverage manifest')
    metadata = [json.loads(line) for line in (ROOT / ROSTER / ROSTER_FILES[0]).read_text().splitlines()]
    roster = {r['episode_index']: (r['sequence_id'], r['frames']) for r in metadata}
    if len(roster) != len(metadata) or len(roster) != json.loads((ROOT / ROSTER / 'info.json').read_text())['total_episodes']:
        raise ValueError('invalid coverage roster')
    out = {}
    for entry in manifest['sequences']:
        if entry['episode'] not in roster or roster[entry['episode']] != (entry['sequence'], entry['frames']):
            raise ValueError(f"episode {entry['episode']}: coverage differs from the released roster")
        if entry['gate']['status'] not in ('PASS', 'HOLD'):
            raise ValueError(f"episode {entry['episode']}: only structurally valid PASS/HOLD references may be covered")
        (label, arm), = entry['arms'].items()
        reference = Path(arm['training_reference']['path'])
        if arm['clean_scoring_reference'] != arm['training_reference']:
            raise ValueError('Tier 3 coverage requires own-input scoring')
        if digest(reference) != arm['training_reference']['sha256']:
            raise ValueError(f'reference changed since the bridge wrote it: {reference}')
        if entry['episode'] in out:
            raise ValueError(f"duplicate episode {entry['episode']}")
        out[entry['episode']] = {'episode': entry['episode'], 'sequence': entry['sequence'], 'frames': entry['frames'],
                                 'label': label, 'root': asset_root(reference, entry['sequence']), 'gate': entry['gate']}
    if 18 in out and out[18]['sequence'] != EP18_SEQUENCE:
        raise ValueError('episode 18 sequence differs from the registered pilot')
    out[18] = {'episode': 18, 'sequence': EP18_SEQUENCE, 'frames': 858, 'label': EP18_LABEL, 'root': ROOT / EP18_ASSETS,
               'gate': {'status': 'PILOT_CANDIDATE', 'reasons': ['registered ep18 pilot reference replaces the bridge output'],
                        'bridge': out.get(18, {}).get('gate')}}
    if set(out) != set(roster):
        raise ValueError(f'coverage missing episodes {sorted(set(roster)-set(out))}; repair them before any policy launch')
    if decision['decision'] == 'selective rescue':
        arm, step = PAIR['ep18-qualified350']
        out[18].update(arm=arm, final_step=step)
    return out


def job_line(entry):
    arm = entry.get('arm', ARM)
    arms_file = f'../{READY}/generated/arms/{arm}.txt'
    return f"t3-ep{entry['episode']:02d}|{arms_file}|{arm}|{entry['label']}|{entry['sequence']}|{entry.get('final_step', FINAL_STEP)}"


def print_jobs(coverage, pilot_decision):
    for _, entry in sorted(entries(coverage, checked_decision(pilot_decision)).items()):
        print(job_line(entry))


def stage(coverage, jobs, payload, run_root, pilot_decision):
    """Copy the closures for the episodes in `jobs` into payload/run_root/input-assets; return the binding path."""
    decision = checked_decision(pilot_decision)
    covered = {job_line(e): e for e in entries(coverage, decision).values()}
    recipes = {str(READY / f'generated/arms/{arm}.txt'): digest(ROOT / READY / f'generated/arms/{arm}.txt')
               for arm in {line.split('|')[2] for line in covered}}
    lines = [line for line in Path(jobs).read_text().splitlines() if line and not line.startswith('#')]
    if not lines or len(set(lines)) != len(lines) or any(line not in covered for line in lines):
        raise ValueError('jobs must be distinct lines printed by `tier3_coverage.py jobs` for this coverage manifest')
    inputs = Path(run_root) / 'input-assets'
    episodes = []
    for line in lines:
        entry = covered[line]
        files = closure(entry['root'], entry['sequence'])
        hashes = {rel: digest(path) for rel, path in files.items()}
        target = inputs / f"ep{entry['episode']:02d}/assets"
        for rel, path in files.items():
            (payload / target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, payload / target / rel)
            if digest(payload / target / rel) != hashes[rel]:
                raise ValueError(f'{rel} changed during staging')
        episodes.append({'episode': entry['episode'], 'sequence': entry['sequence'], 'frames': entry['frames'],
                         'label': entry['label'], 'gate': entry['gate'], 'assets': str(target), 'sha256': hashes})
    roster = inputs / 'roster/meta'
    (payload / roster).mkdir(parents=True)
    for name in ROSTER_FILES:
        shutil.copyfile(ROOT / ROSTER / name, payload / roster / name)
    arms = {}
    for arm in sorted({line.split('|')[2] for line in lines}):
        relative = READY / f'generated/arms/{arm}.txt'
        if arm in ARM_SHA256 and digest(ROOT / relative) != ARM_SHA256[arm]:
            raise ValueError('registered ep18 recipe changed')
        (payload / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, payload / relative)
        arms[str(relative)] = digest(payload / relative)
        if arms[str(relative)] != recipes[str(relative)]:
            raise ValueError('coverage recipe changed during staging')
    binding = inputs / 'binding.json'
    (payload / binding).write_text(json.dumps({'coverage_sha256': digest(coverage), 'episodes': episodes,
        'ep18_sha256': {rel: digest(path) for rel, path in closure(ROOT / EP18_ASSETS, EP18_SEQUENCE).items()},
        'pilot_decision': decision, 'arms': arms, 'recipes_sha256': recipes,
        'roster': {name: digest(payload / roster / name) for name in ROSTER_FILES}}, indent=2) + '\n')
    return binding


def support_enabled(path):
    # Same rule the trainer applies (train_queue.sh): a stage with Cube/Cylinder prims enables supports.
    text = Path(path).read_text()
    return 'def Cube ' in text or 'def Cylinder ' in text


def admit(binding, output):
    frozen = json.loads(Path(binding).read_text())
    for relative, expected in frozen['arms'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('coverage arm changed')
    roster = Path(binding).parent / 'roster/meta'
    if {p.name for p in roster.iterdir()} != set(frozen['roster']) or any(
            digest(roster / name) != h for name, h in frozen['roster'].items()):
        raise ValueError('roster changed')
    sequences = []
    for episode in frozen['episodes']:
        base = Path(episode['assets'])
        actual = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
        if actual != set(episode['sha256']):
            raise ValueError(f"ep{episode['episode']:02d}: asset inventory changed")
        for relative, expected in episode['sha256'].items():
            path = base / relative
            if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()) or digest(path) != expected:
                raise ValueError(f"ep{episode['episode']:02d}: SHA-256 mismatch: {relative}")

        def record(relative):
            return {'path': str((base / relative).resolve()), 'sha256': episode['sha256'][relative]}
        reference, support = reference_rel(episode['sequence']), support_rel(episode['sequence'])
        sequences.append({'sequence': episode['sequence'], 'frames': episode['frames'], 'arms': {episode['label']: {
            'training_reference': record(reference), 'clean_scoring_reference': record(reference),
            'support': {**record(support), 'enabled': support_enabled(base / support)}}},
            'assets': [record(r) for r in sorted(episode['sha256']) if r not in (reference, support)]})
    identity = {key: frozen[key] for key in ('coverage_sha256', 'ep18_sha256', 'pilot_decision', 'recipes_sha256')}
    identity['roster_sha256'] = frozen['roster']
    content = json.dumps({'schema_version': 1, 'tier': 3, 'scoring_mode': 'self',
                          'tier3_coverage': identity, 'sequences': sequences}, indent=2) + '\n'
    output = Path(output)
    if output.exists() and output.read_text() != content:
        raise ValueError('frozen coverage manifest changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    print(f'TIER 3 COVERAGE INPUT PASS: {len(sequences)} episodes match staged SHA-256; self scoring')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    jobs = sub.add_parser('jobs'); jobs.add_argument('--coverage', type=Path, required=True)
    jobs.add_argument('--pilot-decision', type=Path, required=True)
    pre = sub.add_parser('admit'); pre.add_argument('--binding', type=Path, required=True)
    pre.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'jobs': print_jobs(args.coverage, args.pilot_decision)
        else: admit(args.binding, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
