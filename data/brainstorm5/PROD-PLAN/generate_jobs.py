#!/usr/bin/env python3
"""Generate production plans, validate real inputs, compose arms, and dry-run queues.

All writes are confined to this review folder. No CUDA, installation, packing,
provider command, or network operation is performed. Re-run after host retargeting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FLASH = ROOT / 'external/video_to_data/robotic_grounding/flash_chord'
sys.path[:0] = [str(ROOT / 'scripts'), str(FLASH / 'src')]
FULL = [250009600, 290011136, 310011904, 330012672, 349999104]
SHORT = 49999872
OPTIONS = {'A': (350, 350, 350), 'B': (350, 50, 50),
           'C': (50, 50, 50), 'H': (350, 100, 50)}


def write(path, value):
    path = path.resolve()
    if not path.is_relative_to(HERE):
        raise ValueError('output must remain inside PROD-PLAN')
    path.parent.mkdir(parents=True, exist_ok=True)
    content = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n'
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(content)
    temporary.replace(path)


def record(path):
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def load_arms():
    def source(name):
        return {p[0]: p[1:] for line in (ROOT / 'scripts' / name).read_text().splitlines()
                if (p := line.split()) and not p[0].startswith('#')}
    short, full = source('pilot_arms.txt'), source('pilot_arms_350m.txt')
    arms = {'F6-rebalance-tail': full['F6-rebalance-tail'],
            'E6-rebalance-tail': short['E6-rebalance-tail'],
            'T2C-contact': short['T2C-contact']}
    for budget, total, thresholds in (
        (350, FULL[-1], json.loads(full['F6-rebalance-tail'][1].split('=', 1)[1])),
        (100, SHORT * 2, [int(x) * 2 for x in json.loads(short['T2C-contact'][1].split('=', 1)[1])])):
        overrides = {s.split('=', 1)[0]: s.split('=', 1)[1] for s in short['T2C-contact']}
        overrides['training.total_environment_steps'] = str(total)
        overrides['curriculum.thresholds'] = json.dumps(thresholds, separators=(',', ':'))
        arms[f'T2C-contact-{budget}M'] = [f'{k}={v}' for k, v in overrides.items()]
    return arms


def compose_arms(arms):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from pilot_queue import read_arms
    result = []
    with initialize_config_dir(config_dir=str(FLASH / 'src/flash_chord/configs'), version_base='1.3'):
        for name, overrides in arms.items():
            total = int(next(s.split('=', 1)[1] for s in overrides if s.startswith('training.total_environment_steps=')))
            cfg = compose(config_name='train_flash_sac', overrides=[
                'experiment=g1_recon_body_flash_sac', *overrides])
            stages = OmegaConf.to_container(cfg.training.curriculum.stages, resolve=True)
            thresholds = list(cfg.training.curriculum.thresholds)
            assert cfg.training.total_environment_steps == total == thresholds[-1]
            assert len(stages) == 9 and all(t % 2048 == 0 for t in thresholds)
            assert stages[-1]['voc_scale'] == 0.0
            assert cfg.reset.seed == 42
            if name.startswith('T2C'):
                assert all(s['objective_weights']['palm_position'] == .25 for s in stages)
                assert all(s['objective_weights']['force_closure'] == 10 for s in stages)
            else:
                assert all(s['objective_weights']['object_position'] == 6 for s in stages[6:])
                assert all(s['objective_weights']['object_orientation'] == 2 for s in stages[6:])
            result.append({'arm': name, 'steps': total, 'thresholds': thresholds,
                           'last_stage': stages[-1], 'qualification':
                           'new budget; configuration only' if name.endswith(('100M', '350M'))
                           else 'existing frozen arm; configuration only'})
    # The real queue parses every generated arm, including checkpoint arithmetic.
    for row in result:
        parsed = read_arms(HERE / 'generated/arms.txt', [])[row['arm']]
        assert parsed[1] == row['steps']
    return result


def inspect_reference(path, frames):
    import numpy as np
    import pyarrow.parquet as pq
    from flash_chord.assets.registry import resolve_asset_path, support_usda_for_parquet
    from flash_chord.data.motion_v1 import load_motion_v1
    from pxr import Usd
    if not path.is_file():
        return None, [str(path)], []
    row = pq.ParquetFile(path).read(columns=['object_mesh_paths', 'object_urdf_paths',
        'object_body_position', 'robot_joint_positions', 'coord_frame']).to_pylist()[0]
    support = support_usda_for_parquet(str(path), robot_name='g1')
    paths = [Path(resolve_asset_path(asset, reference_path=path))
             for field in ('object_mesh_paths', 'object_urdf_paths')
             for asset in row[field] if asset]
    missing = [str(p) for p in paths if not p.is_file()]
    if support is None or not Path(support).is_file():
        missing.append(str(support) if support else f'{path}: unresolved support')
    if missing:
        return None, missing, []
    stage = Usd.Stage.Open(str(support))
    if stage is None:
        raise ValueError(f'cannot open {support}')
    enabled = any(p.GetTypeName() in ('Cube', 'Cylinder') for p in stage.Traverse())
    if len(row['object_body_position']) != frames:
        raise ValueError(f'{path}: source frame count mismatch')
    if not all(np.isfinite(row[key]).all() for key in ('object_body_position', 'robot_joint_positions')):
        raise ValueError(f'{path}: non-finite motion')
    load_motion_v1(str(path), control_fps=50., source_frame_playback=True)
    ref = record(path)
    arm = {'training_reference': ref, 'clean_scoring_reference': ref,
           'support': {**record(Path(support)), 'enabled': enabled}}
    return arm, [], [record(p) for p in sorted(set(paths))]


def build_inputs(args, episodes):
    import numpy as np
    import pyarrow.parquet as pq
    from contacts import DEFAULT_PARAMS, robust_labels
    from hoi import Sequence
    from robotic_grounding.retarget.read_soma import SOMA
    soma = SOMA(identity_model_type='mhr')
    provided = json.loads(args.tier3_manifest.read_text())
    if provided.get('scoring_mode') != 'self':
        raise ValueError('Tier 3 manifest must be self-scored')
    tier3 = {s['sequence']: s for s in provided['sequences']}
    inventory, manifests = [], {}
    for tier, work, label in ((1, args.tier1_work, 'clean'),
                               (2, args.tier2_work, 'raw_repaired'), (3, args.tier3_work, 'tier3_repaired')):
        manifest = {'schema_version': 1, 'scoring_mode': 'self',
                    'metric_space': 'own tier input only; T1 clean is used for T1 only', 'sequences': []}
        for ep in episodes:
            sequence, eid = ep['sequence_id'], ep['episode_index']
            folder = work / 'sequences' / sequence
            parquet = work / 'motion/whole_body/soma' / f'sequence_id={sequence}/robot_name=g1/data.parquet'
            supplied = tier3.get(sequence, {}).get('arms', {}).get(label)
            if tier == 3 and supplied:
                parquet = Path(supplied['training_reference']['path'])
                # The accepted portable closure is accompanied by this reconstruction's source.
                if 'human_motion_data' in parquet.parts:
                    base = parquet.parents[5].parent
                    folder = base / 'sequences' / sequence
            dependencies = [folder / 'poses.npy', folder / 'soma_params.npz',
                            folder / 'ground_plane.json', folder / 'reconstructed_mesh/output_aligned.glb']
            if tier < 3:
                released = ROOT / 'data/challenge/track_2' / ('tier_1_multiview_caption' if tier == 1 else 'tier_2_synthetic_noise')
                dependencies += [released / f'data/chunk-000/episode_{eid:06d}.parquet']
            missing = [str(p) for p in dependencies if not p.is_file()]
            gates, details = [], []
            try:
                arm, absent, assets = inspect_reference(parquet, ep['frames'])
                missing += absent
                if arm and tier == 3 and supplied:
                    if supplied['training_reference'] != supplied['clean_scoring_reference']:
                        raise ValueError('Tier 3 input is not self-scored')
                    if arm['training_reference']['sha256'] != supplied['training_reference']['sha256']:
                        raise ValueError('Tier 3 manifest hash mismatch')
                if not missing:
                    if len(np.load(folder / 'poses.npy', mmap_mode='r')) != ep['frames']:
                        raise ValueError('poses.npy frame count mismatch')
                    if tier == 2:
                        seq = Sequence.load(folder, soma)
                        expected = robust_labels(seq.hand_gaps(), seq.object_moving(), DEFAULT_PARAMS, seq.foot_gaps())
                        actual = np.asarray(pq.ParquetFile(parquet).read(columns=['hand_contact_active']).column(0)[0].as_py())
                        if not np.array_equal(expected.astype(float), actual):
                            gates.append('contact labels differ from contacts.py DEFAULT_PARAMS')
                        details.append({'contact_frames_expected': expected.sum(axis=1).tolist(),
                                        'contact_frames_actual': actual.sum(axis=1).tolist()})
            except (ValueError, OSError, KeyError) as exc:
                arm, assets = None, []
                gates.append(str(exc))
            if tier == 3:
                gates.append('released scene asset organizer ruling pending')
                if eid == 18 and arm and arm['training_reference']['sha256'] == 'affba27c44a602e83befd2bb4ac07afc7e271e810000f234d2954188840ff4be':
                    gates.append('existing repaired ep18 fails frozen 10 cm self-MPPE learnability veto (62.2 cm)')
                else:
                    gates.append('reconstruction and learnability gate not completed')
            if arm and not missing:
                manifest['sequences'].append({'sequence': sequence, 'frames': ep['frames'], 'arms': {label: arm}})
            inventory.append({'tier': tier, 'episode': eid, 'sequence': sequence, 'frames': ep['frames'],
                'reference': str(parquet), 'input': label, 'missing': sorted(set(missing)), 'gates': gates,
                'ready': not missing and not gates, 'details': details,
                'dependencies': [record(p) for p in dependencies if p.is_file()], 'assets': assets})
        manifests[tier] = manifest
    return inventory, manifests


def jobs_and_verify(arms, inventory, manifests, verify):
    from pilot_queue import read_arms
    counts, checked = {}, []
    arms_path = HERE / 'generated/arms.txt'
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
           'OPENBLAS_NUM_THREADS': '2', 'VECLIB_MAXIMUM_THREADS': '2'}
    for option, budgets in OPTIONS.items():
        for tier, budget in enumerate(budgets, 1):
            name = ('F6-rebalance-tail' if budget == 350 else 'E6-rebalance-tail') if tier == 1 else (
                'T2C-contact' if budget == 50 else f'T2C-contact-{budget}M')
            steps = FULL if tier == 1 and budget == 350 else [int(next(s.split('=', 1)[1] for s in arms[name] if s.startswith('training.total_environment_steps=')))]
            assert read_arms(arms_path, [])[name][1] >= max(steps)
            # Validate selected arm alone, because read_arms validates every line before --arm filtering.
            isolated = HERE / f'generated/arms/{name}.txt'
            write(isolated, ' '.join([name, *arms[name]]) + '\n')
            read_arms(isolated, steps)
            arm_arg = os.path.relpath(isolated, ROOT / 'scripts')
            rows = [r for r in inventory if r['tier'] == tier]
            lines = [f"{option}-t{tier}-ep{r['episode']:02d}|{arm_arg}|{name}|{r['input']}|{r['sequence']}|{' '.join(map(str, steps))}" for r in rows]
            outdir = HERE / 'generated' / option / f'tier{tier}'
            write(outdir / 'jobs.txt', '\n'.join(lines) + '\n')
            write(outdir / 'ready_jobs.txt', ''.join(line + '\n' for r, line in zip(rows, lines) if r['ready']))
            for host in (0, 1):
                write(outdir / f'host{host}_jobs.txt', '\n'.join(lines[host::2]) + '\n')
            present = {s['sequence'] for s in manifests[tier]['sequences']}
            available = [line for r, line in zip(rows, lines) if r['sequence'] in present]
            write(outdir / 'available_jobs.txt', '\n'.join(available) + ('\n' if available else ''))
            counts[f'{option}/tier{tier}'] = {'planned': len(lines), 'available': len(available),
                'ready': sum(r['ready'] for r in rows), 'arm': name, 'steps': steps}
            if verify and available:
                log = outdir / 'queue_dryrun.log'
                command = ['bash', str(ROOT / 'scripts/test_day.sh'), '2', str(outdir / 'DRY-RUN-ONLY'), '--dry-run']
                with log.open('w') as stream:
                    run = subprocess.run(command, cwd=ROOT, env={**env,
                        'JOBS_FILE': str(outdir / 'available_jobs.txt'),
                        'MANIFEST': str(HERE / f'generated/tier{tier}_manifest.json')},
                        stdout=stream, stderr=subprocess.STDOUT)
                data = log.read_text()
                expected = (len(available), len(available) * len(steps), len(available) * len(steps))
                observed = tuple(data.count(s) for s in ('/scripts/train_queue.sh ', '/evaluate_policy.py ', '/scripts/score_rollout.py '))
                if run.returncode or observed != expected:
                    raise ValueError(f'{log}: queue dry-run failed, rc={run.returncode}, {observed} != {expected}')
                checked.append({'plan': f'{option}/tier{tier}', 'returncode': run.returncode,
                                'train_evaluate_score': observed})
    return counts, checked


def reconstruction(episodes):
    source = ROOT / 'data/challenge/track_2/tier_2_synthetic_noise'
    rows, inventory = [], []
    for ep in episodes:
        eid, seq = ep['episode_index'], ep['sequence_id']
        paths = [source / f'videos/chunk-000/observation.images.exo_camera/episode_{eid:06d}.mp4',
                 ROOT / f'data/tier3_bridge/prompts/ep{eid:02d}.json', source / ep['mesh']]
        rows.append('\t'.join([seq, str(ep['frames']), *map(str, paths)]))
        inventory.append({'episode': eid, 'missing': [str(p) for p in paths if not p.is_file()],
                          'inputs': [record(p) for p in paths if p.is_file()],
                          'gate': 'engineering-only released mesh; organizer ruling pending'})
    write(HERE / 'generated/reconstruct30.tsv', '\n'.join(rows) + '\n')
    write(HERE / 'generated/reconstruction_inputs.json', inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier1-work', type=Path, default=ROOT / 'data/work/tier1')
    parser.add_argument('--tier2-work', type=Path, default=ROOT / 'data/work/tier2')
    parser.add_argument('--tier3-work', type=Path, default=ROOT / 'data/work/tier3')
    parser.add_argument('--tier3-manifest', type=Path, default=ROOT / 'data/tier3_bridge/manifest.json')
    parser.add_argument('--verify', action='store_true', help='compose arms and run existing queue dry runs')
    args = parser.parse_args()
    episodes = [json.loads(line) for line in (ROOT / 'data/challenge/track_2/tier_2_synthetic_noise/meta/episodes_metadata.jsonl').read_text().splitlines()]
    if len(episodes) != 30 or len({e['sequence_id'] for e in episodes}) != 30 or len({e['episode_index'] for e in episodes}) != 30:
        raise ValueError('expected 30 unique released episodes')
    episodes.sort(key=lambda e: e['episode_index'])
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', e['sequence_id']) for e in episodes):
        raise ValueError('unsafe sequence ID')
    arms = load_arms()
    write(HERE / 'generated/arms.txt', '\n'.join(' '.join([k, *v]) for k, v in arms.items()) + '\n')
    inventory, manifests = build_inputs(args, episodes)
    for tier, manifest in manifests.items():
        write(HERE / f'generated/tier{tier}_manifest.json', manifest)
    write(HERE / 'generated/input_inventory.json', inventory)
    text = ['# Per-episode readiness [M]', '', 'Missing = absent file; gate = present input not approved for production.', '', '|Tier|Episode|Status|Missing inputs / gates|', '|---|---:|---|---|']
    for row in inventory:
        reasons = row['missing'] + row['gates']
        text.append(f"|{row['tier']}|{row['episode']:02d}|{'READY' if row['ready'] else 'BLOCKED'}|" + '<br>'.join(reasons) + '|')
    write(HERE / 'generated/MISSING_INPUTS.md', '\n'.join(text) + '\n')
    reconstruction(episodes)
    composition = compose_arms(arms) if args.verify else []
    counts, dryruns = jobs_and_verify(arms, inventory, manifests, args.verify)
    report = {'counts': counts, 'composition': composition, 'dryruns': dryruns,
              'scope': 'local CPU paths and composition; no CUDA, rollout, packaging or eligibility proof',
              'gate': 'READY is file/label readiness only; canary and approvals still required'}
    write(HERE / 'generated/verification.json', report)
    print(json.dumps({'counts': counts, 'composed': len(composition), 'dryruns': len(dryruns)}, indent=2))


if __name__ == '__main__':
    main()
