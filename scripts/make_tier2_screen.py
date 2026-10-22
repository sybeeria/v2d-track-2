"""Prepare and verify the non-challenge Tier 2 factorial screen and clean controls.

Run with .venv/bin/python scripts/make_tier2_screen.py; no challenge inputs are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from analyze_denoise import CLEAN, ROOT
from contacts import DEFAULT_PARAMS as CONTACT_PARAMS, rewrite_reference, robust_labels
from denoise import DEFAULT_PARAMS, denoise
from eval_contacts import f1
from eval_denoise import SOMA_ROOT, corrupt_matched, load_metrics, metric_frames, score
from hoi import Sequence
from fit_support import fit_reference
from robotic_grounding.retarget.read_soma import SOMA

RG = ROOT / 'external/video_to_data/robotic_grounding'
FLASH = RG / 'flash_chord'
PIN = '7eb9721088615fe9201634a6be5cf48b049d1090'
ARMS = ('clean', 'raw_upstream', 'raw_repaired', 'denoised_upstream', 'denoised_repaired')


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def file_record(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha256(path)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or sha256(source) != sha256(target):
        shutil.copyfile(source, target)


def save_params(path: Path, params: dict) -> None:
    if path.exists():
        with np.load(path) as old:
            if set(old.files) == set(params) and all(np.array_equal(old[k], v) for k, v in params.items()):
                return
    # Array names can collide with np.savez keyword arguments, including "allow_pickle".
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in params.items():
            with archive.open(f'{name}.npy', 'w', force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.asanyarray(value), allow_pickle=False)


def materialize(source: Sequence, target: Sequence, folder: Path, soma: SOMA) -> float:
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('ground_plane.json', 'reconstructed_mesh/output_aligned.glb'):
        copy_file(source.folder / name, folder / name)
    with np.load(source.folder / 'soma_params.npz') as archive:
        params = dict(archive)
    if str(params['unit']) != 'meters' or bool(params['keep_root']):
        raise ValueError('Screen expects the upstream metre / keep_root=False SOMA export')
    # Recover the rigid root perturbation from corrupt_matched's output, without duplicating its RNG model.
    x = source.joints - source.joints.mean(axis=1, keepdims=True)
    y = target.joints - target.joints.mean(axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(np.einsum('tji,tjk->tik', x, y))
    r = (u @ vt).transpose(0, 2, 1)
    if np.any(np.linalg.det(r) < 0):
        raise ValueError('Human perturbation is not a proper rotation')
    if target is not source:
        params['poses'] = params['poses'].copy()
        params['poses'][:, 0] = Rotation.from_matrix(r @ Rotation.from_rotvec(params['poses'][:, 0]).as_matrix()).as_rotvec()
        params['transl'] = target.joints[:, SOMA_ROOT].copy()
    save_params(folder / 'soma_params.npz', params)
    poses = folder / 'poses.npy'
    if not poses.exists() or not np.array_equal(np.load(poses), target.object_poses_cv):
        np.save(poses, target.object_poses_cv)
    error = float(np.linalg.norm(soma.load_motion(folder / 'soma_params.npz')['joints'] - target.joints, axis=-1).max())
    if error > 1e-5:
        raise ValueError(f'{folder}: SOMA roundtrip joint error {error} m > 1e-5 m')
    np.testing.assert_array_equal(np.load(poses), target.object_poses_cv)
    return error


def upstream_stage(work: Path, sequence: str) -> dict:
    source = CLEAN.parent / 'reconstructed_stage' / f'{sequence}_support.usda'
    target = work / 'motion/whole_body/reconstructed_stage' / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    provenance = {'source': str(source), 'used': False}
    if not source.exists():
        target.write_text('#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = "Z"\n)\ndef Xform "support_surfaces" {}\n')
        provenance['kind'] = 'empty_stage_no_upstream_support'
    elif source.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
        lines = source.read_text().splitlines()
        expected = lines[1].split('sha256:')[1]
        size = int(lines[2].split()[1])
        url = f'https://media.githubusercontent.com/media/nvidia-isaac/video_to_data/{PIN}/{source.relative_to(RG.parent)}'
        if not target.exists() or sha256(target) != expected:
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--max-time', '60',
                            url, '-o', str(target)], check=True)
        if sha256(target) != expected or target.stat().st_size != size:
            raise ValueError(f'{target}: Git-LFS SHA-256/size mismatch')
        provenance.update(kind='verified_git_lfs_blob', url=url)
    else:
        copy_file(source, target)
        provenance['kind'] = 'upstream_bytes'
    return {**file_record(target), **provenance}


def parquet_path(work: Path, sequence: str) -> Path:
    return work / f'motion/whole_body/soma/sequence_id={sequence}/robot_name=g1/data.parquet'


def run_logged(argv: list[str], log: Path, env: dict | None = None) -> None:
    print('+ ' + ' '.join(argv), flush=True)
    with log.open('w') as output:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command exited {result.returncode}; see {log}')


def retarget(work: Path, sequence: str) -> None:
    folder = work / 'sequences' / sequence
    dependencies = [*[folder / name for name in ('poses.npy', 'soma_params.npz', 'ground_plane.json',
                                                  'reconstructed_mesh/output_aligned.glb')],
                    ROOT / 'scripts/retarget_tier.sh',
                    RG / 'scripts/retarget/soma_to_g1.py', *sorted((RG / 'source/robotic_grounding/robotic_grounding/retarget').rglob('*.py')),
                    *sorted((RG / 'source/robotic_grounding/robotic_grounding/retarget/configs/g1').glob('*.json'))]
    signature = {str(p): sha256(p) for p in dependencies if p.is_file()}
    receipt = work / 'retarget_receipt.json'
    parquet = parquet_path(work, sequence)
    if receipt.exists():
        old = json.loads(receipt.read_text())
        # Materialized bytes already capture generator changes; only native retarget code is a dependency.
        previous = {p: h for p, h in old['inputs'].items() if p != str(Path(__file__))}
        if previous == signature and parquet.exists() and old['sha256'] == sha256(parquet):
            print(f'reuse {work.relative_to(ROOT)}', flush=True)
            return
    run_logged(['bash', str(ROOT / 'scripts/retarget_tier.sh'), '--sequences', str(work / 'sequences'), str(work), '1'],
               work / 'retarget.log', {**os.environ, 'PYTHON': sys.executable})
    write_json(receipt, {'inputs': signature, 'sha256': sha256(parquet)})


def reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    row = pq.read_table(path, columns=['object_root_position', 'object_root_axis_angle', 'hand_contact_active']).to_pylist()[0]
    poses = np.tile(np.eye(4), (len(row['object_root_position']), 1, 1))
    poses[:, :3, 3] = row['object_root_position']
    poses[:, :3, :3] = Rotation.from_rotvec(row['object_root_axis_angle']).as_matrix()
    return poses, np.asarray(row['hand_contact_active']) > 0.5


def frame_map(source: np.ndarray, target: np.ndarray) -> dict:
    if len(source) != len(target):
        return {'source_frames': len(source), 'parquet_frames': len(target), 'compatible': False}
    r = target[0, :3, :3] @ source[0, :3, :3].T
    t = target[0, :3, 3] - r @ source[0, :3, 3]
    position = float(np.linalg.norm(source[:, :3, 3] @ r.T + t - target[:, :3, 3], axis=1).max())
    angle = float(Rotation.from_matrix(target[:, :3, :3] @ (r @ source[:, :3, :3]).transpose(0, 2, 1)).magnitude().max())
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r, t
    return {'source_frames': len(source), 'parquet_frames': len(target), 'compatible': position < 1e-5 and angle < 1e-5,
            'max_position_error_m': position, 'max_orientation_error_rad': angle, 'source_cv_to_g1': transform.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/screen/tier2')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'data/screen'):
        parser.error('--out must be under data/screen')
    used = sorted(set().union(*(json.loads(p.read_text()).get('seeds', []) for p in (ROOT / 'data/denoise_eval').glob('*config.json'))))
    if args.seed < 0 or args.seed in used:
        parser.error(f'Use a new nonnegative corruption seed; development used {used}')
    folders = sorted(CLEAN.glob('sequence_id=*'))
    if len(folders) != 3:
        parser.error(f'Expected exactly three non-challenge upstream folders; got {len(folders)}')
    out.mkdir(parents=True, exist_ok=True)
    config = {'seed': args.seed, 'development_seeds': used, 'model': 'eval_denoise.corrupt_matched', 'scale': 1.0,
              'centered': False, 'local_rotation': True, 'denoise': DEFAULT_PARAMS.__dict__, 'contacts': CONTACT_PARAMS.__dict__}
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        parser.error('Existing output has a different experiment configuration; choose another --out')
    write_json(config_path, config)
    soma = SOMA(identity_model_type='mhr')
    metrics = load_metrics()
    manifest = {'schema_version': 1, 'config': config, 'seed_reason': 'Seed 10 is the predeclared Experiment 1 screen seed, disjoint from saved denoiser development seeds 0-4; it is not a final confirmation seed.',
                'paths': 'Absolute paths on this host; regenerate on GPU host to bind its asset paths.',
                'metric_space': 'Fresh clean G1 reference frame; all original source frames, no metric alignment; deterministic hoi mesh samples (seed 0).',
                'sequences': []}
    work_dirs, rows = [], []
    for folder in folders:
        name = folder.name.removeprefix('sequence_id=')
        clean = Sequence.load(folder, soma)
        raw = corrupt_matched(clean, np.random.default_rng(args.seed), 1.0, local_rotation=True)
        seq_out = out / name
        entries = {}
        upstream = upstream_stage(out / 'upstream_stages', name)
        clean_work = seq_out / 'clean'
        clean_error = materialize(clean, clean, clean_work / 'sequences' / name, soma)
        retarget(clean_work, name)
        stage = clean_work / 'motion/whole_body/reconstructed_stage' / f'{name}_support.usda'
        fitted = fit_reference(parquet_path(clean_work, name), stage)
        support = {**file_record(stage), 'kind': 'fitted_clean_reference', 'fit': fitted}
        entries['clean'] = {'work_dir': str(clean_work), 'soma_roundtrip_max_joint_error_m': clean_error,
                            'support': support}
        raw_work = seq_out / 'raw_upstream'
        copy_file(stage, raw_work / stage.relative_to(clean_work))
        raw_folder = raw_work / 'sequences' / name
        roundtrip = materialize(clean, raw, raw_folder, soma)
        raw_loaded = Sequence.load(raw_folder, soma)
        filtered, _, diagnostic = denoise(raw_loaded)
        targets = {'raw_upstream': raw, 'denoised_upstream': filtered}
        for arm, target in targets.items():
            work = seq_out / arm
            target_stage = work / stage.relative_to(clean_work)
            copy_file(stage, target_stage)
            error = roundtrip if arm == 'raw_upstream' else materialize(clean, target, work / 'sequences' / name, soma)
            entries[arm] = {'work_dir': str(work), 'soma_roundtrip_max_joint_error_m': error,
                            'support': {**support, **file_record(target_stage)}}
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda arm: retarget(seq_out / arm, name), targets))
        for arm, base in [('raw_repaired', 'raw_upstream'), ('denoised_repaired', 'denoised_upstream')]:
            work, source_work = seq_out / arm, seq_out / base
            for p in (source_work / 'sequences').rglob('*'):
                if p.is_file():
                    copy_file(p, work / p.relative_to(source_work))
            source_partition = parquet_path(source_work, name).parent.parent
            for p in source_partition.rglob('*'):
                if p.is_file():
                    copy_file(p, work / p.relative_to(source_work))
            seq = Sequence.load(work / 'sequences' / name, soma)
            labels = robust_labels(seq.hand_gaps(), seq.object_moving(), CONTACT_PARAMS, seq.foot_gaps())
            rewrite_reference(parquet_path(work, name), labels)
            target_stage = work / stage.relative_to(clean_work)
            copy_file(stage, target_stage)
            entries[arm] = {'work_dir': str(work), 'support': {**support, **file_record(target_stage)},
                            'soma_roundtrip_max_joint_error_m': entries[base]['soma_roundtrip_max_joint_error_m']}
        scoring = parquet_path(seq_out / 'clean', name)
        clean_poses, truth = reference(scoring)
        original = folder / 'robot_name=g1/data.parquet'
        original_poses, _ = reference(original)
        sequence_record = {'sequence': name, 'frames': len(clean_poses), 'seed': args.seed,
                           'clean_scoring_reference': file_record(scoring),
                           'upstream_support_not_used': upstream,
                           'upstream_original': {**file_record(original), 'used': False,
                                                 'mismatch': frame_map(clean.object_poses_cv, original_poses)},
                           'denoise_diagnostic': diagnostic, 'arms': {}}
        metric_truth = metric_frames(replace(clean, object_poses_cv=clean_poses))
        for arm in ARMS:
            work = seq_out / arm
            parquet = parquet_path(work, name)
            poses, labels = reference(parquet)
            if poses.shape != clean_poses.shape or labels.shape != truth.shape:
                raise ValueError(f'{parquet}: frame/contact grid differs from fresh clean reference')
            mapping = frame_map(np.load(work / 'sequences' / name / 'poses.npy'), poses)
            if not mapping['compatible']:
                raise ValueError(f'{parquet}: object poses do not match source through one rigid transform')
            values = score(metrics, poses, metric_truth)
            values.update(position_mean_cm=float(np.linalg.norm(poses[:, :3, 3] - clean_poses[:, :3, 3], axis=1).mean() * 100),
                          orientation_mean_rad=float(Rotation.from_matrix(poses[:, :3, :3] @ clean_poses[:, :3, :3].transpose(0, 2, 1)).magnitude().mean()),
                          f1_left=f1(labels[0], truth[0]), f1_right=f1(labels[1], truth[1]))
            values['f1_mean'] = (values['f1_left'] + values['f1_right']) / 2
            record = {**entries[arm], 'training_reference': file_record(parquet), 'clean_scoring_reference': file_record(scoring),
                      'mesh': file_record(parquet.parent.parent / 'object/textured_mesh.obj'), 'seed': args.seed,
                      'frame_map': mapping, 'metrics': values,
                      'files': {str(p.relative_to(work)): sha256(p) for p in sorted(work.rglob('*')) if p.is_file() and p.suffix not in ('.log',) and p.name not in ('retarget_receipt.json', 'summary.txt')}}
            sequence_record['arms'][arm] = record
            work_dirs.append(str(work))
            rows.append({'sequence': name, 'arm': arm, **values})
        for base, repaired in [('raw_upstream', 'raw_repaired'), ('denoised_upstream', 'denoised_repaired')]:
            a, b = [pq.read_table(parquet_path(seq_out / arm, name)).drop(['hand_contact_active']) for arm in (base, repaired)]
            if not a.equals(b):
                raise ValueError(f'{name}: repair changed fields other than contact labels')
        manifest['sequences'].append(sequence_record)
    flash_python = ROOT / '.venv-flash/bin/python'
    if not flash_python.is_file():
        flash_python = ROOT / '.venv-flash-cpu/bin/python'
    env = {**os.environ, 'PYTHONPATH': str(FLASH / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')}
    run_logged([str(flash_python), str(ROOT / 'scripts/check_references.py'), *work_dirs], out / 'loader_check.txt', env)
    # Match train_queue.sh's support resolver and USD test in its own environment.
    support_check = """import sys, json, contextlib, io
import newton
from flash_chord.scene.support import add_support_surfaces
from flash_chord.assets.registry import support_usda_for_parquet
from pxr import Usd
values = {}
for parquet in sys.argv[1:]:
    path = support_usda_for_parquet(parquet, robot_name="g1")
    if path is None:
        raise FileNotFoundError(parquet)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(path)
    count = sum(p.GetTypeName() in ("Cylinder", "Cube") for p in stage.Traverse())
    values[parquet] = count > 0
    if count:
        with contextlib.redirect_stdout(io.StringIO()):
            binding = add_support_surfaces(newton.ModelBuilder(), path)
        assert len(binding.shapes) == count
print(json.dumps(values))
"""
    run_logged([str(flash_python), '-c', support_check,
                *[str(parquet_path(Path(w), Path(w).parent.name)) for w in work_dirs]], out / 'support_check.json', env)
    support_values = json.loads((out / 'support_check.json').read_text())
    for sequence in manifest['sequences']:
        for arm in sequence['arms'].values():
            arm['support']['enabled'] = support_values[arm['training_reference']['path']]
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'reference_metrics.csv', index=False)
    f1_table = frame[['sequence', 'arm', 'f1_left', 'f1_right', 'f1_mean']].to_string(index=False, float_format=lambda x: f'{x:.6f}')
    error_table = frame[['sequence', 'arm', 'AUC', 'SP-SR', 'MP-SR', 'MPPE', 'position_mean_cm', 'orientation_mean_rad']].to_string(index=False, float_format=lambda x: f'{x:.6f}')
    report = f'CONTACT F1 (all hands, empty/empty=1)\n{f1_table}\n\nREFERENCE OBJECT ERROR (G1 frame; MPPE cm)\n{error_table}\n\n' + (out / 'loader_check.txt').read_text()
    (out / 'verification.txt').write_text(report)
    manifest['verification'] = {'loader': file_record(out / 'loader_check.txt'), 'support': file_record(out / 'support_check.json'),
                                'metrics': file_record(out / 'reference_metrics.csv'), 'report': file_record(out / 'verification.txt')}
    write_json(out / 'manifest.json', manifest)
    print(report)
    print(f'Manifest: {out / "manifest.json"}')


if __name__ == '__main__':
    main()
