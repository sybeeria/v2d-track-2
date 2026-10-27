"""Freeze explicit upload files, video-only prompts, native jobs and bridge inputs."""
import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch import atomic, digest

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path('/workspace/v2d-track-2')
SCENE = Path('data/challenge/track_2/tier_2_synthetic_noise')
PROMPTS = Path('data/tier3_bridge/prompts')
GATED = ['sam-3d-body-dinov3/model.ckpt', 'sam-3d-body-dinov3/assets/mhr_model.pt', 'sam-3d-body-dinov3/model_config.yaml']


def validate_prompts(path, width, height, frames):
    prompts = json.loads(path.read_text())['prompts']
    if len(prompts) != 2 or any(type(x['object_id']) is not int for x in prompts) or {x['object_id'] for x in prompts} != {0, 1}:
        raise ValueError('exactly human=0 and object=1 required')
    for v in prompts:
        f = v['frame_index']
        if type(f) is not int or not 0 <= f < frames or v.get('role') != ('human', 'object')[v['object_id']]:
            raise ValueError('invalid prompt frame/role')
        points, box, labels = v.get('points', []), v.get('box'), v.get('point_labels', [])
        if bool(points) == bool(box) or len(points) != len(labels) or any(type(x) is not int or x not in (0, 1) for x in labels):
            raise ValueError('one prompt modality and matching binary labels required')
        if box:
            vals = [box[k] for k in ('x0', 'y0', 'x1', 'y1')]
            if not all(type(x) in (int, float) and math.isfinite(x) for x in vals) or not (0 <= vals[0] < vals[2] <= width and 0 <= vals[1] < vals[3] <= height):
                raise ValueError('invalid prompt box')
        for point in points:
            x, y = point['x'], point['y']
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                raise ValueError('invalid prompt point')


def allowed_upload(rel):
    return (rel.parts[:4] == ('external', 'video_to_data', 'reconstruction', 'modules')
            or (rel.parts[:len(SCENE.parts)] == SCENE.parts and len(rel.parts) > len(SCENE.parts) and rel.parts[len(SCENE.parts)] in ('videos', 'mesh'))
            or rel.parts[:3] == PROMPTS.parts
            or (len(rel.parts) == 5 and rel.parts[:3] == ('data', 'brainstorm6', 'T3-EVIDENCE') and re.fullmatch(r'ep[0-2][0-9]', rel.parts[3]) and rel.parts[4] == 'prompt.json')
            or rel.as_posix() in ['data/tier3_weights/' + x for x in GATED]) and not any(
                x.startswith('.') or x in ('__pycache__', 'build', 'token', 'tokens') for x in rel.parts)


def verify(run):
    record = json.loads((run / 'payload/upload.json').read_text())
    root = Path(record['local_project'])
    lines = (run / 'upload.list').read_text().splitlines()
    if lines != list(record['files']):
        raise ValueError('upload list/receipt mismatch')
    for name, sha in record['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not allowed_upload(rel):
            raise ValueError('upload outside allowlist: ' + name)
        path = root / rel
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != sha:
            raise ValueError('changed/escaped upload file: ' + name)
    return record


def prepare(run, include18, prompt_review=None, restore_from=None, episode_ids=None):
    if run.exists():
        raise FileExistsError('choose a fresh run directory')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run.name):
        raise ValueError('run basename must be shell-safe')
    inventory = json.loads((ROOT / PROMPTS / 'video_manifest.json').read_text())
    if len(inventory) != 30 or {v['episode'] for v in inventory} != set(range(30)):
        raise ValueError('expected exactly all 30 video records')
    if episode_ids is not None:
        available = {f"ep{v['episode']:02d}" for v in inventory if include18 or v['episode'] != 18}
        if not episode_ids or len(set(episode_ids)) != len(episode_ids) or set(episode_ids) - available:
            raise ValueError('unknown/duplicate episode ids; use ep00..ep29 (ep18 needs --include18)')
        if restore_from:
            raise ValueError('--episodes requires fresh roots; do not use --restore-from')
    # Project only asset locators. Task/caption text is never propagated or used.
    metadata = {v['episode_index']: {k: v[k] for k in ('sequence_id', 'mesh', 'ground_plane', 'support_surface')}
                for v in map(json.loads, (ROOT / SCENE / 'meta/episodes_metadata.jsonl').read_text().splitlines())}
    reviewed = {}
    if prompt_review:
        review = json.loads(prompt_review.read_text())
        if len(review) != 30 or {v['episode'] for v in review} != set(range(30)):
            raise ValueError('review must cover exactly 30 prompts')
        for item in review:
            if item['final_status'] != 'VISUAL_SEED_REVIEWED; SAM2_PROPAGATION_UNTESTED':
                raise ValueError('prompt not visually reviewed')
            final = (prompt_review.parent / item['final_prompt']).resolve()
            reviewed[item['episode']] = final.relative_to(ROOT)
    else:
        # Existing syntax inventory is useful, but known visual corrections must not be ignored.
        known = ROOT / 'data/brainstorm6/T3-EVIDENCE/prompt_review.json'
        if known.exists() and any(v['changed'] for v in json.loads(known.read_text())):
            raise ValueError('known prompt corrections exist; pass --prompt-review explicitly')
    files, episodes, bridge, sanity = {}, [], [], []
    def upload(rel):
        if not allowed_upload(rel):
            raise ValueError('upload outside allowlist: ' + str(rel))
        files[rel.as_posix()] = digest(ROOT / rel)
        return str(REMOTE / rel)
    for v in sorted(inventory, key=lambda x: x['episode']):
        if episode_ids is not None and f"ep{v['episode']:02d}" not in episode_ids:
            continue
        ep = v['episode']; m = metadata[ep]
        video = SCENE / f'videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4'
        prompts = reviewed.get(ep, PROMPTS / f'ep{ep:02d}.json')
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(ROOT / video)]))['streams'][0]
        count, width, height = int(info['nb_frames']), int(info['width']), int(info['height'])
        if count != v['frames'] or width != v['width'] or height != v['height'] or digest(ROOT / video) != v['video_sha256']:
            raise ValueError('video differs from video-only inventory')
        validate_prompts(ROOT / prompts, width, height, count)
        sanity.append({'episode': ep, 'frames': count, 'width': width, 'height': height, 'status': 'SCHEMA_PASS', 'visual_status': 'T3-EVIDENCE reviewed seed; propagation untested' if prompt_review else 'not checked here', 'prompts_sha256': digest(ROOT / prompts), 'video_sha256': digest(ROOT / video)})
        row = {'episode': ep, 'sequence': f'ep{ep:02d}', 'frames': count}
        for key, rel in [('video', video), ('prompts', prompts), ('mesh', SCENE / m['mesh'])]:
            row[key] = upload(rel); row[key + '_sha256'] = files[rel.as_posix()]
        item = {'episode': ep, 'sequence': m['sequence_id'], 'native_sequence': row['sequence'], 'frames': count,
                'video': str((ROOT / video).resolve()), 'video_sha256': row['video_sha256'],
                'prompts': str((ROOT / prompts).resolve()), 'prompts_sha256': row['prompts_sha256'],
                'mesh': str((ROOT / SCENE / m['mesh']).resolve()), 'mesh_sha256': row['mesh_sha256'],
                'ground_plane': str((ROOT / SCENE / m['ground_plane']).resolve()),
                'support_surface': str((ROOT / SCENE / m['support_surface']).resolve())}
        for key in ('ground_plane', 'support_surface'):
            item[key + '_sha256'] = digest(item[key])
        bridge.append(item)
        if include18 or ep != 18:
            episodes.append(row)
    for name in GATED:
        upload(Path('data/tier3_weights') / name)
    modules = Path('external/video_to_data/reconstruction/modules')
    for path in sorted((ROOT / modules).rglob('*')):
        rel = path.relative_to(ROOT)
        if path.is_file() and allowed_upload(rel) and path.suffix not in ('.pyc', '.so', '.o', '.a'):
            upload(rel)
    run.mkdir(parents=True)
    payload = run / 'payload'; payload.mkdir()
    atomic(payload / 'plan.json', {'schema': 1, 'engineering_only': True, 'episodes': episodes})
    atomic(run / 'bridge_plan.json', {'schema': 1, 'engineering_only': True, 'episodes': bridge})
    atomic(run / 'prompt_sanity.json', sanity)
    atomic(payload / 'upload.json', {'local_project': str(ROOT), 'files': dict(sorted(files.items()))})
    (run / 'upload.list').write_text('\n'.join(sorted(files)) + '\n')
    (run / 'entry.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
P=/workspace/v2d-track-2
R=$P/runs/$VAST_RUN_NAME
export NATIVE_PYTHON=python PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$R/build/weights/hf_home
python -I -c 'import torch,shutil,sys; from pathlib import Path; assert torch.__version__.startswith("2.5.1"); assert torch.version.cuda == "12.4"; assert torch.cuda.is_available(); assert all(8 <= torch.cuda.get_device_capability(i)[0] < 10 and torch.cuda.get_device_properties(i).total_memory >= 24000*1024**2 for i in range(torch.cuda.device_count())); root=Path(sys.argv[1])/"episodes"; restored=sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink() and not any(part.startswith(".") for part in p.relative_to(root).parts)); assert shutil.disk_usage("/workspace").free + restored >= 200*1024**3' "$R"
mkdir -p "$R/build/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/assets"
cp "$P/data/tier3_weights/sam-3d-body-dinov3/model.ckpt" "$R/build/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/"
cp "$P/data/tier3_weights/sam-3d-body-dinov3/model_config.yaml" "$R/build/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/"
cp "$P/data/tier3_weights/sam-3d-body-dinov3/assets/mhr_model.pt" "$R/build/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/"
# Native setup is paid once for the box. Its phase receipts resume identical setup.
bash "$P/scripts/tier3_native/setup.sh" --project "$P" --work-root "$R/build" --weights "$R/build/weights" all
python -I "$P/scripts/tier3_native/batch.py" --project "$P" --root "$R/episodes" --weights "$R/build/weights" --plan "$R/control/plan.json" --session "$R/control/session.json" --canary
''')
    verify(run)
    if restore_from:
        previous = restore_from.resolve()
        if previous.name != run.name or not (previous / 'STOPPED').exists() or not (previous / 'QUIESCED').exists() or (previous / 'BACKUP_FAILED').exists():
            raise ValueError('restore needs same run basename, stopped writer and complete backup')
        if digest(previous / 'payload/plan.json') != digest(payload / 'plan.json'):
            raise ValueError('restore plan changed; retain previous attempt and use a fresh native root')
        # rsync uses temp+rename (never --inplace), so unchanged backup bytes can
        # be hardlinked without duplicating ~121 GiB on the Mac.
        target = run / 'pulled' / run.name
        target.parent.mkdir()
        shutil.copytree(previous / 'pulled' / previous.name, target, copy_function=os.link, symlinks=True, ignore=shutil.ignore_patterns('.rsync-partial', 'control'))
    return {'episodes_to_reconstruct': len(episodes), 'prompt_pairs': len(sanity), 'upload_bytes': sum((ROOT / name).stat().st_size for name in files),
            'weights_uploaded_bytes': sum((ROOT / 'data/tier3_weights' / name).stat().st_size for name in GATED)}


def main():
    global ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'verify-upload'))
    p.add_argument('run', type=Path)
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--include18', action='store_true')
    p.add_argument('--episodes', nargs='+', help='explicit ep00..ep29 subset, fresh roots only; ep18 needs --include18')
    p.add_argument('--prompt-review', type=Path)
    p.add_argument('--restore-from', type=Path)
    a = p.parse_args()
    ROOT = a.project.resolve()
    print(json.dumps(prepare(a.run.resolve(), a.include18, a.prompt_review.resolve() if a.prompt_review else None, a.restore_from, a.episodes) if a.action == 'prepare' else {'verified_files': len(verify(a.run)['files'])}, indent=2))


if __name__ == '__main__':
    main()
