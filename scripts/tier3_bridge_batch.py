"""Sequential Mac bridge; publish a self-scoring manifest only for frozen-gate PASS."""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/tier3_native'))
from batch import atomic, digest
from stage import fingerprint, remove_owned
from bounded import run, ResourceHold
from tier3_rest_repair import preflight

PROTOCOL_SHA256 = 'eec81f1beac187d63cca82ca6b098999be62dff56e7fe1be64c9f874d3af41da'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'native-root', 'out', 'tier1-root', 'protocol'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--python', type=Path, default=ROOT / '.venv/bin/python')
    p.add_argument('--python-deps', type=Path, default=ROOT / 'data/tier3_bridge/deps')
    p.add_argument('--ep18-cache', type=Path, help='explicit existing decoded/retargeted cache; never claims fresh decoding')
    p.add_argument('--existing-ep18', type=Path, default=ROOT / 'data/tier3_native/ep18')
    p.add_argument('--rgb-dir', type=Path, help='epNN.json independent video-only outlines, dimensions and video_sha256')
    rest = p.add_mutually_exclusive_group()
    rest.add_argument('--rest-evidence-dir', type=Path, help='video-only epNN/rgb_evidence.json; preflight HOLD keeps generic repair')
    rest.add_argument('--rest-evidence', type=Path, help='single selected episode evidence, including the legacy schema')
    p.add_argument('--legacy-rest-video-sha256', help='explicit video binding for legacy evidence without video metadata')
    p.add_argument('--episodes', type=int, nargs='+')
    # Fresh ep18 (858 frames) on the 16 GB M1: export 7 min at 5.5 GB peak RSS; retarget 2.3 min at 6.9 GB sampled.
    p.add_argument('--max-rss-bytes', type=int, default=8_000_000_000)
    p.add_argument('--stage-timeout-seconds', type=int, default=4 * 3600)
    p.add_argument('--released-scene-assets-engineering-only', action='store_true', required=True)
    a = p.parse_args()
    if a.rest_evidence and (not a.episodes or len(a.episodes) != 1):
        p.error('--rest-evidence requires exactly one --episodes selection')
    if a.legacy_rest_video_sha256 and (not a.rest_evidence or not re.fullmatch(r'[a-f0-9]{64}', a.legacy_rest_video_sha256)):
        p.error('--legacy-rest-video-sha256 requires --rest-evidence and a SHA256')
    if a.ep18_cache and (a.rest_evidence or a.rest_evidence_dir):
        p.error('video rest repair requires fresh decoding/retargeting; omit --ep18-cache')
    if digest(a.protocol) != PROTOCOL_SHA256:
        raise ValueError('frozen protocol changed; review gates before running')
    rows = json.loads(a.plan.read_text())['episodes']
    if len({r['episode'] for r in rows}) != len(rows) or len({r['sequence'] for r in rows}) != len(rows):
        raise ValueError('duplicate bridge episode/sequence')
    for row in rows:
        if type(row['episode']) is not int or not 0 <= row['episode'] < 30 or type(row['frames']) is not int or row['frames'] < 15:
            raise ValueError('invalid episode/frame count')
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', row[key]) for key in ('sequence', 'native_sequence')):
            raise ValueError('invalid sequence path component')
        for key in ('video', 'mesh', 'ground_plane', 'support_surface', 'prompts'):
            if not Path(row[key]).is_absolute() or not re.fullmatch(r'[a-f0-9]{64}', row[key + '_sha256']):
                raise ValueError('bridge inputs must be absolute and hash-bound')
    if a.episodes:
        if not set(a.episodes) <= {r['episode'] for r in rows}:
            raise ValueError('episode absent from frozen plan')
        rows = [r for r in rows if r['episode'] in a.episodes]
    a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / '.bridge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Do not leave a previous success manifest visible while revalidating inputs.
        manifest = a.out / 'manifest.json'
        if manifest.exists():
            manifest.replace(a.out / 'manifest.previous.json')
        results, accepted = [], []
        for row in rows:
            ep = row['episode']; sequence = row['sequence']; name = row['native_sequence']
            root = a.out / f'ep{ep:02d}'; root.mkdir(exist_ok=True)
            result = {'episode': ep, 'sequence': sequence, 'status': 'FAIL', 'reasons': [], 'stages': []}
            evidence_path = a.rest_evidence or (a.rest_evidence_dir / f'ep{ep:02d}/rgb_evidence.json' if a.rest_evidence_dir else None)
            rest_binding = {'path': str(evidence_path.resolve()) if evidence_path else None,
                            'sha256': digest(evidence_path) if evidence_path and evidence_path.is_file() else None,
                            'legacy_video_sha256': a.legacy_rest_video_sha256}
            result['rest_repair'] = {'status': 'SKIPPED', 'reasons': ['video rest evidence not checked' if evidence_path else 'video rest evidence not configured'], **rest_binding}
            binding = {'plan_row': row, 'protocol_sha256': PROTOCOL_SHA256, 'driver_sha256': digest(Path(__file__)),
                       'code_sha256': {x.name: digest(x) for x in (ROOT / 'scripts').glob('tier3*.py')},
                       'weights': {str(x.relative_to(ROOT / 'data/tier3_weights')): digest(x) for x in (ROOT / 'data/tier3_weights/sam-3d-body-dinov3').rglob('*') if x.is_file()},
                       'rest_evidence': rest_binding,
                       'rgb_sha256': digest(a.rgb_dir / f'ep{ep:02d}.json') if a.rgb_dir and (a.rgb_dir / f'ep{ep:02d}.json').exists() else None}
            if ep == 18 and a.ep18_cache:
                cache = a.ep18_cache.resolve()
                raw = cache / 'raw/ep18'
                if not raw.exists(): raw = cache.parent / 'raw/ep18'
                sources = [raw, cache / 'sequences', cache / 'motion']
                binding['decoded_cache'] = {'path': str(cache), 'files': {str(x): digest(x) for origin in sources
                    for x in sorted(origin.rglob('*')) if x.is_file() and x.suffix != '.pyc'}}
            def execute(stage, args, outputs=None):
                outputs = [Path(x) for x in (outputs or [])]
                marker = root / ('.' + stage + '.done.json')
                started = root / ('.' + stage + '.started.json')
                identity = {'inputs': binding, 'command': args}
                if outputs:
                    if any(root not in x.parents for x in outputs):
                        raise ValueError('bridge outputs outside owned episode root')
                    if marker.exists():
                        saved = json.loads(marker.read_text())
                        if saved['identity'] != identity or saved['artifacts'] != fingerprint(outputs):
                            raise ValueError('completed bridge stage changed: ' + stage)
                        return
                    if started.exists():
                        if json.loads(started.read_text()) != identity:
                            raise ValueError('interrupted bridge stage changed: ' + stage)
                        for x in outputs: remove_owned(x)
                    elif any(x.exists() for x in outputs):
                        raise ValueError('unowned partial bridge output: ' + stage)
                    atomic(started, identity)
                log = root / (stage + '.log')
                info = run(args, log, max_bytes=a.max_rss_bytes, timeout=a.stage_timeout_seconds, env={'WARP_CACHE_PATH': str(root / 'warp_cache'), 'TMPDIR': str(root / 'tmp')})
                result['stages'].append({'stage': stage, **info})
                if outputs:
                    artifacts = fingerprint(outputs)
                    if not artifacts or any(not x.exists() for x in outputs): raise ValueError('missing stage output')
                    atomic(marker, {'identity': identity, 'artifacts': artifacts}); started.unlink()
            def python(script, *args):
                return [str(a.python.absolute()), '-B', str(ROOT / 'scripts' / script), *map(str, args)]
            try:
                (root / 'tmp').mkdir(exist_ok=True)
                old = root / 'inputs.json'
                if old.exists() and json.loads(old.read_text()) != binding:
                    raise ValueError('bridge input/code binding changed; choose a fresh output root')
                for key in ('video', 'mesh', 'ground_plane', 'support_surface', 'prompts'):
                    if digest(row[key]) != row[key + '_sha256']:
                        raise ValueError('changed bridge input: ' + key)
                atomic(old, binding)
                rest_allowed = False
                if evidence_path:
                    if not evidence_path.is_file():
                        result['rest_repair']['reasons'] = ['video rest evidence missing']
                    else:
                        _, holds = preflight(json.loads(evidence_path.read_text()), episode=ep, frames=row['frames'],
                            video_sha256=row['video_sha256'], legacy_video_sha256=a.legacy_rest_video_sha256)
                        rest_allowed = not holds
                        result['rest_repair'].update(status='READY' if rest_allowed else 'SKIPPED', reasons=holds)
                native = a.native_root / name
                if ep == 18 and not native.exists():
                    native = a.existing_ep18
                receipt = native / 'result.json'
                if receipt.exists():
                    record = json.loads(receipt.read_text())
                    if record['status'] != 'SUCCESS' or digest(native / 'initializers.pth') != record['initializers_sha256']:
                        raise ValueError('native success/artifact receipt failed')
                    for key in ('video', 'mesh', 'prompts'):
                        if record['inputs'][key + '_sha256'] != row[key + '_sha256']:
                            raise ValueError('native/bridge input binding differs: ' + key)
                elif ep == 18:
                    if json.loads((native / 'prompts.json').read_text()) != json.loads(Path(row['prompts']).read_text()):
                        raise ValueError('legacy ep18 prompt semantics differ')
                    result['legacy_prompts_sha256'] = digest(native / 'prompts.json')
                    legacy = json.loads((native.parent / 'input_manifest.json').read_text())
                    for key, suffix in [('video', 'episode_000018.mp4'), ('mesh', '.glb')]:
                        if not any(v['path'].endswith(suffix) and v['sha256'] == row[key + '_sha256'] for v in legacy):
                            raise ValueError('legacy ep18 input receipt differs: ' + key)
                else:
                    raise ValueError('pulled native success receipt missing')
                output = native / 'output' / name
                if not output.exists():
                    output = native
                human, obj, camera = output / f'mhr_init/{name}.pkl', output / f'foundationpose/{name}.pkl', output / f'export/{name}/wild_export.json'
                if receipt.exists():
                    wanted = {str(x): digest(x) for x in (human, obj, camera)}
                    found = set()
                    for marker in (native / '.stages').glob('*.done.json'):
                        for stored, size, sha in json.loads(marker.read_text())['artifacts']:
                            for local, actual in wanted.items():
                                suffix = '/' + str(Path(local).relative_to(native))
                                if stored.endswith(suffix):
                                    if sha != actual: raise ValueError('pulled artifact hash differs from native stage receipt')
                                    found.add(local)
                    if found != set(wanted): raise ValueError('incomplete native stage artifact receipts')
                (root / 'adapter').mkdir(exist_ok=True)
                bundle = root / 'adapter/initializers.pth'; mesh = root / 'adapter/mesh.glb'
                # Reuse the native stage engine for atomic, hash-bound adapter output.
                adapter = python('tier3_initializers.py', '--human', human, '--object', obj, '--wild-export', camera,
                                 '--expected-frames', row['frames'], '--out', bundle,
                                 '--released-mesh-engineering-only', row['mesh'], '--mesh-out', mesh)
                wrapped = [str(a.python.absolute()), '-B', str(ROOT / 'scripts/tier3_native/stage.py'), '--root', str(root), '--name', 'adapter',
                           '--input', str(old), '--input', str(human), '--input', str(obj), '--input', str(camera), '--input', row['mesh'],
                           '--output', str(bundle), '--output', str(mesh), '--', *adapter]
                execute('adapter', wrapped)
                cache = a.ep18_cache if ep == 18 else None
                if cache:
                    # This is a cache/provenance replay, deliberately explicit in the result.
                    cache = cache.resolve()
                    raw_cache = cache / 'raw/ep18'
                    if not raw_cache.exists(): raw_cache = cache.parent / 'raw/ep18'
                    report = json.loads((raw_cache / 'tier3_export.json').read_text())
                    expected_sources = {digest(x) for x in (human, obj, camera)}
                    if set(report['source_metadata']['sources'].values()) != expected_sources or report['source_metadata']['ground_truth_used'] is not False:
                        raise ValueError('cached SOMA source hashes differ from pulled native output')
                    if json.loads((cache / 'sequences' / sequence / 'tier3_alignment.json').read_text())['reference_frames_used'] != [0]:
                        raise ValueError('cached alignment is not frame-0-only')
                    for rel in ('ground_plane.json', 'support_surface.usda'):
                        file = cache / 'sequences' / sequence / rel
                        key = 'ground_plane' if rel == 'ground_plane.json' else 'support_surface'
                        # Retargeter writes the support into motion; handle that closure below.
                        if file.exists() and digest(file) != row[key + '_sha256']:
                            raise ValueError('cached scene asset differs')
                    cache_sources = {'raw': raw_cache.parent, 'sequences': cache / 'sequences'}
                    for rel, origin in cache_sources.items():
                        target = root / rel
                        if not target.exists():
                            temp = root / ('.' + rel + '.tmp'); shutil.rmtree(temp, ignore_errors=True)
                            shutil.copytree(origin, temp); temp.replace(target)
                    for rel, origin in cache_sources.items():
                        for file in origin.rglob('*'):
                            if file.is_file() and digest(file) != digest(root / rel / file.relative_to(origin)):
                                raise ValueError('copied bridge cache changed')
                    result['input_arm'] = 'tier3_repaired' if (cache / 'sequences' / sequence / 'tier3_repair.json').exists() else 'tier3_raw'
                    motion = cache / 'motion'
                    result['mode'] = 'verified existing SOMA/alignment/G1 cache; fresh adapter/materialization/gates only'
                    result['cache'] = binding['decoded_cache']
                else:
                    result['input_arm'] = 'tier3_repaired'
                    result['mode'] = 'fresh export/alignment/retarget'
                    raw = root / 'raw' / name
                    # PoseInversion couples the frames of a batch. Batch 8 (the exporter default) reproduces the qualified
                    # ep18 decode exactly and runs ~5.6x faster on CPU than batch 1, whose poses differ by up to .1 rad.
                    execute('export', python('tier3_export.py', '--bundle', bundle, '--mesh', mesh, '--weights', ROOT / 'data/tier3_weights',
                        '--checkpoint', ROOT / 'data/tier3_weights/sam-3d-body-dinov3/model.ckpt',
                        '--mhr-model', ROOT / 'data/tier3_weights/sam-3d-body-dinov3/assets/mhr_model.pt',
                        '--soma-assets', ROOT / 'external/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma',
                        '--out', raw, '--device', 'cpu', '--batch-size', 8, '--python-deps', a.python_deps), [raw])
                    execute('alignment', python('tier3_align.py', '--source', raw, '--tier1-root', a.tier1_root, '--episode', ep,
                        '--out', root / 'aligned', '--motion-root', root / 'alignment_motion', '--ground-plane', row['ground_plane'],
                        '--support-surface', row['support_surface'], '--released-scene-assets-engineering-only'), [root / 'aligned', root / 'alignment_motion'])
                    repaired = root / ('generic_repair' if rest_allowed else 'sequences')
                    execute('repair', python('tier3_repair.py', '--source', root / 'aligned' / sequence, '--out', repaired / sequence), [repaired])
                    if rest_allowed:
                        rest_args = python('tier3_rest_repair.py', '--source', repaired / sequence,
                            '--camera', camera, '--evidence', evidence_path, '--evidence-sha256', rest_binding['sha256'],
                            '--episode', ep, '--frames', row['frames'], '--video-sha256', row['video_sha256'],
                            '--python-deps', a.python_deps, '--out', root / 'sequences' / sequence)
                        if a.legacy_rest_video_sha256:
                            rest_args += ['--legacy-video-sha256', a.legacy_rest_video_sha256]
                        execute('video_rest_repair', rest_args, [root / 'sequences'])
                        result['rest_repair']['status'] = 'APPLIED'
                        result['input_arm'] = 'tier3_rest_repaired'
                    motion = root / 'retarget/motion'
                    execute('retarget', python('tier3_native/bridge_retarget.py', '--project', ROOT, '--sequences', root / 'sequences',
                        '--alignment-motion', root / 'alignment_motion', '--out', root / 'retarget'), [root / 'retarget'])
                execute('materialize', python('tier3_materialize.py', '--motion-root', motion, '--out', root / 'assets'), [root / 'assets'])
                rgb = a.rgb_dir / f'ep{ep:02d}.json' if a.rgb_dir else None
                if rgb and rgb.exists():
                    if json.loads(rgb.read_text()).get('video_sha256') != row['video_sha256']:
                        raise ValueError('RGB evidence video hash differs')
                else:
                    rgb = None
                gate_args = python('tier3_gates.py', '--base', root, '--sequence', sequence, '--frames', row['frames'], '--camera', camera,
                                   '--protocol', a.protocol, '--out', root / 'gate.json')
                if rgb:
                    gate_args += ['--rgb', str(rgb)]
                execute('gates', gate_args)
                gate = json.loads((root / 'gate.json').read_text())
                result.update(status=gate['status'], reasons=gate['reasons'], inspected=gate.get('inspected', []))
                if gate['status'] == 'PASS':
                    ref = root / f'assets/human_motion_data/whole_body/soma/sequence_id={sequence}/robot_name=g1/data.parquet'
                    reference = {'path': str(ref), 'sha256': digest(ref)}
                    accepted.append({'episode': ep, 'sequence': sequence, 'frames': row['frames'], 'arms': {result['input_arm']: {
                        'training_reference': reference, 'clean_scoring_reference': reference, 'support': {**gate['support'], 'enabled': False}}}})
            except ResourceHold as error:
                result.update(status='HOLD', reasons=['resource limit: ' + str(error)])
            except Exception as error:
                result.update(status='FAIL', reasons=[type(error).__name__ + ': ' + str(error)])
            atomic(root / 'result.json', result); results.append(result)
            atomic(a.out / 'status.json', results)
            lines = ['episode\tsequence\tstatus\treason\trest_repair\trest_reason'] + [f"{r['episode']:02d}\t{r['sequence']}\t{r['status']}\t" +
                '; '.join(r['reasons']).replace('\n', ' ').replace('\t', ' ') + '\t' + r['rest_repair']['status'] + '\t' +
                '; '.join(r['rest_repair']['reasons']).replace('\n', ' ').replace('\t', ' ') for r in results]
            tmp = a.out / '.status.tsv.tmp'; tmp.write_text('\n'.join(lines) + '\n'); tmp.replace(a.out / 'status.tsv')
        if accepted:
            atomic(manifest, {'schema_version': 1, 'tier': 3, 'scoring_mode': 'self', 'sequences': accepted})
        print(json.dumps({'episodes': len(results), 'passing': len(accepted), 'manifest': str(manifest) if accepted else None}))
        raise SystemExit(0 if len(accepted) == len(rows) and rows else 1)


if __name__ == '__main__':
    main()
