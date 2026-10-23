"""Longest-first native reconstruction, one episode per visible GPU; no training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import time
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def tree_digest(root):
    # Runtime caches/build products are not model inputs. Hash all actual weights,
    # configuration and source, once per batch, rather than once per stage/frame.
    suffixes = {'.py', '.sh', '.toml', '.txt', '.yaml', '.yml', '.json', '.pt', '.pth', '.ckpt', '.safetensors'}
    entries = []
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if p.is_file() and not any(x.startswith('.') or x in ('build', '__pycache__') for x in rel.parts) and p.suffix in suffixes:
            entries.append([rel.as_posix(), digest(p)])
    if not entries:
        raise ValueError('empty source/weight input tree: ' + str(root))
    return hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest()


def read_plan(path):
    plan = json.loads(path.read_text())
    rows = plan['episodes']
    ids, names = set(), set()
    for row in rows:
        ep, name, frames = row['episode'], row['sequence'], row['frames']
        if type(ep) is not int or not 0 <= ep < 30 or ep in ids or name in names:
            raise ValueError('invalid/duplicate episode')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or type(frames) is not int or frames < 1:
            raise ValueError('invalid sequence/frame count')
        ids.add(ep); names.add(name)
        for key in ('video', 'prompts', 'mesh'):
            if not Path(row[key]).is_absolute() or not re.fullmatch('[a-f0-9]{64}', row[key + '_sha256']):
                raise ValueError('absolute hash-bound inputs required')
    if not rows:
        raise ValueError('empty plan')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--gpus', default='auto', help='comma-separated GPU UUIDs/indices, or auto')
    p.add_argument('--run-command', type=Path, default=Path(__file__).with_name('run.sh'))
    p.add_argument('--canary', action='store_true', help='wait for hash-bound GO.json after the longest episode')
    p.add_argument('--session', type=Path, help='fresh controller session, excluded from reusable stage input identity')
    p.add_argument('--canary-poll-seconds', type=float, default=30)
    a = p.parse_args()
    if a.canary_poll_seconds <= 0:
        raise ValueError('positive canary poll interval required')
    rows = sorted(read_plan(a.plan), key=lambda r: (-r['frames'], r['episode']))
    a.root = a.root.resolve(); a.root.mkdir(parents=True, exist_ok=True)
    with (a.root / '.batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.gpus == 'auto':
            text = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total,compute_cap', '--format=csv,noheader,nounits'], text=True)
            devices = [line.split(',') for line in text.strip().splitlines()]
            if any(float(mem) < 24000 or not 8 <= float(cc) < 10 for _, mem, cc in devices):
                raise ValueError('native pins require Ampere/Hopper/Ada with >=24000 MiB per GPU; Blackwell unqualified')
            gpus = [uuid.strip() for uuid, _, _ in devices]
            capacities = {uuid.strip(): float(mem) for uuid, mem, _ in devices}
        else:
            gpus = a.gpus.split(',')
            capacities = dict.fromkeys(gpus)
        if not gpus or len(set(gpus)) != len(gpus) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', gpu) for gpu in gpus):
            raise ValueError('invalid/duplicate GPU list')
        source = tree_digest(a.project / 'scripts') + tree_digest(a.project / 'external/video_to_data/reconstruction/modules')
        common = {'weights_sha256': tree_digest(a.weights), 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'plan_sha256': digest(a.plan), 'runner_sha256': digest(a.run_command)}
        session = json.loads(a.session.read_text()) if a.session else {'session_id': uuid.uuid4().hex, 'plan_sha256': common['plan_sha256']}
        if session.get('plan_sha256') != common['plan_sha256'] or not re.fullmatch('[a-f0-9]{32}', session.get('session_id', '')):
            raise ValueError('invalid controller session/plan')
        atomic(a.root / 'batch_inputs.json', common)

        def episode(row, gpu):
            root = a.root / row['sequence']; root.mkdir(exist_ok=True)
            started = time.monotonic()
            result = {'episode': row['episode'], 'sequence': row['sequence'], 'frames': row['frames'], 'gpu': gpu, 'gpu_memory_total_mib': capacities[gpu], 'session_id': session['session_id'],
                      'status': 'FAIL', 'error_tail': '', 'inputs': {**common, **row}, 'stage_metrics': [], 'peak_gpu_memory_mib_sampled': None}
            try:
                for key in ('video', 'prompts', 'mesh'):
                    if digest(row[key]) != row[key + '_sha256']:
                        raise ValueError('changed ' + key)
                binding = result['inputs']
                bound = root / 'inputs.json'
                if bound.exists() and json.loads(bound.read_text()) != binding:
                    raise ValueError('episode inputs changed; retain this attempt and choose a new root')
                atomic(bound, binding)
                env = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACEHUB_API_TOKEN')}
                env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
                cmd = ['bash', str(a.run_command), '--project', str(a.project.resolve()), '--work-root', str(a.root),
                       '--weights', str(a.weights.resolve()), '--sequence', row['sequence'], '--frames', str(row['frames']),
                       '--video', row['video'], '--prompts', row['prompts'], '--mesh', row['mesh'], '--released-mesh-engineering-only']
                log = root / f'episode.{time.time_ns()}.log'
                with log.open('w') as f:
                    code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                result['log'] = log.name
                result['error_tail'] = log.read_text(errors='replace')[-8192:] if code else ''
                result['exit_code'] = code
                if code:
                    raise RuntimeError('native runner exited ' + str(code))
                if not (root / 'initializers.pth').is_file():
                    raise ValueError('runner returned success without initializers')
                result['initializers_sha256'] = digest(root / 'initializers.pth')
                result['status'] = 'SUCCESS'
            except Exception as e:
                result['error'] = type(e).__name__ + ': ' + str(e)
            metrics = sorted((root / 'logs').glob('*.metrics.json'))
            try:
                result['stage_metrics'] = [json.loads(x.read_text()) for x in metrics]
            except (ValueError, OSError) as error:
                result.update(status='FAIL', error='stage metrics unreadable: ' + str(error))
            peaks = [x.get('peak_gpu_memory_mib_sampled') for x in result['stage_metrics'] if x.get('peak_gpu_memory_mib_sampled') is not None]
            result['peak_gpu_memory_mib_sampled'] = max(peaks) if peaks else None
            result['wall_seconds_this_attempt'] = time.monotonic() - started
            atomic(root / 'result.json', result)
            print('GPU ' + gpu + ': ' + row['sequence'] + ' ' + result['status'], flush=True)
            return result['status'] == 'SUCCESS'

        if a.canary:
            row = rows.pop(0)
            prior = a.root / row['sequence']
            archive = a.root / '.canary-prior'
            # A replacement box must measure its own fresh canary. Retain the
            # old attempt, while other completed episodes keep their markers.
            if prior.exists():
                if not (prior / 'inputs.json').exists() or json.loads((prior / 'inputs.json').read_text()) != {**common, **row}:
                    raise ValueError('canary inputs changed; retain this attempt and choose a new root')
                archive.mkdir(exist_ok=True)
                prior.rename(archive / (row['sequence'] + '.' + str(time.time_ns())))
            if (a.root / 'GO.json').exists():
                archive.mkdir(exist_ok=True)
                (a.root / 'GO.json').rename(archive / ('GO.' + str(time.time_ns()) + '.json'))
            if not episode(row, gpus[0]):
                raise SystemExit('canary failed; remaining episodes not scheduled')
            canary = a.root / row['sequence'] / 'result.json'
            expected = {'plan_sha256': common['plan_sha256'], 'canary_sha256': json.loads(canary.read_text())['initializers_sha256'], 'session_id': session['session_id']}
            atomic(a.root / 'CANARY_READY.json', expected)
            while True:
                try:
                    if json.loads((a.root / 'GO.json').read_text()) == expected:
                        break
                except (OSError, ValueError):
                    pass
                print('GPU ' + gpus[0] + ': awaiting first-episode review', flush=True)
                time.sleep(a.canary_poll_seconds)
        pending = queue.Queue()
        for row in rows:
            pending.put(row)

        def worker(gpu):
            ok = True
            while True:
                try:
                    row = pending.get_nowait()
                except queue.Empty:
                    return ok
                ok = episode(row, gpu) and ok

        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            outcomes = list(pool.map(worker, gpus))
        raise SystemExit(0 if all(outcomes) else 1)


if __name__ == '__main__':
    main()
