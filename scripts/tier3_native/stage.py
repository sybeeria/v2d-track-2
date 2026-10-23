"""Run an owned stage; every declared output must appear explicitly in command args."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def atomic(path, value):
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def fingerprint(paths):
    result = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        for p in files:
            if p.is_file():
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(block)
                result.append([str(p), p.stat().st_size, h.hexdigest()])
    return result


def remove_owned(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--input', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, action='append', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    root = a.root.resolve()
    outputs = [x.parent.resolve() / x.name for x in a.output]
    if not a.name.replace('_', '').isalnum() or len(set(outputs)) != len(outputs) or any(root not in x.parents for x in outputs):
        raise ValueError('stage outputs must be unique and strictly below owned root')
    if any(x in y.parents for x in outputs for y in outputs if x != y):
        raise ValueError('overlapping outputs')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if any(not any(str(raw.absolute()) in arg or str(final) in arg for arg in command) for raw, final in zip(a.output, outputs)):
        raise ValueError('each declared output must appear explicitly in command arguments')
    code = [Path(arg).resolve() for arg in command if Path(arg).suffix in ('.py', '.sh') and Path(arg).is_file()]
    identity = {'command': command, 'inputs': fingerprint(list(dict.fromkeys([*a.input, *code]))), 'outputs': list(map(str, outputs))}
    state = root / '.stages'; state.mkdir(parents=True, exist_ok=True)
    with (state / (a.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = state / (a.name + '.done.json')
        started = state / (a.name + '.started.json')
        staging = root / '.staging' / a.name
        if done.exists():
            saved = json.loads(done.read_text())
            if saved['identity'] != identity or saved['artifacts'] != fingerprint(outputs):
                raise ValueError('completed stage inputs/outputs changed; choose a new episode root')
            print('SKIP', a.name); return
        if started.exists():
            if json.loads(started.read_text()) != identity:
                raise ValueError('interrupted stage inputs changed; choose a new episode root')
            for path in [*outputs, staging]:
                remove_owned(path)
        elif staging.exists() or any(x.exists() or x.is_symlink() for x in outputs):
            raise FileExistsError('refusing to overwrite output without ownership record')
        atomic(started, identity)
        staging.mkdir(parents=True)
        replacements = [(str(final), str(staging / str(i) / final.name)) for i, final in enumerate(outputs)]
        for _, temporary in replacements:
            Path(temporary).parent.mkdir(parents=True)
        cmd = command.copy()
        for i, arg in enumerate(cmd):
            for raw, (final, temporary) in zip(a.output, replacements):
                arg = arg.replace(str(raw.absolute()), temporary).replace(final, temporary)
            cmd[i] = arg
        logs = root / 'logs'; logs.mkdir(exist_ok=True)
        log = logs / f'{a.name}.{time.time_ns()}.log'
        start = time.monotonic(); peak = None; proc = None; rc = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            with log.open('w') as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                next_sample = 0
                while proc.poll() is None:
                    if time.monotonic() >= next_sample and os.environ.get('CUDA_VISIBLE_DEVICES'):
                        try:
                            sample = subprocess.check_output(['nvidia-smi', '--id=' + os.environ['CUDA_VISIBLE_DEVICES'], '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=2)
                            memory = float(sample.strip())
                            peak = memory if peak is None else max(peak, memory)
                        except (OSError, subprocess.SubprocessError, ValueError):
                            pass
                        next_sample = time.monotonic() + 1
                    time.sleep(.05)
                rc = proc.returncode
        except KeyboardInterrupt:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
            rc = 130
        finally:
            atomic(log.with_suffix('.metrics.json'), {'stage': a.name, 'wall_seconds': time.monotonic() - start,
                   'exit_code': rc, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'peak_gpu_memory_mib_sampled': peak, 'gpu_memory_scope': 'device-wide ~1s samples; may miss peaks',
                   'log': log.name, 'error_tail': log.read_text(errors='replace')[-8192:] if rc else ''})
        if rc:
            print(log.read_text(errors='replace')[-8192:], file=sys.stderr)
            raise SystemExit(rc)
        temps = [Path(x) for _, x in replacements]
        if not fingerprint(temps) or any(not x.exists() for x in temps):
            raise ValueError('stage did not produce its declared artifacts')
        # Wild-export JSON records absolute mesh paths. Rebase only paths of this
        # stage's own published artifacts, preserving provenance input locators.
        def rebase(value):
            if isinstance(value, str):
                for final, temporary in replacements:
                    value = value.replace(temporary, final)
                return value
            if isinstance(value, dict):
                return {k: rebase(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rebase(v) for v in value]
            return value
        for temporary in temps:
            for item in temporary.rglob('*.json') if temporary.is_dir() else []:
                data = json.loads(item.read_text())
                atomic(item, rebase(data))
        for final, temporary in replacements:
            Path(final).parent.mkdir(parents=True, exist_ok=True)
            Path(temporary).replace(final)
        atomic(done, {'identity': identity, 'artifacts': fingerprint(outputs)})
        started.unlink(); shutil.rmtree(staging)
        print('DONE', a.name)


if __name__ == '__main__':
    main()
