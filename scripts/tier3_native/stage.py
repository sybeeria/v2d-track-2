"""Run one owned native stage, resume identical successes, retain failed logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def fingerprint(paths):
    result = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        for p in files:
            if p.is_file():
                digest = hashlib.sha256()
                with p.open('rb') as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(block)
                result.append([str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns, digest.hexdigest()])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--input', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, action='append', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    root = a.root.resolve()
    outputs = [x.resolve() for x in a.output]
    if not a.name.replace('_', '').isalnum() or any(root not in x.parents for x in outputs):
        raise ValueError('stage outputs must be strictly below the owned episode root')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    code_inputs = [Path(arg) for arg in command if Path(arg).suffix in ('.py', '.sh') and Path(arg).is_file()]
    identity = {'command': command, 'inputs': fingerprint(list(dict.fromkeys([*a.input, *code_inputs]))),
                'outputs': list(map(str, outputs))}
    state = root / '.stages'
    state.mkdir(parents=True, exist_ok=True)
    # A local lock prevents two batch drivers from cleaning/executing the same stage.
    import fcntl
    with (state / (a.name + '.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = state / (a.name + '.done.json')
        started = state / (a.name + '.started.json')
        if done.exists():
            saved = json.loads(done.read_text())
            if saved['identity'] != identity or saved['artifacts'] != fingerprint(outputs):
                raise ValueError('completed stage inputs/outputs changed; choose a new episode root')
            print('SKIP', a.name)
            return
        if started.exists():
            if json.loads(started.read_text()) != identity:
                raise ValueError('interrupted stage inputs changed; choose a new episode root')
            for path in outputs:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
        elif any(x.exists() for x in outputs):
            raise FileExistsError('refusing to overwrite outputs without an ownership record')
        started.write_text(json.dumps(identity, indent=2) + '\n')
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)
        from time import time_ns
        with (logs / f'{a.name}.{time_ns()}.log').open('w') as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise SystemExit(code)
        artifacts = fingerprint(outputs)
        if not artifacts or any(not x.exists() for x in outputs):
            raise ValueError('stage did not produce its declared artifacts')
        pending = done.with_suffix('.tmp')
        pending.write_text(json.dumps({'identity': identity, 'artifacts': artifacts}, indent=2) + '\n')
        pending.replace(done)
        started.unlink()
        print('DONE', a.name)


if __name__ == '__main__':
    main()
