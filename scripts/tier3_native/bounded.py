"""Run one Mac bridge command with a sampled process-tree RSS/deadline guard."""
import os
from pathlib import Path
import subprocess
import time


class ResourceHold(RuntimeError):
    pass


def run(command, log, *, max_bytes=1_850_000_000, timeout=1100, env=None):
    import psutil
    peak = 0; start = time.monotonic(); reason = None
    fullenv = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'PYTHONDONTWRITEBYTECODE': '1', **(env or {})}
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as f:
        child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, env=fullenv)
        proc = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                rss = 0
                for item in [proc, *proc.children(recursive=True)]:
                    try:
                        rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                if rss > max_bytes or time.monotonic() - start > timeout:
                    reason = 'RSS limit' if rss > max_bytes else 'deadline'
                    break
                time.sleep(.05)
        finally:
            if child.poll() is None:
                descendants = proc.children(recursive=True)
                for item in reversed(descendants):
                    try:
                        item.terminate()
                    except psutil.NoSuchProcess:
                        pass
                child.terminate()
                _, alive = psutil.wait_procs([proc, *descendants], timeout=3)
                for item in alive:
                    try:
                        item.kill()
                    except psutil.NoSuchProcess:
                        pass
            child.wait()
    info = {'command': command, 'exit_code': child.returncode, 'wall_seconds': time.monotonic()-start, 'peak_tree_rss_bytes_sampled': peak,
            'memory_limit_bytes': max_bytes, 'limitation': 'sampled summed RSS; not instantaneous unique physical memory', 'resource_hold': reason}
    from batch import atomic
    atomic(log.with_suffix('.metrics.json'), info)
    if reason:
        raise ResourceHold(reason)
    if child.returncode:
        raise RuntimeError('command exited ' + str(child.returncode) + '; ' + log.read_text(errors='replace')[-4096:])
    return info
