"""Authorized morning controller: launch, arm watchdog during upload, validate canary, release rest."""
import argparse
import fcntl
import ipaddress
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch import atomic, digest
from bounded import run
from prepare import verify

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path); p.add_argument('instance_id', type=int)
    p.add_argument('--cap', type=float, required=True); p.add_argument('--credit-floor', type=float, default=1)
    p.add_argument('--hourly-rate-ceiling', type=float, required=True)
    p.add_argument('--gpu-memory-mib', type=int, required=True)
    p.add_argument('--confirm-paid-reconstruction', action='store_true', required=True)
    a = p.parse_args(); a.run = a.run.resolve()
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACEHUB_API_TOKEN'):
        os.environ.pop(key, None)
    if a.cap <= 0 or a.credit_floor < 0 or a.hourly_rate_ceiling <= 0 or a.gpu_memory_mib < 24000:
        raise ValueError('invalid cap/floor/rate/VRAM')
    ownership = (a.run / '.native-controller.lock').open('a')
    fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.run / 'host').exists():
        raise ValueError('run already has a host; use a fresh controller directory for replacement launch')
    verify(a.run)
    backed_up = sum(p.stat().st_size for p in (a.run / 'pulled').rglob('*') if p.is_file() and not p.is_symlink() and not any(part.startswith('.') for part in p.relative_to(a.run / 'pulled').parts))
    if shutil.disk_usage(a.run).free + backed_up < 150 * 1024**3:
        raise ValueError('reserve 150 GiB Mac capacity including an existing restored backup')
    # This is a real provider call, deliberately confined to this explicitly invoked morning command.
    instance = json.loads(subprocess.check_output(['vastai', 'show', 'instance', str(a.instance_id), '--raw']))
    rate = float(instance['dph_total'])
    if not 0 < rate <= a.hourly_rate_ceiling:
        raise ValueError('offer hourly rate exceeds frozen ceiling')
    plan = json.loads((a.run / 'payload/plan.json').read_text())
    rows = sorted(plan['episodes'], key=lambda r: (-r['frames'], r['episode']))
    session = {'plan_sha256': digest(a.run / 'payload/plan.json'), 'session_id': uuid.uuid4().hex}
    atomic(a.run / 'payload/session.json', session)
    projected_seconds = sum(r['frames'] for r in rows) * 766.2301342209103 / 858
    # Budget for the serial worst case; multi-GPU speedup is not assumed before the canary.
    if a.cap < rate * (2 * projected_seconds / 3600 + 1.2):
        raise ValueError('cap below 2x serial inference + 1.2h setup allowance; prepare an explicit smaller tranche')
    host = str(ipaddress.ip_address(instance['public_ipaddr'].strip()))
    port = int(instance['ports']['22/tcp'][0]['HostPort'])
    if not 1 <= port <= 65535: raise ValueError('invalid SSH port')
    (a.run / 'host').write_text(f'{a.instance_id} {host} {port}\n')
    with (a.run / 'launch.log').open('w') as f, (a.run / 'controller_watchdog.log').open('w') as wf:
        # Reconstruction pulls are tens of GiB; a 120 s transfer per 10-min cycle would leave the box idle for hours.
        env = {**os.environ, 'WATCHDOG_COMMAND_TIMEOUT_SECONDS': os.environ.get('WATCHDOG_COMMAND_TIMEOUT_SECONDS', '1800')}
        watchdog = subprocess.Popen(['bash', str(ROOT / 'scripts/vast/watchdog.sh'), str(a.run), str(a.cap), str(a.credit_floor)], stdout=wf, stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + 60
        while not (a.run / 'watchdog.log').exists() or 'watchdog start ' not in (a.run / 'watchdog.log').read_text():
            if watchdog.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('watchdog failed to arm; no upload/writer started; inspect controller_watchdog.log')
            time.sleep(.1)
        # The same instance binding is already protected by our controller lock
        # and live watchdog. This hands off CODE-REVIEW/02's launch lock guard;
        # a second independent launcher remains rejected by the watchdog lock.
        if watchdog.poll() is not None or (a.run / 'STOPPED').exists():
            raise RuntimeError('watchdog stopped before launch; no writer started')
        launch = subprocess.Popen(['bash', str(ROOT / 'scripts/vast/launch.sh'), str(a.run), str(a.instance_id)],
                                  stdout=f, stderr=subprocess.STDOUT, env={**os.environ, 'LAUNCH_LOCKED': str(a.run)})
        if launch.wait():
            raise RuntimeError('launch failed; watchdog remains active to enforce cap/floor')
        if watchdog.poll() is not None or (a.run / 'STOPPED').exists():
            raise RuntimeError('watchdog stopped during launch; inspect final backup before any retry')
        canary = a.run / 'pulled' / a.run.name / 'episodes' / rows[0]['sequence']
        while True:
            receipt = a.run / 'pull.success.json'
            if ((canary / 'result.json').exists() and receipt.exists()
                and json.loads(receipt.read_text()).get('canary', {}).get('session_id') == session['session_id']
                and json.loads((canary / 'result.json').read_text()).get('session_id') == session['session_id']):
                break
            if watchdog.poll() is not None:
                raise RuntimeError('watchdog stopped before canary; inspect STOPPED/BACKUP_FAILED/logs')
            time.sleep(10)
        result = json.loads((canary / 'result.json').read_text())
        threshold = 2 * 766.2301342209103 * rows[0]['frames'] / 858 + 90
        metrics = {}
        for metric in result['stage_metrics']:
            metrics[metric['stage']] = metric
        backup_canary = json.loads((a.run / 'pull.success.json').read_text())['canary']
        go = backup_canary == {**session, 'canary_sha256': result.get('initializers_sha256')}
        go &= result['status'] == 'SUCCESS' and result['inputs']['plan_sha256'] == digest(a.run / 'payload/plan.json')
        go &= result.get('initializers_sha256') == digest(canary / 'initializers.pth')
        go &= result['wall_seconds_this_attempt'] <= threshold and result['peak_gpu_memory_mib_sampled'] is not None
        go &= result['peak_gpu_memory_mib_sampled'] is not None and result.get('gpu_memory_total_mib') is not None and result['peak_gpu_memory_mib_sampled'] <= .9 * min(a.gpu_memory_mib, result['gpu_memory_total_mib'])
        go &= set(metrics) == {'video', 'sam2', 'pack_masks', '01_depth', '02_export', '03_human', '04_align', '05_object', 'validate'} and all(x['exit_code'] == 0 for x in metrics.values())
        go &= not (a.run / 'BACKUP_FAILED').exists()
        if not go:
            atomic(a.run / 'CANARY_NO_GO.json', {'result': result, 'max_seconds': threshold, 'max_sampled_vram_mib': .9*a.gpu_memory_mib})
            raise RuntimeError('canary NO-GO; remaining episodes stay paused, watchdog still enforces cap/floor')
        checks = a.run / 'canary_check'; checks.mkdir()
        seq = rows[0]['sequence']; out = canary / 'output' / seq
        # Re-open actual pulled pickles and enforce coverage/rigidity/no-oracle through the public adapter.
        run([sys.executable, '-B', str(ROOT / 'scripts/tier3_initializers.py'), '--human', str(out / f'mhr_init/{seq}.pkl'),
             '--object', str(out / f'foundationpose/{seq}.pkl'), '--wild-export', str(out / f'export/{seq}/wild_export.json'),
             '--expected-frames', str(rows[0]['frames']), '--out', str(checks / 'initializers.pth')], checks / 'validation.log')
        approval = {**session, 'canary_sha256': result['initializers_sha256']}
        atomic(a.run / 'GO.json', approval)
        _, host, port = (a.run / 'host').read_text().split()
        subprocess.run(['rsync', '-az', '-e', f'ssh -i {Path.home()}/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=accept-new', str(a.run / 'GO.json'),
                        f'root@{host}:/workspace/v2d-track-2/runs/{a.run.name}/episodes/GO.json'], check=True)
        print('Canary GO: native execution/provenance only; reference quality remains separately gated.', flush=True)
        code = watchdog.wait()
        if code or (a.run / 'BACKUP_FAILED').exists() or not (a.run / 'STOPPED').exists():
            raise RuntimeError('watchdog did not finish with a successful final backup/destroy request')
        results = list((a.run / 'pulled' / a.run.name / 'episodes').glob('*/result.json'))
        statuses = {json.loads(x.read_text())['episode']: json.loads(x.read_text())['status'] for x in results}
        atomic(a.run / 'native_status.json', statuses)
        if set(statuses) != {r['episode'] for r in rows} or any(v != 'SUCCESS' for v in statuses.values()):
            raise SystemExit('native batch incomplete/failed; successes retained, retry in a replacement run')
        print('Native batch pulled; STOPPED is a destroy-request receipt, not independent billing termination proof.')


if __name__ == '__main__':
    main()
