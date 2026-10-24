#!/usr/bin/env bash
# Usage: scripts/vast/start_run.sh RUN_DIR INSTANCE_ID CAP_USD [CREDIT_FLOOR_USD]
# Bind and arm the detached financial watchdog before uploading or starting a writer.
# A failed launch can be retried with the same arguments while its original watchdog keeps enforcing limits.
# The default $4 floor covers ~13 boxes' spend between two credit checks (~$2 per 10 min at $0.92/h each).
set -euo pipefail
D=$(cd "${1:?run dir}" && pwd) id=${2:?instance id} cap=${3:?cap in USD} floor=${4:-4}
P=$(cd "$(dirname "$0")/../.." && pwd)
cd "$P"
exec python3 -I - "$D" "$id" "$cap" "$floor" <<'PY'
import fcntl, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

d, instance, cap, floor = Path(sys.argv[1]), *sys.argv[2:]
if not instance.isdecimal() or not all(math.isfinite(float(x)) and float(x) >= 0 for x in (cap, floor)):
    sys.exit('invalid instance id, cap or credit floor; no launch started')
for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACEHUB_API_TOKEN'):
    os.environ.pop(key, None)

def locked(path, bind=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit('a launch or watchdog already owns this run or instance')
    if bind:
        previous = path.read_text().strip()
        if previous and previous != str(d.resolve()) and not (Path(previous) / 'STOPPED').exists():
            sys.exit('instance has an unfinished run binding; resume that run directory')
        handle.truncate(0); handle.write(str(d.resolve()) + '\n'); handle.flush()
    return handle

owner = locked(d / '.start-run.lock')
pending = d / 'controller-launch.pending.json'
record = json.loads(pending.read_text()) if pending.exists() else None
script = str(Path('scripts/vast/watchdog.sh').resolve())

def live(pid):
    args = subprocess.run(['ps', '-ww', '-o', 'args=', '-p', str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
    return args.endswith(f'{script} {d} {cap} {floor}')

locks = []
if record and record['instance'] == instance and record['cap'] == cap and record['floor'] == floor and live(record['pid']):
    pid, offset = record['pid'], record['log_offset']
    if (d / 'host').read_text().split()[0] != instance:
        sys.exit('pending launch instance binding changed; no launch started')
else:
    locks = [locked(d / '.watchdog.lock'), locked(Path('data/vast-controller-locks') / f'{instance}.lock', bind=True)]
    if (d / 'host').exists() and (d / 'host').read_text().split()[0] != instance and not (d / 'STOPPED').exists():
        sys.exit('previous instance has no completed stop; use a distinct run directory')
    info = json.loads(subprocess.check_output(['vastai', 'show', 'instance', instance, '--raw'], text=True, timeout=120))
    host = (info.get('public_ipaddr') or '').strip()
    ports = (info.get('ports') or {}).get('22/tcp') or []
    port = ports[0]['HostPort'] if ports else ''
    (d / 'host').write_text(f'{instance} {host} {port}\n')
    for name in ('STOPPED', 'BACKUP_FAILED', 'QUIESCED'):
        (d / name).unlink(missing_ok=True)
    offset = (d / 'watchdog.log').stat().st_size if (d / 'watchdog.log').exists() else 0
    awake = ['caffeinate', '-i'] if shutil.which('caffeinate') else []
    with (d / 'controller.log').open('a') as log:
        process = subprocess.Popen([*awake, 'bash', script, str(d), cap, floor], stdout=log, stderr=log,
            start_new_session=True, pass_fds=tuple(lock.fileno() for lock in locks),
            env={**os.environ, 'WATCHDOG_LOCKED': str(d)})
    pid = process.pid
    (d / 'controller-watchdog.pid').write_text(f'{pid}\n')
    record = {'instance': instance, 'cap': cap, 'floor': floor, 'pid': pid, 'log_offset': offset}
    temporary = pending.with_suffix('.tmp'); temporary.write_text(json.dumps(record)); temporary.replace(pending)

end = time.monotonic() + 60
while True:
    if not live(pid) or (d / 'STOPPED').exists():
        sys.exit('watchdog failed to arm; no writer started; inspect controller.log')
    if (d / 'watchdog.log').exists() and 'watchdog start ' in (d / 'watchdog.log').read_text()[offset:]:
        break
    if time.monotonic() >= end:
        sys.exit('watchdog not yet armed; no writer started; retry with the same arguments')
    time.sleep(.1)

result = subprocess.run(['bash', 'scripts/vast/launch.sh', str(d), instance],
    env={**os.environ, 'LAUNCH_LOCKED': str(d), 'VAST_GUARDED_LAUNCH': '1'}, pass_fds=tuple(lock.fileno() for lock in locks))
if result.returncode:
    if not live(pid): sys.exit('launch failed and watchdog stopped; inspect STOPPED/controller.log before proceeding')
    sys.exit('launch failed; watchdog remains active; retry with the same arguments')
if not live(pid) and not (d / 'STOPPED').exists():
    sys.exit('watchdog exited during launch; inspect controller.log before proceeding')
pending.unlink()
print(f'watchdog {pid}: cap ${cap}, credit floor ${floor}')
PY
