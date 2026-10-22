#!/usr/bin/env bash
# Usage: scripts/vast/watchdog.sh RUN_DIR CAP_DOLLARS [CREDIT_FLOOR]   (run detached under caffeinate)
# Every 10 min: logs status to RUN_DIR/watchdog.log and pulls runs/<run> (including policy and learner checkpoints) and /workspace/*.log into
# RUN_DIR/pulled. Destroys the instance in RUN_DIR/host after a final pull when the box script has logged "box done",
# when it has been gone for 60 min without that (unless RUN_DIR/KEEP exists), when this instance's estimated spend
# reaches CAP_DOLLARS, or when the account's Vast credit falls below CREDIT_FLOOR (default 1). Each cycle also arms
# scripts/vast/deadman.sh on the box, which stops the instance itself if this watchdog stops working.
set -uo pipefail
D=$(cd "${1:?run dir}" && pwd) cap=${2:?cap dollars} floor=${3:-1}
name=$(basename "$D")
read -r id host port < "$D/host"
if [ "${WATCHDOG_LOCKED:-}" != "$D" ]; then
    exec python3 - "$0" "$D" "$@" <<'PYWATCHLOCK'
import fcntl, os, sys
lock = open(os.path.join(sys.argv[2], '.watchdog.lock'), 'a')
try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError: sys.exit('another watchdog owns this run directory')
os.set_inheritable(lock.fileno(), True)
os.environ['WATCHDOG_LOCKED'] = sys.argv[2]
os.execvp('bash', ['bash', sys.argv[1], *sys.argv[3:]])
PYWATCHLOCK
fi
# Bound wall time as well as socket inactivity: a live remote process can hang forever.
bounded() {
    python3 -I - "$@" <<'PYBOUND'
import os, signal, subprocess, sys
timeout = float(os.environ.get('WATCHDOG_COMMAND_TIMEOUT_SECONDS', '120'))
with subprocess.Popen(sys.argv[1:], start_new_session=True, close_fds=False) as process:
    try:
        sys.exit(process.wait(timeout=timeout))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        print(f'watchdog command timed out after {timeout}s: {sys.argv[1]}', file=sys.stderr)
        sys.exit(124)
PYBOUND
}
ssh_opts=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=2 -i "$HOME/.ssh/id_ed25519" -p "$port")
read -r start rate < <(bounded vastai show instance "$id" --raw | python3 -I -c "import json,sys,math; o=json.load(sys.stdin); s,r=float(o['start_date']),float(o['dph_total']); assert math.isfinite(s) and s>0 and math.isfinite(r) and r>0; print(s,r)") || {
    echo "cannot read valid instance start/rate; watchdog not armed" >&2; exit 1;
}
python3 -I -c "import sys,math; c,f=map(float,sys.argv[1:]); assert all(math.isfinite(x) and x>=0 for x in (c,f))" "$cap" "$floor" || exit 2
# The box-side backstop (scripts/vast/deadman.sh) stops the instance 30 min past the cap if this watchdog cannot.
deadline=$(python3 -I -c "import sys; s,c,r=map(float,sys.argv[1:]); print(int(s + c / r * 3600 + 1800))" "$start" "$cap" "$rate")
mkdir -p "$D/pulled"
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/watchdog.log"; }

pull() {
    bounded rsync -az --timeout=90 -e "ssh ${ssh_opts[*]}" --exclude '.*.tmp' --exclude 'state_*.safetensors' --exclude 'replay_*.safetensors' --exclude '*.ckpt' --exclude '*.pt' \
        root@"$host":/workspace/v2d-track-2/runs/"$name" "$D/pulled/" 2>> "$D/watchdog.log" || return 1
    # Actors are small; compact learner state dominates disk. Keep two immutable state generations per policy.
    if ! bounded ssh -n "${ssh_opts[@]}" root@"$host" "python3 - /workspace/v2d-track-2/runs/$name <<'PYSTATES'
from pathlib import Path
import re, sys
root = Path(sys.argv[1])
groups = {}
for path in root.rglob('state_*.safetensors'):
    match = re.fullmatch(r'state_([0-9]+)\.safetensors', path.name)
    if match and path.is_file():
        groups.setdefault(path.parent, []).append((int(match[1]), path))
for directory in sorted(groups):
    for _, path in sorted(groups[directory])[-2:]:
        print(path.relative_to(root).as_posix())
PYSTATES" > "$D/.state-files" 2>> "$D/watchdog.log"; then return 1; fi
    bounded rsync -az --timeout=90 -e "ssh ${ssh_opts[*]}" --files-from="$D/.state-files" \
        root@"$host":/workspace/v2d-track-2/runs/"$name"/ "$D/pulled/$name/" 2>> "$D/watchdog.log" || return 1
    python3 -I - "$D/pulled/$name" "$D/.state-files" <<'PYPRUNE' || return 1
from pathlib import Path
import re, sys
root = Path(sys.argv[1])
keep = {Path(p) for p in Path(sys.argv[2]).read_text().splitlines()}
for parent in {p.parent for p in keep}:
    for path in (root / parent).glob('state_*.safetensors'):
        if re.fullmatch(r'state_[0-9]+\.safetensors', path.name) and path.relative_to(root) not in keep:
            path.unlink()
PYPRUNE
    bounded rsync -az --timeout=90 -e "ssh ${ssh_opts[*]}" root@"$host":'/workspace/*.log' "$D/pulled/" 2>> "$D/watchdog.log"
}

# destroy REASON [hard]. Soft reasons keep the instance when it cannot be stopped or backed up and retry next
# cycle, so the watchdog stays alive to enforce the cap. Hard reasons (cap, credit floor) destroy anyway: every
# cycle already pulls, so at most one cycle of work is lost, while an exited watchdog would bill without bound.
destroy() {
    local hard=${2:-}
    # A dead parent can leave a live trainer; stop its entire setsid group on every exit path.
    if ! bounded ssh -n "${ssh_opts[@]}" root@"$host" "
            read -r pid < /workspace/$name.pid || exit 1
            case \"\$pid\" in ''|*[!0-9]*) exit 1;; esac
            [ \"\$pid\" -gt 1 ] || exit 1
            kill -TERM -- -\$pid 2>/dev/null || true
            sleep 5
            kill -KILL -- -\$pid 2>/dev/null || true" >> "$D/watchdog.log" 2>&1; then
        log "cannot stop writer ($1)"
        [ -n "$hard" ] || return 1
    fi
    if ! pull && ! pull; then
        log "backup failed ($1)"
        [ -n "$hard" ] || return 1
        echo "$1" > "$D/BACKUP_FAILED"
    fi
    if ! bounded vastai destroy instance "$id" -y >> "$D/watchdog.log" 2>&1; then
        log "destroy failed ($1); retrying next cycle"; return 1
    fi
    log "destroy requested for instance $id ($1)"; echo "$1" > "$D/STOPPED"
    exit 0
}

dead_since=
progress_before=
progress_since=$(date +%s)
log "watchdog start id=$id rate=$rate cap=$cap floor=$floor"
while true; do
    spend=$(python3 -I -c "import time; print(max(0,time.time()-$start)/3600*$rate)")
    credit=$(bounded vastai show user --raw 2>/dev/null | python3 -I -c "import json,sys; print(float(json.load(sys.stdin)['credit']))" 2>/dev/null)
    state=$(bounded ssh -n "${ssh_opts[@]}" root@"$host" "
        if grep -q '^box done' /workspace/$name.log; then echo done
        elif grep -q '^box failed' /workspace/$name.log; then echo failed
        elif pgrep -f '[/]workspace/$name.sh' > /dev/null; then echo running
        else echo dead; fi" 2>/dev/null | tail -1)
    bounded ssh -n "${ssh_opts[@]}" root@"$host" "
        grep -E '^(box|bootstrap|screen) ' /workspace/$name.log | tail -1
        grep -E '^GPU [0-9]+: ' /workspace/$name.log | tail -2
        for f in \$(ls -t \$(find /workspace/v2d-track-2/runs/$name -name train.log 2>/dev/null) 2>/dev/null | head -2); do
            grep -E '^training progress' \"\$f\" | tail -1 | cut -c1-120; done
        tail -1 /workspace/gpu_samples.log 2>/dev/null" > "$D/status.txt" 2>/dev/null
    log "state=${state:-unreachable} spend=\$$spend credit=\$${credit:-?} $(tr '\n' '|' < "$D/status.txt" | cut -c1-400)"
    grace=5400; [ ! -e "$D/KEEP" ] || grace=0
    bounded ssh -n "${ssh_opts[@]}" root@"$host" "echo $deadline $grace > /workspace/$name.deadline.tmp && mv /workspace/$name.deadline.tmp /workspace/$name.deadline &&
        { setsid nohup bash /workspace/v2d-track-2/scripts/vast/deadman.sh $name >> /workspace/$name.deadman.log 2>&1 < /dev/null & }" \
        >> "$D/watchdog.log" 2>&1 || log "cannot arm box deadman"
    pull
    if [ "$state" = running ]; then
        active=$(bounded ssh -n "${ssh_opts[@]}" root@"$host" "python3 - /workspace/v2d-track-2/runs/$name <<'PYJOBS'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
jobs = []
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv = proc.read_bytes().decode().split('\0')
        script = next((Path(a).name for a in argv if Path(a).name in ('train_flash_sac.py', 'evaluate_policy.py')), None)
        if script is None: continue
        key = 'output_dir=' if script == 'train_flash_sac.py' else 'evaluation.metrics_output='
        path = Path(next(a[len(key):].strip(\"'\") for a in argv if a.startswith(key))).resolve()
        if not path.is_relative_to(root): continue
        log = path / 'train.log' if script == 'train_flash_sac.py' else path
        if script == 'train_flash_sac.py':
            lines = [line for line in log.read_text(errors='replace').splitlines() if line.startswith('training progress')]
            fingerprint = lines[-1] if lines else ''
        else:
            fingerprint = str(log.stat().st_mtime_ns) if log.exists() else ''
        jobs.append({'id': proc.parent.name, 'progress': fingerprint})
    except (OSError, UnicodeError, StopIteration): continue
print(json.dumps(jobs))
PYJOBS" 2>/dev/null) || active=
        if [ -n "$active" ]; then
            stalled=$(printf '%s' "$active" | python3 -I -c '
import json, sys
from pathlib import Path
path, now = Path(sys.argv[1]), int(sys.argv[2])
try: before = json.loads(path.read_text())
except (OSError, ValueError): before = {}
current, stalled = {}, False
for job in json.load(sys.stdin):
    old = before.get(job["id"], {})
    since = old["since"] if old.get("progress") == job["progress"] else now
    current[job["id"]] = {"progress": job["progress"], "since": since}
    stalled |= now - since >= 3600
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(current)); temporary.replace(path)
print("stalled" if stalled else "active")' "$D/job-progress-$id.json" "$(date +%s)")
            if [ "$stalled" = stalled ] && [ ! -e "$D/KEEP" ]; then destroy "job no progress for 60 min"; fi
        fi
        progress=$(bounded ssh -n "${ssh_opts[@]}" root@"$host" "
            grep -E '^(box|bootstrap|screen|GPU [0-9]+:) ' /workspace/$name.log
            find /workspace/v2d-track-2/runs/$name -name train.log -exec sh -c 'grep -a \"^training progress\" \"\$1\" | tail -1' sh {} \;
            find /workspace/v2d-track-2/runs/$name -type f \( -name 'policy_*.safetensors' -o -name scores.json -o -name metrics.json \) -printf '%p %T@ %s\n'
            true" 2>/dev/null) || progress=
        if [ -n "$progress" ] && [ "$progress" = "$progress_before" ]; then
            if [ ! -e "$D/KEEP" ] && [ $(( $(date +%s) - progress_since )) -ge 3600 ]; then
                destroy "no progress for 60 min"
            fi
        else
            progress_before=$progress; progress_since=$(date +%s)
        fi
    fi
    case "$state" in
        done) destroy finished ;;
        failed) destroy "box failed" ;;
        dead) dead_since=${dead_since:-$(date +%s)}
              if [ ! -e "$D/KEEP" ] && [ $(( $(date +%s) - dead_since )) -ge 3600 ]; then destroy "box script dead"; fi ;;
        *) dead_since= ;;
    esac
    if python3 -I -c "import sys; sys.exit(0 if $spend >= $cap else 1)"; then destroy "budget cap \$$cap" hard; fi
    if [ -n "$credit" ] && python3 -I -c "import sys; sys.exit(0 if $credit < $floor else 1)"; then destroy "credit \$$credit below \$$floor" hard; fi
    sleep 600
done
