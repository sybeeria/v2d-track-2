#!/usr/bin/env bash
# Usage: scripts/vast/launch.sh RUN_DIR INSTANCE_ID
# Copies the project (no data/external/venvs/runs) plus the screen's seed configs to the box, installs
# scripts/vast/box.sh as /workspace/<run>.sh and RUN_DIR/jobs.txt as /workspace/<run>.jobs, where <run> is
# RUN_DIR's name (plus optional RUN_DIR/pre.sh and post.sh hooks as /workspace/<run>.pre.sh and .post.sh), and
# starts the box script detached (log: /workspace/<run>.log). Records the host in RUN_DIR/host.
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN
D=$(cd "${1:?run dir}" && pwd) id=${2:?instance id}
P=$(cd "$(dirname "$0")/../.." && pwd)
name=$(basename "$D")
# A run directory has one live instance binding and one controller owner.
if [ "${LAUNCH_LOCKED:-}" != "$D" ]; then
    exec python3 - "$0" "$D" "$@" <<'PYLAUNCHLOCK'
import fcntl, os, sys
from pathlib import Path
instance = sys.argv[4]
if not instance.isdecimal(): sys.exit('invalid instance id')
shared = Path(sys.argv[1]).resolve().parents[2] / 'data/vast-controller-locks'
shared.mkdir(parents=True, exist_ok=True)
locks = [open(os.path.join(sys.argv[2], '.watchdog.lock'), 'a'), open(shared / (instance + '.lock'), 'a')]
for lock in locks:
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit('a launch or watchdog already owns this run or instance')
    if lock is locks[1]:
        previous = Path(lock.name).read_text().strip()
        owner = Path(sys.argv[2]).resolve()
        if previous and previous != str(owner) and not (Path(previous) / 'STOPPED').exists():
            sys.exit('instance has an unfinished run binding; resume that run directory')
        lock.truncate(0); lock.write(str(owner) + '\n'); lock.flush()
    os.set_inheritable(lock.fileno(), True)
os.environ['LAUNCH_LOCKED'] = sys.argv[2]
os.execvp('bash', ['bash', sys.argv[1], *sys.argv[3:]])
PYLAUNCHLOCK
fi
if [ -f "$D/host" ]; then
    read -r previous_id _ < "$D/host"
    if [ "$previous_id" != "$id" ] && [ ! -f "$D/STOPPED" ]; then
        echo "previous instance $previous_id has no completed stop; use a distinct run directory" >&2; exit 2
    fi
fi
# A staged pilot without its binding must fail verification, not skip it.
if [ -d "$D/payload/runs/$name/input-assets" ]; then
    python3 -B "$P/data/brainstorm6/PROD-READY/verify_payload.py" --pilot "$D"
fi
read -r host port < <(vastai show instance "$id" --raw | python3 -I -c "import json,sys; o=json.load(sys.stdin); print(o['public_ipaddr'].strip(), o['ports']['22/tcp'][0]['HostPort'])")
echo "$id $host $port" > "$D/host"
# The box gets the working tree without .git; tie each launch to the commit a code_commit_url can name.
echo "$(date -u +%FT%TZ) $id $(git -C "$P" rev-parse HEAD 2>/dev/null || echo no-git) changed_files=$(git -C "$P" status --porcelain 2>/dev/null | wc -l | tr -d ' ')" >> "$D/launch-code.log"
if [ "${VAST_GUARDED_LAUNCH:-}" != 1 ]; then rm -f "$D/STOPPED" "$D/BACKUP_FAILED" "$D/QUIESCED"; fi
ssh_opts=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -i "$HOME/.ssh/id_ed25519" -p "$port")
# A pre-upload watchdog cycle cannot arm a script that is not installed yet.
ensure_deadman() {
    [ "${VAST_GUARDED_LAUNCH:-}" = 1 ] || return 0
    local deadline grace
    read -r deadline grace < "$D/deadman-arm-$id" || { echo "guarded launch has no deadman budget" >&2; return 1; }
    [[ "$deadline" =~ ^[0-9]+$ && "$grace" =~ ^[0-9]+$ ]] || return 1
    ssh "${ssh_opts[@]}" root@"$host" "
        echo $deadline $grace > /workspace/$name.deadline.tmp && mv /workspace/$name.deadline.tmp /workspace/$name.deadline || exit 1
        { setsid nohup bash /workspace/v2d-track-2/scripts/vast/deadman.sh $name >> /workspace/$name.deadman.log 2>&1 < /dev/null & }
        for attempt in {1..50}; do
            if read -r armed_instance armed_pid < /workspace/$name.deadman.ready 2>/dev/null &&
               [ \"\$armed_instance\" = $id ] && kill -0 \"\$armed_pid\" 2>/dev/null; then exit 0; fi
            sleep .1
        done
        echo 'box deadman failed to arm; no writer started' >&2; exit 1"
}
# An SSH acknowledgement can be lost after the box starts. Preserve that writer and its log on retry.
existing=$(ssh "${ssh_opts[@]}" root@"$host" "
    if [ -s /workspace/$name.pid ]; then
        read -r existing_pid < /workspace/$name.pid
        case \"\$existing_pid\" in ''|*[!0-9]*) exit 1;; esac
        if kill -0 \"\$existing_pid\" 2>/dev/null || grep -qE '^box (start|done|failed) ' /workspace/$name.log; then echo started; fi
    fi
    true" | tail -1)
if [ "$existing" = started ]; then ensure_deadman; echo "box already launched: $name; original writer retained"; exit 0; fi
ssh "${ssh_opts[@]}" root@"$host" "mkdir -p /workspace/v2d-track-2/data/denoise_eval" 2>&1 | grep -v "Welcome\|Have fun" || true
if [ -f "$D/entry.sh" ]; then
    ssh "${ssh_opts[@]}" root@"$host" "mkdir -p /workspace/v2d-track-2/scripts/tier3_native"
    rsync -az -e "ssh ${ssh_opts[*]}" --exclude '.*' --exclude __pycache__ --include '*/' --include '*.py' --include '*.sh' --include '*.diff' --exclude '*' \
        "$P/scripts/tier3_native/" root@"$host":/workspace/v2d-track-2/scripts/tier3_native/
    rsync -az -e "ssh ${ssh_opts[*]}" "$P/scripts/tier3_initializers.py" root@"$host":/workspace/v2d-track-2/scripts/
    if [ -f "$P/scripts/vast/deadman.sh" ]; then
        ssh "${ssh_opts[@]}" root@"$host" "mkdir -p /workspace/v2d-track-2/scripts/vast"
        rsync -az -e "ssh ${ssh_opts[*]}" "$P/scripts/vast/deadman.sh" root@"$host":/workspace/v2d-track-2/scripts/vast/
    fi
else
    rsync -az -e "ssh ${ssh_opts[*]}" --exclude '.venv*' --exclude external --exclude data --exclude runs \
        --exclude '__pycache__' --exclude .git "$P/" root@"$host":/workspace/v2d-track-2/
fi
if [ ! -f "$D/entry.sh" ]; then
    rsync -az -e "ssh ${ssh_opts[*]}" "$P"/data/denoise_eval/*config.json root@"$host":/workspace/v2d-track-2/data/denoise_eval/
fi
if [ -f "$D/upload.list" ]; then
    # Files are an explicit project-relative allowlist produced by prepare.py.
    python3 -I "$P/scripts/tier3_native/prepare.py" verify-upload "$D"
    rsync -az --files-from="$D/upload.list" -e "ssh ${ssh_opts[*]}" "$P/" root@"$host":/workspace/v2d-track-2/
fi
if [ -d "$D/payload" ] && [ -f "$D/entry.sh" ]; then
    ssh "${ssh_opts[@]}" root@"$host" "mkdir -p /workspace/v2d-track-2/runs/$name/control"
    rsync -az -e "ssh ${ssh_opts[*]}" "$D/payload/" root@"$host":/workspace/v2d-track-2/runs/"$name"/control/
fi
rsync -az -e "ssh ${ssh_opts[*]}" "$P/scripts/vast/box.sh" root@"$host":/workspace/"$name".sh
[ ! -f "$D/jobs.txt" ] || rsync -az -e "ssh ${ssh_opts[*]}" "$D/jobs.txt" root@"$host":/workspace/"$name".jobs
# Stage explicit run-owned production files despite the general data exclusion.
if [ -d "$D/payload" ] && [ ! -f "$D/entry.sh" ]; then
    rsync -az -e "ssh ${ssh_opts[*]}" "$D/payload/" root@"$host":/workspace/v2d-track-2/
fi
[ ! -f "$D/env.sh" ] || rsync -az -e "ssh ${ssh_opts[*]}" "$D/env.sh" root@"$host":/workspace/"$name.env".sh
# Restore the frozen run tree before starting any writer on a replacement box.
if [ -d "$D/pulled/$name" ]; then
    ssh "${ssh_opts[@]}" root@"$host" "mkdir -p /workspace/v2d-track-2/runs/$name"
    rsync -az --exclude .rsync-partial --exclude control -e "ssh ${ssh_opts[*]}" "$D/pulled/$name/" root@"$host":/workspace/v2d-track-2/runs/"$name"/
fi
for hook in pre post entry; do
    [ ! -f "$D/$hook.sh" ] || rsync -az -e "ssh ${ssh_opts[*]}" "$D/$hook.sh" root@"$host":/workspace/"$name.$hook".sh
done
ensure_deadman
[ "${VAST_GUARDED_LAUNCH:-}" != 1 ] || [ ! -f "$D/STOPPED" ] || { echo "watchdog stopped this instance before launch" >&2; exit 1; }
ssh "${ssh_opts[@]}" root@"$host" "setsid nohup bash /workspace/$name.sh > /workspace/$name.log 2>&1 < /dev/null & echo \$! > /workspace/$name.pid; echo launched" 2>&1 | grep -v "Welcome\|Have fun"
