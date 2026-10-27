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
endpoint=$(vastai show instance "$id" --raw | python3 -I -c "
import json,sys; o=json.load(sys.stdin); h=o['public_ipaddr'].strip(); p=str(o['ports']['22/tcp'][0]['HostPort'])
sys.exit(1) if not h or len(h.split()) != 1 or not p.isdecimal() else print(h, p)") ||
    { echo "cannot read the SSH endpoint of instance $id; $D/host kept" >&2; exit 1; }
read -r host port <<< "$endpoint"
echo "$id $host $port" > "$D/host"
# The box gets the working tree without .git; tie each launch to the commit a code_commit_url can name.
code_commit=$(git -C "$P" rev-parse HEAD 2>/dev/null || echo no-git)
changed_files=$(git -C "$P" status --porcelain 2>/dev/null | wc -l | tr -d ' ') || changed_files=unknown
if [[ "$code_commit" =~ ^[0-9a-f]{40}$ && "$changed_files" =~ ^[0-9]+$ ]] && [ -d "$D/payload" ] && [ ! -f "$D/entry.sh" ]; then
    payload_changes=$(python3 - "$P" "$D/payload" "$code_commit" <<'PYPAYLOAD'
import re, subprocess, sys
from pathlib import Path
project, payload, commit = sys.argv[1:]
def git(*args): return subprocess.check_output(['git', '-C', project, *args])
tracked = set(git('ls-tree', '-r', '--name-only', '-z', commit).decode().split('\0'))
name = re.escape(Path(payload).parent.name)
asset_extensions = {'.parquet', '.npy', '.obj', '.mtl', '.png', '.urdf', '.usda'}
def run_data(relative):
    seed = r'data/brainstorm6/PROD-READY/seed-inputs/' + name + '/'
    inputs = 'runs/' + name + '/input-assets/'
    if re.fullmatch(seed + r'manifest\.json|' + inputs + r'binding\.json', relative): return True
    if re.fullmatch(inputs + r'roster/meta/(?:info\.json|episodes_metadata\.jsonl)', relative): return True
    location = (seed + r'files/data/work/tier[123]/.+|' +
                inputs + r'(?:ep[0-9]+/)?assets/human_motion_data/whole_body/.+')
    return bool(re.fullmatch(location, relative)) and Path(relative).suffix in asset_extensions
changed = 0
for path in Path(payload).rglob('*'):
    relative = path.relative_to(payload).as_posix()
    if path.is_symlink(): changed += 1
    elif path.is_file():
        changed += (path.read_bytes() != git('show', commit + ':' + relative) if relative in tracked else not run_data(relative))
print(changed)
PYPAYLOAD
)
    # Dirty training is allowed; exports require changed_files=0 after the payload overlay too.
    changed_files=$((changed_files + payload_changes))
fi
code_record="$code_commit changed_files=$changed_files"
echo "$(date -u +%FT%TZ) $id $code_record" >> "$D/launch-code.log"
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
            if read -r armed_instance armed_pid 2>/dev/null < /workspace/$name.deadman.ready &&
               [ \"\$armed_instance\" = $id ] && kill -0 \"\$armed_pid\" 2>/dev/null; then exit 0; fi
            sleep .1
        done
        echo 'box deadman failed to arm; no writer started' >&2; exit 1"
}
# An SSH acknowledgement can be lost after the box starts. Preserve that writer and its log on retry.
# A dead pid group with only "box start" is an interrupted writer (e.g. container restart): start one, appending its log.
existing=$(ssh "${ssh_opts[@]}" root@"$host" "
    if [ -s /workspace/$name.pid ]; then
        read -r existing_pid < /workspace/$name.pid
        case \"\$existing_pid\" in ''|*[!0-9]*) exit 1;; esac
        if kill -0 \"\$existing_pid\" 2>/dev/null || kill -0 -- -\$existing_pid 2>/dev/null || grep -qE '^box (done|failed) ' /workspace/$name.log; then echo started; fi
    fi
    true" | tail -1)
if [ "$existing" = started ]; then ensure_deadman; echo "box already launched: $name; original writer retained"; exit 0; fi
ssh "${ssh_opts[@]}" root@"$host" "mkdir -p /workspace/v2d-track-2/data/denoise_eval" 2>&1 | grep -v "Welcome\|Have fun" || true
# A failed or partial upload cannot retain an earlier full-project identity.
ssh "${ssh_opts[@]}" root@"$host" "rm -f /workspace/v2d-track-2/LAUNCH_COMMIT"
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
        --exclude '__pycache__' --exclude .git --exclude LAUNCH_COMMIT "$P/" root@"$host":/workspace/v2d-track-2/
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
if [ ! -f "$D/entry.sh" ]; then
    ssh "${ssh_opts[@]}" root@"$host" "printf '%s\n' '$code_record' > /workspace/v2d-track-2/LAUNCH_COMMIT.tmp && mv /workspace/v2d-track-2/LAUNCH_COMMIT.tmp /workspace/v2d-track-2/LAUNCH_COMMIT"
fi
ensure_deadman
[ "${VAST_GUARDED_LAUNCH:-}" != 1 ] || [ ! -f "$D/STOPPED" ] || { echo "watchdog stopped this instance before launch" >&2; exit 1; }
ssh "${ssh_opts[@]}" root@"$host" "setsid nohup bash /workspace/$name.sh >> /workspace/$name.log 2>&1 < /dev/null & echo \$! > /workspace/$name.pid; echo launched" 2>&1 | grep -v "Welcome\|Have fun"
