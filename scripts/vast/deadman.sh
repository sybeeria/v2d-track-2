#!/usr/bin/env bash
# Usage (on the GPU box, armed by scripts/vast/watchdog.sh): deadman.sh NAME
# Box-side backstop for the Mac watchdog, which stops enforcing anything when the Mac sleeps or its Vast session
# expires. /workspace/NAME.deadline holds "DEADLINE_EPOCH GRACE_SECONDS", rewritten at the start of every watchdog pull
# (the deadline becomes "now" when Vast refuses the watchdog's destroy request). Every
# minute: once the deadline has passed, or once /workspace/NAME.sh is not running and no watchdog cycle has rewritten
# the deadline for GRACE_SECONDS (0 = never), it asks Vast to stop this instance with the per-instance
# CONTAINER_API_KEY. A live watchdog may still be pulling a finished box, so idleness alone never stops it.
# Stop, not destroy: unpulled policies stay on disk.
set -uo pipefail
name=${1:?name}
ws=${DEADMAN_WORKSPACE:-/workspace} interval=${DEADMAN_INTERVAL:-60} api=${VAST_API_URL:-https://console.vast.ai/api/v0}
exec 8> "$ws/$name.deadman.lock"
flock -n 8 || exit 0
log() { echo "$(date -u +%FT%TZ) $*"; }
# Vast sets these for the container's first process; an ssh-started shell may not inherit them.
from_init() { tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1; }
key=${CONTAINER_API_KEY:-$(from_init CONTAINER_API_KEY)} id=${CONTAINER_ID:-$(from_init CONTAINER_ID)}
case "$id" in ''|*[!0-9]*) log "no CONTAINER_ID; deadman disarmed"; exit 1;; esac
[ -n "$key" ] || { log "no CONTAINER_API_KEY; deadman disarmed"; exit 1; }
stop() {
    DEADMAN_KEY=$key python3 -I - "$api/instances/$id/" <<'PYSTOP'
import json, os, sys, urllib.request
request = urllib.request.Request(sys.argv[1], method='PUT', data=json.dumps({'state': 'stopped'}).encode(),
                                 headers={'Authorization': 'Bearer ' + os.environ['DEADMAN_KEY'],
                                          'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=30) as response:
    body = json.loads(response.read() or b'{}')
sys.exit(0 if body.get('success') is True else 1)  # the CLI also counts only an explicit success

PYSTOP
}
trap 'rm -f "$ws/$name.deadman.ready"' EXIT
echo "$id $$" > "$ws/$name.deadman.ready.tmp" && mv "$ws/$name.deadman.ready.tmp" "$ws/$name.deadman.ready"
log "armed for instance $id"
while true; do
    now=$(date +%s) reason=
    deadline= grace=
    read -r deadline grace < "$ws/$name.deadline" 2>/dev/null
    case "$deadline" in ''|*[!0-9]*) deadline=;; esac
    case "$grace" in ''|*[!0-9]*) grace=0;; esac
    armed=$(date -r "$ws/$name.deadline" +%s 2>/dev/null || echo "$now")
    if [ -n "$deadline" ] && [ "$now" -ge "$deadline" ]; then reason="deadline $deadline passed"
    elif [ "$grace" -gt 0 ] && [ $((now - armed)) -ge "$grace" ] && ! pgrep -f "$ws/$name[.]sh" > /dev/null; then
        reason="box script not running and watchdog silent for ${grace}s"
    fi
    if [ -n "$reason" ]; then
        if stop; then log "stop requested ($reason)"; exit 0; fi
        log "stop failed ($reason); retrying"
    fi
    sleep "$interval"
done
