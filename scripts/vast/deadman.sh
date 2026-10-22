#!/usr/bin/env bash
# Usage (on the GPU box, armed by scripts/vast/watchdog.sh): deadman.sh NAME
# Box-side backstop for the Mac watchdog, which stops enforcing anything when the Mac sleeps or its Vast session
# expires. /workspace/NAME.deadline holds "DEADLINE_EPOCH GRACE_SECONDS", rewritten by every watchdog cycle. Every
# minute: once the deadline has passed, or once /workspace/NAME.sh has not been running for GRACE_SECONDS (0 = never),
# it asks Vast to stop this instance with the per-instance CONTAINER_API_KEY. Stop, not destroy: unpulled policies
# stay on disk.
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
sys.exit(0 if body.get('success', True) else 1)
PYSTOP
}
log "armed for instance $id"
idle_since=
while true; do
    now=$(date +%s) reason=
    deadline= grace=
    read -r deadline grace < "$ws/$name.deadline" 2>/dev/null
    case "$deadline" in ''|*[!0-9]*) deadline=;; esac
    case "$grace" in ''|*[!0-9]*) grace=0;; esac
    if pgrep -f "$ws/$name[.]sh" > /dev/null; then idle_since=; else idle_since=${idle_since:-$now}; fi
    if [ -n "$deadline" ] && [ "$now" -ge "$deadline" ]; then reason="deadline $deadline passed"
    elif [ "$grace" -gt 0 ] && [ -n "$idle_since" ] && [ $((now - idle_since)) -ge "$grace" ]; then
        reason="box script not running for ${grace}s"
    fi
    if [ -n "$reason" ]; then
        if stop; then log "stop requested ($reason)"; exit 0; fi
        log "stop failed ($reason); retrying"
    fi
    sleep "$interval"
done
