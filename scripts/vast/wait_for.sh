#!/usr/bin/env bash
# Usage: scripts/vast/wait_for.sh RUN_DIR REGEX   Polls the box log each minute; exits 0 when REGEX appears in
# /workspace/<run>.log, 1 when the box script is gone without it. Prints the log tail either way.
set -uo pipefail
D=$(cd "${1:?run dir}" && pwd) pattern=${2:?regex}
name=$(basename "$D")
read -r id host port < "$D/host"
ssh_opts=(-n -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i "$HOME/.ssh/id_ed25519" -p "$port")
while true; do
    state=$(ssh "${ssh_opts[@]}" root@"$host" "
        if grep -qE '$pattern' /workspace/$name.log; then echo hit
        elif pgrep -f '[/]workspace/$name.sh' > /dev/null; then echo running
        else echo dead; fi" 2>/dev/null | tail -1)
    if [ "$state" = hit ] || [ "$state" = dead ]; then
        ssh "${ssh_opts[@]}" root@"$host" "grep -v '^ *$' /workspace/$name.log | tail -25" 2>/dev/null | grep -v "Welcome\|Have fun"
        echo "STATE=$state"
        [ "$state" = hit ]; exit
    fi
    sleep 60
done
