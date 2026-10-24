#!/usr/bin/env bash
# Usage: scripts/vast/start_run.sh RUN_DIR INSTANCE_ID CAP_USD [CREDIT_FLOOR_USD]
# One paid run: launch.sh, then its watchdog detached in its own session (caffeinate keeps the Mac awake), PID in
# RUN_DIR/controller-watchdog.pid and output in RUN_DIR/controller.log, exactly as MORNING.md arms it by hand.
# Rerun after a launch failure (e.g. container not ready yet); launch.sh refuses a second live owner of RUN_DIR.
set -euo pipefail
D=$(cd "${1:?run dir}" && pwd) id=${2:?instance id} cap=${3:?cap in USD} floor=${4:-1}
P=$(cd "$(dirname "$0")/../.." && pwd)
cd "$P"
bash scripts/vast/launch.sh "$D" "$id"
python3 -I - "$D" "$cap" "$floor" <<'PY'
import shutil, subprocess, sys
from pathlib import Path
d, cap, floor = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
awake = ['caffeinate', '-i'] if shutil.which('caffeinate') else []
with (d / 'controller.log').open('a') as log:
    p = subprocess.Popen([*awake, 'bash', 'scripts/vast/watchdog.sh', str(d), cap, floor],
                         stdout=log, stderr=log, start_new_session=True)
(d / 'controller-watchdog.pid').write_text(f'{p.pid}\n')
print(f'watchdog {p.pid}: cap ${cap}, credit floor ${floor}')
PY
