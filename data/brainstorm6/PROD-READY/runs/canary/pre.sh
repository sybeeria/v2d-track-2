set -euo pipefail
archive="runs/canary/frozen-inputs"
if [ -f "$archive/manifest.json" ]; then .venv/bin/python -B scripts/freeze_inputs.py restore --archive "$archive"; fi
if [ -f runs/canary/frozen-tier2/manifest.json ]; then .venv/bin/python -B scripts/freeze_inputs.py restore --archive runs/canary/frozen-tier2; fi
.venv/bin/python -B data/brainstorm6/PROD-READY/generate_jobs.py --production-only --verify
.venv/bin/python -B scripts/freeze_inputs.py capture --manifest data/brainstorm6/PROD-READY/generated/tier1/manifest.json --work data/work/tier1 --archive "$archive"
export MANIFEST="$PWD/data/brainstorm6/PROD-READY/generated/tier1/manifest.json"
.venv/bin/python - <<'DISK'
import shutil
free=shutil.disk_usage(".").free
assert free >= 115_000_000_000, f"production disk admission failed: {free} bytes free"
DISK
.venv/bin/python -B scripts/freeze_inputs.py capture --manifest data/brainstorm6/PROD-READY/generated/tier2/manifest.json --work data/work/tier2 --archive runs/canary/frozen-tier2
