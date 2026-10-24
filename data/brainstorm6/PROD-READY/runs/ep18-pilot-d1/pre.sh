set -euo pipefail
if [ -e runs/ep18-pilot-d1/decision.json ]; then echo "pilot already decided; retain attempts" >&2; exit 1; fi
.venv/bin/python -B scripts/ep18_pilot.py admit --binding runs/ep18-pilot-d1/input-assets/binding.json --output runs/ep18-pilot-d1/manifest.json
export MANIFEST="$PWD/runs/ep18-pilot-d1/manifest.json"
