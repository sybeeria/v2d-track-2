set -euo pipefail
.venv/bin/python -B scripts/production_repeats.py prepare --ep18 --jobs /workspace/ep18-pilot-d1.jobs --manifest "$MANIFEST" --source runs/ep18-pilot-d1 --tier-root runs/ep18-pilot-d1/input-assets --tier 3 --k 3 --out runs/ep18-pilot-d1/qualification || true
.venv/bin/python -B scripts/production_repeats.py run --gpus "$(nvidia-smi -L | wc -l)" --out runs/ep18-pilot-d1/qualification || true
.venv/bin/python -B scripts/production_repeats.py qualify --out runs/ep18-pilot-d1/qualification || true
.venv/bin/python -B scripts/ep18_pilot.py decide --qualification runs/ep18-pilot-d1/qualification/qualification.json --output runs/ep18-pilot-d1/decision.json
