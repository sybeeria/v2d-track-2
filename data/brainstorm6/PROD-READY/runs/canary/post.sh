set -euo pipefail
.venv/bin/python -B scripts/production_repeats.py prepare --jobs /workspace/canary.jobs --manifest "$MANIFEST" --source runs/canary --tier-root data/challenge/track_2/tier_1_multiview_caption --tier 1 --k 3 --out runs/canary/qualification
.venv/bin/python -B scripts/production_repeats.py run --gpus "$(nvidia-smi -L | wc -l)" --out runs/canary/qualification
.venv/bin/python -B scripts/production_repeats.py qualify --out runs/canary/qualification
