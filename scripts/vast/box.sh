#!/usr/bin/env bash
# Runs on the GPU box as /workspace/<run>.sh (installed by scripts/vast/launch.sh): bootstrap, rebuild the
# non-challenge screen with this host's paths, then run /workspace/<run>.jobs through scripts/test_day.sh into
# runs/<run>. The last line is "box done", which scripts/vast/watchdog.sh waits for.
set -euo pipefail
name=$(basename "$0" .sh)
exec 9> "/workspace/$name.lock"
flock -n 9 || { echo "box already running: $name" >&2; exit 2; }
echo "$$" > "/workspace/$name.pid"
cd /workspace/v2d-track-2
date -u +"box start %FT%TZ"
BOOTSTRAP_TIERS=" " bash scripts/gpu_bootstrap.sh
date -u +"bootstrap done %FT%TZ"
.venv/bin/python scripts/make_tier2_screen.py --jobs 6
date -u +"screen done %FT%TZ"
(while sleep 60; do
    echo "$(date -u +%T) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"
done) > /workspace/gpu_samples.log 2>&1 &
sampler=$!
# The sampler shares this script's command line, so it must not outlive it: the watchdog would see a live box script.
trap 'kill $sampler 2>/dev/null' EXIT
# Optional per-run hooks from RUN_DIR: pre.sh is sourced (so it can export MANIFEST), post.sh runs after the jobs.
[ ! -f "/workspace/$name.pre.sh" ] || source "/workspace/$name.pre.sh"
failed=0
JOBS_FILE=/workspace/$name.jobs scripts/test_day.sh "$(nvidia-smi -L | wc -l)" "runs/$name" || failed=1
if [ -f "/workspace/$name.post.sh" ]; then bash "/workspace/$name.post.sh" || failed=1; fi
if [ "$failed" -ne 0 ]; then date -u +"box failed %FT%TZ"; exit 1; fi
date -u +"box done %FT%TZ"
